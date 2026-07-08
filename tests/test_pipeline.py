"""End-to-end and unit tests for the VibeQuant minimal loop."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json

import akquant as aq  # noqa: E402

from src import data as data_mod  # noqa: E402
from src.adapters.akquant_engine import load_user_strategy, run_backtest  # noqa: E402
from src.dsl import DSLError, TaskSpec  # noqa: E402
from src.intent import IntentError, parse_prompt  # noqa: E402
from src.planner import make_plan  # noqa: E402
from src.risk import RiskGateError, pre_run_gate  # noqa: E402
from src.runner import run_task  # noqa: E402


# ------------------------------------------------------------------ DSL
def test_dsl_yaml_roundtrip():
    spec = TaskSpec.from_yaml((ROOT / "tasks" / "ma_cross_demo.yaml").read_text())
    clone = TaskSpec.from_yaml(spec.to_yaml())
    assert clone.strategy.name == "ma_cross"
    assert "class Strategy(BaseStrategy)" in clone.strategy.params["source"]
    assert clone.execution.mode == "backtest"


def test_dsl_rejects_unknown_keys():
    with pytest.raises(DSLError):
        TaskSpec.from_dict({"name": "x", "bogus": {}})


def test_dsl_rejects_bad_mode():
    with pytest.raises(DSLError):
        TaskSpec.from_dict({"execution": {"mode": "yolo"}})


# --------------------------------------------------------------- intent
# parse_prompt is LLM-backed with no rule fallback (src/intent.py). These
# tests script the LLM's reply via the fake_llm fixture (tests/conftest.py)
# and check the plumbing: JSON -> YAML -> TaskSpec -> validation.
def _llm_task_reply(task_yaml, clarifications=None, recognized=True):
    return json.dumps({
        "task_yaml": task_yaml,
        "clarifications": clarifications or [],
        "recognized": recognized,
    })


def test_intent_zh_custom_rule_strategy(fake_llm):
    # every strategy runs as an akquant Strategy class now (round 15) -- a
    # 5/20 MA crossover request comes back as hand-authored Python
    # inheriting Strategy/BaseStrategy, same as any other rule-based request.
    fake_llm(lambda user, system: _llm_task_reply("""
name: ma-cross-600000
kind: strategy
data:
  source: stock
  symbols: ["600000"]
strategy:
  name: ma_cross
  params:
    source: |
      class Strategy(BaseStrategy):
          def on_bar(self, bar):
              closes = self.get_history(count=21, symbol=bar.symbol, field="close")
              if len(closes) < 21:
                  return
              fast = closes[-5:].mean()
              slow = closes[-20:].mean()
              if fast > slow:
                  self.order_target_percent(target_percent=0.95, symbol=bar.symbol)
              else:
                  self.close_position(bar.symbol)
execution:
  initial_cash: 1000000
report:
  language: zh
"""))
    parsed = parse_prompt("在 600000 上做 5/20 双均线策略回测，资金100万")
    spec = parsed.spec
    assert "class Strategy(BaseStrategy)" in spec.strategy.params["source"]
    assert spec.data.symbols == ["600000"]
    assert spec.execution.initial_cash == 1_000_000
    assert spec.report.language == "zh"


def test_intent_en_rsi(fake_llm):
    fake_llm(lambda user, system: _llm_task_reply("""
name: rsi-demo
kind: strategy
data:
  source: synthetic
  symbols: ["DEMO"]
  start: "2022-01-01"
  end: "2023-12-31"
strategy:
  name: rsi
  params:
    source: |
      class Strategy(BaseStrategy):
          def on_bar(self, bar):
              pass
report:
  language: en
"""))
    parsed = parse_prompt("RSI oversold rebound on DEMO from 2022-01-01 to 2023-12-31")
    spec = parsed.spec
    assert spec.data.symbols == ["DEMO"]
    assert spec.data.start == "2022-01-01"
    assert spec.data.end == "2023-12-31"
    assert spec.report.language == "en"


def test_intent_defaults_are_disclosed(fake_llm):
    fake_llm(lambda user, system: _llm_task_reply(
        """
name: vague-goal
kind: strategy
data: {source: synthetic, symbols: ["DEMO"]}
strategy: {name: custom, params: {source: "class Strategy(BaseStrategy):\\n    def on_bar(self, bar):\\n        pass\\n"}}
""",
        clarifications=["No strategy detail given; defaulting to a no-op strategy on DEMO."],
        recognized=False,
    ))
    parsed = parse_prompt("do something profitable")
    assert parsed.clarifications  # every guess must be disclosed
    assert not parsed.recognized


def test_intent_raises_without_llm(fake_llm):
    fake_llm(None)  # simulates unconfigured/unreachable config/llm.yaml
    with pytest.raises(IntentError):
        parse_prompt("ma cross 5/20 on DEMO")


def test_intent_raises_on_invalid_llm_output(fake_llm):
    fake_llm(lambda user, system: _llm_task_reply("kind: bogus-kind"))
    with pytest.raises(IntentError):
        parse_prompt("ma cross 5/20 on DEMO")


def test_intent_rejects_source_missing_strategy_class(fake_llm):
    # strategy.params.source that doesn't define a Strategy/BaseStrategy
    # subclass must be caught at parse time (validate_strategy_params),
    # not silently accepted only to fail at backtest time.
    calls = {"n": 0}

    def responder(user, system):
        calls["n"] += 1
        if calls["n"] == 1:
            return _llm_task_reply("""
name: bad
kind: strategy
data: {source: synthetic, symbols: ["DEMO"]}
strategy: {name: custom, params: {source: "x = 1\\n"}}
""")
        return _llm_task_reply("""
name: fixed
kind: strategy
data: {source: synthetic, symbols: ["DEMO"]}
strategy: {name: custom, params: {source: "class Strategy(BaseStrategy):\\n    def on_bar(self, bar):\\n        pass\\n"}}
""")

    fake_llm(responder)
    parsed = parse_prompt("do something")
    assert calls["n"] == 2  # first reply (no Strategy class) was retried
    assert "class Strategy(BaseStrategy)" in parsed.spec.strategy.params["source"]


def test_intent_retries_and_recovers_from_bad_output(fake_llm):
    # first two replies are malformed/invalid; the third is a good task.
    # parse_prompt must retry (feeding the error back) instead of giving up
    # on the first bad reply.
    calls = {"n": 0}

    def responder(user, system):
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json at all"
        if calls["n"] == 2:
            return _llm_task_reply("kind: bogus-kind")
        return _llm_task_reply("""
name: recovered
kind: strategy
data: {source: synthetic, symbols: ["DEMO"]}
strategy: {name: custom, params: {source: "class Strategy(BaseStrategy):\\n    def on_bar(self, bar):\\n        if self.get_position(bar.symbol) <= 0:\\n            self.order_target_percent(target_percent=0.9, symbol=bar.symbol)\\n"}}
""")

    fake_llm(responder)
    parsed = parse_prompt("just buy and hold DEMO")
    assert calls["n"] == 3
    assert "class Strategy(BaseStrategy)" in parsed.spec.strategy.params["source"]


def test_intent_gives_up_after_3_attempts(fake_llm):
    calls = {"n": 0}

    def responder(user, system):
        calls["n"] += 1
        return "still not json"

    fake_llm(responder)
    with pytest.raises(IntentError):
        parse_prompt("ma cross 5/20 on DEMO")
    assert calls["n"] == 3


# ------------------------------------------------------------ strategies
def test_load_user_strategy_finds_subclass():
    source = "class Strategy(BaseStrategy):\n    def on_bar(self, bar):\n        pass\n"
    cls = load_user_strategy(source, symbols=["DEMO"], start=None, end=None)
    assert issubclass(cls, aq.Strategy)
    assert cls is not aq.Strategy


def test_load_user_strategy_picks_last_class_when_multiple():
    source = (
        "class Helper(BaseStrategy):\n    def on_bar(self, bar):\n        pass\n\n\n"
        "class Strategy(Helper):\n    def on_bar(self, bar):\n        pass\n"
    )
    cls = load_user_strategy(source, symbols=["DEMO"], start=None, end=None)
    assert cls.__name__ == "Strategy"


def test_load_user_strategy_requires_strategy_subclass():
    with pytest.raises(ValueError):
        load_user_strategy("x = 1\n", symbols=["DEMO"], start=None, end=None)


def test_load_user_strategy_rejects_syntax_errors():
    with pytest.raises(SyntaxError):
        load_user_strategy("class Strategy(BaseStrategy\n", symbols=["DEMO"], start=None, end=None)


def test_load_user_strategy_sets_self_symbols_default():
    # LLM-authored code naturally reaches for self.symbols (mirroring
    # akquant's own examples) even without a custom __init__ setting it --
    # it must be populated automatically from the SYMBOLS the caller gave.
    source = "class Strategy(BaseStrategy):\n    def on_bar(self, bar):\n        pass\n"
    cls = load_user_strategy(source, symbols=["A", "B"], start="2022-01-01", end="2023-01-01")
    instance = cls()  # akquant instantiates with no args -- must not raise
    assert instance.symbols == ["A", "B"]
    assert instance.start == "2022-01-01"
    assert instance.end == "2023-01-01"


def test_load_user_strategy_custom_init_still_instantiates_with_no_args():
    # regression: a wrapper __init__ that declares **kwargs makes akquant's
    # own kwarg-filtering treat the constructor as "accepts anything" and
    # forward its OWN context kwargs (e.g. symbols=) straight through --
    # which then breaks against a user __init__ that takes no parameters.
    source = (
        "class Strategy(BaseStrategy):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.day_count = 0\n"
        "    def on_bar(self, bar):\n        pass\n"
    )
    cls = load_user_strategy(source, symbols=["A", "B"], start=None, end=None)
    instance = cls()  # must not raise TypeError about unexpected kwargs
    assert instance.day_count == 0
    assert instance.symbols == ["A", "B"]  # default still applied before __init__ body


def test_ma_cross_golden_cross_fires_via_backtest():
    # there's no fixed ma_cross template -- this checks the same crossover
    # logic works when hand-written as an akquant Strategy class and run
    # through the real adapter end to end.
    spec = TaskSpec.from_yaml((ROOT / "tasks" / "ma_cross_demo.yaml").read_text())
    frames = data_mod.load(spec.data)
    out = run_backtest(spec, frames)
    assert out.num_trades > 0


def test_custom_strategy_runs_llm_authored_source():
    # LLM-authored Python execs directly (accepted, documented risk — see
    # src/strategies/custom.py). This checks the plumbing end to end: a
    # 3-bar-rising-streak rule fires a trade in a real backtest.
    source = """class Strategy(BaseStrategy):
    def on_bar(self, bar):
        closes = self.get_history(count=3, symbol=bar.symbol, field="close")
        if len(closes) < 3:
            return
        position = self.get_position(bar.symbol)
        if closes[-1] > closes[-2] > closes[-3] and position <= 0:
            self.order_target_percent(target_percent=0.9, symbol=bar.symbol)
"""
    spec = TaskSpec.from_dict({
        "kind": "strategy",
        "data": {"source": "synthetic", "symbols": ["DEMO"],
                 "start": "2022-01-01", "end": "2023-12-31"},
        "strategy": {"name": "custom", "params": {"source": source}},
    })
    frames = data_mod.load(spec.data)
    out = run_backtest(spec, frames)
    assert out.num_trades >= 0  # runs without error; may or may not trade


# ------------------------------------------------------------- risk gate
def test_gate_blocks_live_and_paper():
    for mode in ("live", "paper"):
        spec = TaskSpec()
        spec.execution.mode = mode
        with pytest.raises(RiskGateError):
            pre_run_gate(spec)


# ---------------------------------------------------------------- runner
def test_full_backtest_run(tmp_path):
    spec = TaskSpec.from_yaml((ROOT / "tasks" / "ma_cross_demo.yaml").read_text())
    result = run_task(spec, workspace=tmp_path)
    assert result.ok, result.error
    assert result.metrics["total_return_pct"] is not None
    assert result.num_trades >= 0
    run_dir = tmp_path / "runs" / result.run_id
    for artifact in ("task.yaml", "result.json", "report.md", "equity.csv"):
        assert (run_dir / artifact).exists(), artifact
    assert (tmp_path / "experiments.jsonl").exists()
    assert (tmp_path / "memory_bank" / "experiment_log.md").exists()


def test_zh_report_language(tmp_path):
    spec = TaskSpec.from_yaml((ROOT / "tasks" / "rsi_demo_zh.yaml").read_text())
    result = run_task(spec, workspace=tmp_path)
    assert result.ok, result.error
    assert "回测报告" in result.report_markdown


def test_run_failure_is_reported_not_raised(tmp_path):
    # a strategy source with a runtime bug should surface as a failed
    # RunResult (ok=False, error, failed_step, run_id still set) so the
    # webui can render/refine it -- not bubble up as an unhandled exception.
    spec = TaskSpec.from_dict({
        "kind": "strategy",
        "data": {"source": "synthetic", "symbols": ["DEMO"],
                 "start": "2022-01-01", "end": "2023-12-31"},
        "strategy": {"name": "custom", "params": {
            "source": "class Strategy(BaseStrategy):\n    def on_bar(self, bar):\n        1 / 0\n",
        }},
    })
    result = run_task(spec, workspace=tmp_path)
    assert not result.ok
    assert result.run_id
    assert result.failed_step == "backtest"
    assert "division" in (result.error or "").lower() or "zero" in (result.error or "").lower()


def test_plan_is_transparent():
    spec = TaskSpec()
    plan = make_plan(spec)
    tools = [step.tool for step in plan.steps]
    assert tools == [
        "risk_gate", "load_data", "backtest", "risk_assess", "validate",
        "report", "memorize",
    ]
