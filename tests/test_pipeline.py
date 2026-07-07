"""End-to-end and unit tests for the VibeQuant minimal loop."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json

from src.dsl import DSLError, TaskSpec  # noqa: E402
from src.intent import IntentError, parse_prompt  # noqa: E402
from src.planner import make_plan  # noqa: E402
from src.risk import RiskGateError, pre_run_gate  # noqa: E402
from src.runner import run_task  # noqa: E402
from src.strategies import REGISTRY, build_signal  # noqa: E402


# ------------------------------------------------------------------ DSL
def test_dsl_yaml_roundtrip():
    spec = TaskSpec.from_yaml((ROOT / "tasks" / "ma_cross_demo.yaml").read_text())
    clone = TaskSpec.from_yaml(spec.to_yaml())
    assert clone.strategy.name == "custom"
    assert clone.strategy.params["warmup"] == 21
    assert "def signal(" in clone.strategy.params["source"]
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
    # no fixed ma_cross template anymore (round 14) -- a 5/20 MA crossover
    # request now comes back as strategy.name="custom" with hand-authored
    # Python, same as any other rule-based request.
    fake_llm(lambda user, system: _llm_task_reply("""
name: ma-cross-600000
kind: strategy
data:
  source: stock
  symbols: ["600000"]
strategy:
  name: custom
  params:
    warmup: 21
    source: |
      def signal(closes, position):
          if len(closes) < 21:
              return None
          fast = sum(closes[-5:]) / 5
          slow = sum(closes[-20:]) / 20
          return 1.0 if fast > slow else 0.0
execution:
  initial_cash: 1000000
report:
  language: zh
"""))
    parsed = parse_prompt("在 600000 上做 5/20 双均线策略回测，资金100万")
    spec = parsed.spec
    assert spec.strategy.name == "custom"
    assert "def signal(" in spec.strategy.params["source"]
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
  name: custom
  params:
    warmup: 15
    source: |
      def signal(closes, position):
          return None
report:
  language: en
"""))
    parsed = parse_prompt("RSI oversold rebound on DEMO from 2022-01-01 to 2023-12-31")
    spec = parsed.spec
    assert spec.strategy.name == "custom"
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
strategy: {name: custom, params: {source: "def signal(closes, position):\\n    return None\\n"}}
""",
        clarifications=["No strategy detail given; defaulting to a no-op signal on DEMO."],
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
strategy: {name: custom, params: {source: "def signal(closes, position):\\n    return 1.0 if position <= 0 else None\\n"}}
""")

    fake_llm(responder)
    parsed = parse_prompt("just buy and hold DEMO")
    assert calls["n"] == 3
    assert parsed.spec.strategy.name == "custom"


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
def test_all_templates_build_and_signal():
    closes = [100 + i * 0.5 for i in range(60)]
    for name in REGISTRY:
        if name == "factor_rotation":  # cross-sectional; runs via the adapter
            continue
        fn = build_signal(name, {})
        result = fn(closes, 0.0)
        assert result is None or 0.0 <= result <= 1.0


def test_ma_cross_golden_cross_fires():
    # there's no fixed ma_cross template anymore (round 14: only
    # factor_rotation + custom remain) -- this checks the same crossover
    # logic works when hand-written as a "custom" signal.
    source = """def signal(closes, position):
    fast, slow = 2, 4
    if len(closes) < slow + 1:
        return None
    fast_now = sum(closes[-fast:]) / fast
    slow_now = sum(closes[-slow:]) / slow
    fast_prev = sum(closes[-fast-1:-1]) / fast
    slow_prev = sum(closes[-slow-1:-1]) / slow
    if fast_prev <= slow_prev and fast_now > slow_now:
        return 1.0
    if fast_prev >= slow_prev and fast_now < slow_now:
        return 0.0
    return None
"""
    fn = build_signal("custom", {"source": source})
    closes = [10, 9, 8, 7, 6, 5, 6, 14]  # sharp reversal -> golden cross
    assert fn(closes, 0.0) == 1.0


def test_custom_strategy_runs_llm_authored_source():
    # strategy.name="custom" execs LLM-authored Python directly (accepted,
    # documented risk — see src/strategies/custom.py). This checks the
    # plumbing: source -> compiled signal() -> normal SignalFn contract.
    source = """def signal(closes, position):
    if len(closes) < 3:
        return None
    if closes[-1] > closes[-2] > closes[-3]:
        return 1.0
    return 0.0
"""
    fn = build_signal("custom", {"source": source, "warmup": 3})
    assert fn([1.0, 2.0, 3.0], 0.0) == 1.0
    assert fn([3.0, 2.0, 1.0], 0.0) == 0.0


def test_custom_strategy_requires_signal_function():
    with pytest.raises(KeyError):
        build_signal("custom", {"source": "x = 1\n"})


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


def test_plan_is_transparent():
    spec = TaskSpec()
    plan = make_plan(spec)
    tools = [step.tool for step in plan.steps]
    assert tools == [
        "risk_gate", "load_data", "backtest", "risk_assess", "validate",
        "report", "memorize",
    ]
