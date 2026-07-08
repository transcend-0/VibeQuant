"""factor_rotation: the factor->strategy bridge."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json

from src.data_sources.etf_pool import POOL_SYMBOLS  # noqa: E402
from src.dsl import TaskSpec  # noqa: E402
from src.intent import parse_prompt  # noqa: E402
from src.runner import run_task  # noqa: E402
from src.strategies import REGISTRY  # noqa: E402
from src.strategies.factor_rotation import DEFAULT_SOURCE as ROTATION_SOURCE  # noqa: E402
from src.adapters.akquant_engine import _DateKeyedScores  # noqa: E402


def test_factor_scores_tolerates_non_string_date_lookup():
    # regression: a real generated strategy did
    # `FACTOR_SCORES.get(trading_date, {})` with the raw date object
    # akquant passes to on_daily_rebalance, instead of the documented
    # `str(trading_date)[:10]`. A plain dict silently returns the default
    # ({}) on that type mismatch -- no exception, so the strategy quietly
    # never rebalanced and returned 0% with 0 trades. FACTOR_SCORES must
    # match regardless of what key shape the caller looks it up with.
    import datetime

    scores = _DateKeyedScores()
    scores["2024-01-02"] = {"A": 1.0, "B": -1.0}
    assert scores.get(datetime.date(2024, 1, 2)) == {"A": 1.0, "B": -1.0}
    assert scores.get("2024-01-02") == {"A": 1.0, "B": -1.0}
    assert scores.get(datetime.date(2024, 1, 3)) is None
    assert datetime.date(2024, 1, 2) in scores
    assert scores[datetime.date(2024, 1, 2)] == {"A": 1.0, "B": -1.0}


def test_skeleton_registered_with_expressions():
    skeleton = REGISTRY["factor_rotation"]
    assert skeleton.params["expressions"]
    assert "FACTOR_SCORES" in skeleton.source
    assert "top_k" in skeleton.source


def test_parse_prompt_fills_in_pool24_symbols(fake_llm):
    # the model is told NOT to guess the 24 codes and just say universe:
    # pool24 with symbols: ["DEMO"] — parse_prompt must inject the real
    # pool itself (regression: this used to silently leave ["DEMO"]).
    fake_llm(lambda user, system: json.dumps({
        "task_yaml": """
name: ma-cross-pool24
kind: strategy
data:
  source: etf
  universe: pool24
  symbols: ["DEMO"]
strategy:
  name: custom
  params:
    source: "class Strategy(BaseStrategy):\\n    def on_bar(self, bar):\\n        pass\\n"
""",
        "clarifications": [],
        "recognized": True,
    }))
    p = parse_prompt("在精选24ETF池上做双均线策略")
    assert p.spec.data.symbols == list(POOL_SYMBOLS)
    assert len(p.spec.data.symbols) == 24


def test_parse_prompt_rejects_bare_labels_as_factor_expressions(fake_llm):
    # strategy.params.expressions must be real "Name = Expr" factor
    # expressions, not bare labels like "momentum_20" — regression: this
    # used to only fail at backtest time ("Unknown function: momentum"),
    # not at parse time where query_structured could retry.
    calls = {"n": 0}
    source = "class Strategy(BaseStrategy):\\n    def on_daily_rebalance(self, trading_date, timestamp):\\n        pass\\n"

    def responder(user, system):
        calls["n"] += 1
        exprs = (
            '["momentum_20", "low_vol_20"]' if calls["n"] == 1
            else '["Mom20 = Delta(Close, 20) / Delay(Close, 20)"]'
        )
        return json.dumps({
            "task_yaml": f"""
name: rotation-pool24
kind: strategy
data: {{source: etf, universe: pool24, symbols: ["DEMO"]}}
strategy:
  name: factor_rotation
  params:
    expressions: {exprs}
    source: "{source}"
""",
            "clarifications": [],
            "recognized": True,
        })

    fake_llm(responder)
    p = parse_prompt("用内置24ETF池构建多因子轮动策略")
    assert calls["n"] == 2  # first (bad) reply was retried, not accepted
    assert all("=" in e for e in p.spec.strategy.params["expressions"])


def test_nl_maps_rotation_to_template(fake_llm):
    # parse_prompt is LLM-backed (src/intent.py); script the reply the way a
    # correctly-behaving LLM would answer this prompt, and check the
    # plumbing turns it into a valid TaskSpec.
    symbols_yaml = ", ".join(f'"{s}"' for s in POOL_SYMBOLS)
    fake_llm(lambda user, system: json.dumps({
        "task_yaml": f"""
name: rotation-pool24
kind: strategy
data:
  source: etf
  universe: pool24
  symbols: [{symbols_yaml}]
strategy:
  name: factor_rotation
  params:
    expressions: ["Mom20 = Delta(Close, 20) / Delay(Close, 20)"]
    source: "class Strategy(BaseStrategy):\\n    def on_daily_rebalance(self, trading_date, timestamp):\\n        pass\\n"
""",
        "clarifications": ["Using the curated 24-ETF pool as the universe."],
        "recognized": True,
    }))
    p = parse_prompt("用 内置24 ETF 构建一个多因子轮动策略")
    assert p.spec.strategy.name == "factor_rotation"
    assert p.spec.data.universe == "pool24"
    assert len(p.spec.data.symbols) == 24


def test_rotation_backtest_on_synthetic(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBEQUANT_DATA", str(tmp_path / "data"))
    spec = TaskSpec.from_dict(
        {
            "name": "rotation-synth",
            "kind": "strategy",
            "data": {
                "source": "synthetic",
                "symbols": [f"S{i}" for i in range(8)],
                "start": "2022-01-01",
                "end": "2023-12-31",
            },
            "strategy": {
                "name": "factor_rotation",
                "params": {
                    "expressions": ["Mom20 = Delta(Close, 20) / Delay(Close, 20)"],
                    "source": ROTATION_SOURCE,
                },
            },
        }
    )
    result = run_task(spec, workspace=tmp_path)
    assert result.ok, result.error
    assert result.num_trades > 10  # it actually rotates
    assert result.metrics["total_return_pct"] is not None
    assert result.validation.get("overfit_risk") in ("low", "medium", "high")


def test_rotation_two_phase_rebalance_avoids_cash_rejections(tmp_path, monkeypatch):
    """Regression: same-day sell-then-buy sequencing used to cause dozens
    of cash-rejected buy orders; the two-phase rebalance (exit day t, enter
    day t+1) fixes it -- still true now that the logic is editable source."""
    monkeypatch.setenv("VIBEQUANT_DATA", str(tmp_path / "data"))
    spec = TaskSpec.from_dict(
        {
            "name": "rotation-rejections",
            "kind": "strategy",
            "data": {
                "source": "synthetic",
                "symbols": [f"S{i}" for i in range(10)],
                "start": "2022-01-01",
                "end": "2023-12-31",
            },
            "strategy": {
                "name": "factor_rotation",
                "params": {
                    "expressions": ["Mom20 = Delta(Close, 20) / Delay(Close, 20)"],
                    "source": ROTATION_SOURCE,
                },
            },
        }
    )
    result = run_task(spec, workspace=tmp_path)
    assert result.ok, result.error
