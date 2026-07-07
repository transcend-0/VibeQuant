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


def test_template_registered_with_defaults():
    tpl = REGISTRY["factor_rotation"]
    assert tpl.defaults["top_k"] == 5
    assert tpl.defaults["expressions"]


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
  params: {source: "def signal(closes, position):\\n    return None\\n"}
""",
        "clarifications": [],
        "recognized": True,
    }))
    p = parse_prompt("在精选24ETF池上做双均线策略")
    assert p.spec.data.symbols == list(POOL_SYMBOLS)
    assert len(p.spec.data.symbols) == 24


def test_parse_prompt_rejects_bare_labels_as_factor_expressions(fake_llm):
    # factor_rotation.params.expressions must be real "Name = Expr" factor
    # expressions, not bare labels like "momentum_20" — regression: this
    # used to only fail at backtest time ("Unknown function: momentum"),
    # not at parse time where query_structured could retry.
    calls = {"n": 0}

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
                    "top_k": 3,
                    "rebalance_days": 5,
                },
            },
        }
    )
    result = run_task(spec, workspace=tmp_path)
    assert result.ok, result.error
    assert result.num_trades > 10  # it actually rotates
    assert result.metrics["total_return_pct"] is not None
    assert result.validation.get("overfit_risk") in ("low", "medium", "high")
