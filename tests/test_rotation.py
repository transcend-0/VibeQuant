"""factor_rotation: the factor->strategy bridge."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dsl import TaskSpec  # noqa: E402
from src.intent import parse_prompt  # noqa: E402
from src.runner import run_task  # noqa: E402
from src.strategies import REGISTRY  # noqa: E402


def test_template_registered_with_defaults():
    tpl = REGISTRY["factor_rotation"]
    assert tpl.defaults["top_k"] == 5
    assert tpl.defaults["expressions"]


def test_nl_maps_rotation_to_template():
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
