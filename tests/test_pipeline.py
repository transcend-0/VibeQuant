"""End-to-end and unit tests for the VibeQuant minimal loop."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dsl import DSLError, TaskSpec  # noqa: E402
from src.intent import parse_prompt  # noqa: E402
from src.planner import make_plan  # noqa: E402
from src.risk import RiskGateError, pre_run_gate  # noqa: E402
from src.runner import run_task  # noqa: E402
from src.strategies import REGISTRY, build_signal  # noqa: E402


# ------------------------------------------------------------------ DSL
def test_dsl_yaml_roundtrip():
    spec = TaskSpec.from_yaml((ROOT / "tasks" / "ma_cross_demo.yaml").read_text())
    clone = TaskSpec.from_yaml(spec.to_yaml())
    assert clone.strategy.name == "ma_cross"
    assert clone.strategy.params == {"fast": 5, "slow": 20}
    assert clone.execution.mode == "backtest"


def test_dsl_rejects_unknown_keys():
    with pytest.raises(DSLError):
        TaskSpec.from_dict({"name": "x", "bogus": {}})


def test_dsl_rejects_bad_mode():
    with pytest.raises(DSLError):
        TaskSpec.from_dict({"execution": {"mode": "yolo"}})


# --------------------------------------------------------------- intent
def test_intent_zh_ma_cross():
    parsed = parse_prompt("在 600000 上做 5/20 双均线策略回测，资金100万")
    spec = parsed.spec
    assert spec.strategy.name == "ma_cross"
    assert spec.strategy.params == {"fast": 5, "slow": 20}
    assert spec.data.symbols == ["600000"]
    assert spec.execution.initial_cash == 1_000_000
    assert spec.report.language == "zh"


def test_intent_en_rsi():
    parsed = parse_prompt("RSI oversold rebound on DEMO from 2022-01-01 to 2023-12-31")
    spec = parsed.spec
    assert spec.strategy.name == "rsi_reversion"
    assert spec.data.symbols == ["DEMO"]
    assert spec.data.start == "2022-01-01"
    assert spec.data.end == "2023-12-31"
    assert spec.report.language == "en"


def test_intent_bollinger_not_confused_with_rsi():
    # regression: "reveRSIon" must not trigger the RSI rule
    parsed = parse_prompt("bollinger band reversion on DEMO")
    assert parsed.spec.strategy.name == "bollinger"


def test_intent_defaults_are_disclosed():
    parsed = parse_prompt("do something profitable")
    assert parsed.clarifications  # every guess must be disclosed


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
    fn = build_signal("ma_cross", {"fast": 2, "slow": 4})
    closes = [10, 9, 8, 7, 6, 5, 6, 14]  # sharp reversal -> golden cross
    assert fn(closes, 0.0) == 1.0


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
