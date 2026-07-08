"""Tests for the third round: factor ops, LLM config, deployments, registry."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dsl import DSLError, TaskSpec  # noqa: E402
from src.factors.analysis import apply_factor_ops  # noqa: E402
from src.live.deploy import (  # noqa: E402
    DeployError,
    compute_signals,
    create_deployment,
    list_deployments,
)
from src.adapters.akquant_factor import known_expression_functions  # noqa: E402
from src.research.llm_ideas import _expression_valid, _strip_fence  # noqa: E402


# ----------------------------------------------------------- factor ops
def _panel():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 4 + ["2024-01-02"] * 4),
            "symbol": ["A", "B", "C", "D"] * 2,
            "f": [1.0, 2.0, 3.0, 100.0, 4.0, 3.0, 2.0, 1.0],
        }
    )


def test_ops_truncation_and_demean():
    out = apply_factor_ops(_panel(), ["f"], truncation=0.25, neutralization="demean")
    day1 = out[out.date == "2024-01-01"]["f"]
    assert abs(day1.sum()) < 1e-9  # demeaned
    assert day1.max() < 100  # outlier truncated


def test_ops_rank_neutralization():
    out = apply_factor_ops(_panel(), ["f"], neutralization="rank")
    day1 = sorted(out[out.date == "2024-01-01"]["f"])
    assert day1 == [0.25, 0.5, 0.75, 1.0]


def test_ops_zscore():
    out = apply_factor_ops(_panel(), ["f"], neutralization="zscore")
    day2 = out[out.date == "2024-01-02"]["f"]
    assert abs(day2.mean()) < 1e-9
    assert abs(day2.std(ddof=1) - 1.0) < 1e-9


def test_ops_linear_decay():
    out = apply_factor_ops(_panel(), ["f"], decay=2)
    got = out[(out.symbol == "A") & (out.date == "2024-01-02")]["f"].iloc[0]
    assert abs(got - (4.0 * 2 + 1.0 * 1) / 3) < 1e-9  # weights 2,1 newest-first


def test_factor_spec_validates_ops():
    base = {
        "kind": "factor",
        "data": {"symbols": ["A", "B"]},
        "factor": {"expressions": ["Delta(Close, 1)"]},
    }
    spec = TaskSpec.from_dict(base)
    assert spec.factor.neutralization == "none"
    bad = dict(base, factor={"expressions": ["x"], "neutralization": "sector"})
    with pytest.raises(DSLError):
        TaskSpec.from_dict(bad)
    bad2 = dict(base, factor={"expressions": ["x"], "truncation": 0.7})
    with pytest.raises(DSLError):
        TaskSpec.from_dict(bad2)


# ------------------------------------------------------------- llm bits
def test_llm_expression_validator():
    assert _expression_valid("Mom = Delta(Close, 20) / Delay(Close, 20)")
    assert _expression_valid("Rank(Ts_Corr(Close, Volume, 10))")
    assert not _expression_valid("X = __import__('os').system('rm')")
    assert not _expression_valid("PE = Earnings / Price")  # unknown columns


def test_llm_expression_validator_uses_akquants_full_grammar():
    # the allow-list is sourced from akquant's real OPS_MAP (adapters layer),
    # not a hand-typed subset — aliases beyond the original hardcoded list
    # must be accepted too.
    assert "Standardize" in known_expression_functions()
    assert _expression_valid("Z = Standardize(Close)")
    assert _expression_valid("M = Mean(Close, 10)")  # alias for Ts_Mean


def test_llm_strip_fence():
    assert _strip_fence('```json\n[{"a":1}]\n```') == '[{"a":1}]'
    assert _strip_fence('[{"a":1}]') == '[{"a":1}]'


def test_llm_config_roundtrip(tmp_path, monkeypatch):
    import src.llm.client as llm_client

    monkeypatch.setattr(llm_client, "config_dir", lambda: tmp_path)
    assert llm_client.load_llm_config() is None
    llm_client.save_llm_config("gpt-5-mini", "sk-test", "https://x/v1")
    conf = llm_client.load_llm_config()
    assert conf == {
        "model": "gpt-5-mini",
        "api_key": "sk-test",
        "base_url": "https://x/v1",
    }


# ----------------------------------------------------------- deployment
def test_deployment_requires_strategy_and_real_data(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBEQUANT_WORKSPACE", str(tmp_path))
    with pytest.raises(DeployError):  # factor task
        create_deployment(
            "kind: factor\n"
            "data: {symbols: [a, b]}\n"
            "factor: {expressions: ['Delta(Close,1)']}\n"
        )
    with pytest.raises(DeployError):  # synthetic data
        create_deployment(
            "kind: strategy\ndata: {source: synthetic, symbols: [DEMO]}\n"
        )
    dep = create_deployment(
        "kind: strategy\n"
        "name: t\n"
        "data: {source: etf, symbols: [510300], start: '2023-01-01'}\n"
        "strategy: {name: custom, params: {source: \"class Strategy(BaseStrategy):\\n    def on_bar(self, bar):\\n        pass\\n\"}}\n",
        email_to="a@b.c",
        run_at="16:30",
    )
    assert dep.id.startswith("dep-")
    assert [d.id for d in list_deployments()] == [dep.id]


def test_compute_signals_actions():
    # synthetic source keeps this offline; compute_signals runs the real
    # backtest through today and reads the resulting position history.
    spec = TaskSpec.from_dict(
        {
            "kind": "strategy",
            "data": {
                "source": "synthetic",
                "symbols": ["DEMO"],
                "start": "2023-01-01",
                "end": "2024-06-28",
            },
            "strategy": {
                "name": "custom",
                "params": {
                    "source": "class Strategy(BaseStrategy):\n"
                    "    def on_bar(self, bar):\n"
                    "        if self.get_position(bar.symbol) <= 0:\n"
                    "            self.order_target_percent(target_percent=0.95, symbol=bar.symbol)\n",
                },
            },
        }
    )
    signals = compute_signals(spec)
    assert len(signals) == 1
    sig = signals[0]
    assert sig["action"] in ("BUY", "SELL", "HOLD", "STAY_FLAT")
    assert sig["position"] > 0  # buy-and-hold always ends long
    assert sig["last_close"] > 0


# ------------------------------------------------------ factor registry
def test_factor_registry_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBEQUANT_DATA", str(tmp_path))
    from src.factors.registry import list_factors, load_factor_panel, register_factor

    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "symbol": ["A", "A"],
            "MyF": [0.1, 0.2],
        }
    )
    register_factor(
        "MyF", "Delta(Close, 1)", panel, "run-x",
        spec_summary={"symbols": ["A"]}, stats={"ic_mean": 0.05},
    )
    entries = list_factors()
    assert entries and entries[0]["name"] == "MyF"
    loaded = load_factor_panel(entries[0]["file"])
    assert list(loaded["value"]) == [0.1, 0.2]
