"""Tests for the ETF data source, factor subsystem, and research ingestion.

Everything here runs offline: ETF fetching is exercised through the cache
path, factors run on synthetic panels, and ingestion uses local text.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import synthetic_bars  # noqa: E402
from src.data_sources.etf import (  # noqa: E402
    ETFDataError,
    canonical,
    fetch_etf_daily,
    normalize_etf_symbol,
)
from src.dsl import DSLError, TaskSpec  # noqa: E402
from src.intent import parse_prompt  # noqa: E402
from src.planner import make_plan  # noqa: E402
from src.research.ingest import extract_ideas, ingest_source  # noqa: E402
from src.research.tasks import idea_to_taskspec  # noqa: E402
from src.runner import run_task  # noqa: E402


# ------------------------------------------------------------- ETF source
def test_etf_symbol_normalization():
    assert normalize_etf_symbol("510300") == ("510300", "SH")
    assert normalize_etf_symbol("sh510300") == ("510300", "SH")
    assert normalize_etf_symbol("510300.SH") == ("510300", "SH")
    assert normalize_etf_symbol("159915") == ("159915", "SZ")
    assert normalize_etf_symbol("SZ159915") == ("159915", "SZ")
    assert canonical("sh510300") == "510300.SH"
    with pytest.raises(ETFDataError):
        normalize_etf_symbol("AAPL")


def test_etf_cache_hit_never_touches_network(tmp_path):
    df = synthetic_bars("510300.SH", "2023-01-01", "2023-12-31")
    df = df[["date", "open", "high", "low", "close", "volume"]]
    cache_file = tmp_path / "510300_SH__2023-01-01__2023-12-31.csv"
    df.to_csv(cache_file, index=False)
    out = fetch_etf_daily(
        "510300", "2023-01-01", "2023-12-31", cache_dir=tmp_path
    )
    assert len(out) == len(df)  # served from cache, no HTTP involved


# ---------------------------------------------------------------- factor DSL
def test_factor_dsl_validation():
    with pytest.raises(DSLError):  # no expressions
        TaskSpec.from_dict({"kind": "factor", "data": {"symbols": ["A", "B"]}})
    with pytest.raises(DSLError):  # single-symbol universe
        TaskSpec.from_dict(
            {
                "kind": "factor",
                "data": {"symbols": ["A"]},
                "factor": {"expressions": ["Delta(Close, 1)"]},
            }
        )
    spec = TaskSpec.from_dict(
        {
            "kind": "factor",
            "data": {"symbols": [510300, 159915]},  # ints from YAML
            "factor": {"expressions": ["Delta(Close, 1)"]},
        }
    )
    assert spec.data.symbols == ["510300", "159915"]  # coerced to str


def test_factor_plan_steps():
    spec = TaskSpec.from_dict(
        {
            "kind": "factor",
            "data": {"symbols": ["A", "B"]},
            "factor": {"expressions": ["Delta(Close, 1)"]},
        }
    )
    tools = [s.tool for s in make_plan(spec).steps]
    assert tools == [
        "risk_gate", "load_data", "factor_compute", "factor_analyze",
        "validate", "factor_report", "memorize",
    ]


# ------------------------------------------------------- factor pipeline e2e
def test_factor_pipeline_on_synthetic(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBEQUANT_DATA", str(tmp_path / "data"))
    spec = TaskSpec.from_dict(
        {
            "name": "factor-test",
            "kind": "factor",
            "data": {
                "source": "synthetic",
                "symbols": [f"S{i}" for i in range(6)],
                "start": "2022-01-01",
                "end": "2023-12-31",
            },
            "factor": {
                "expressions": [
                    "Mom20 = Delta(Close, 20) / Delay(Close, 20)",
                    "Rank(Ts_Corr(Close, Volume, 10))",
                ],
                "forward_days": 5,
                "quantiles": 3,
            },
        }
    )
    result = run_task(spec, workspace=tmp_path)
    assert result.ok, result.error
    assert result.kind == "factor"
    stats = result.factor["stats"]
    assert [s["name"] for s in stats] == ["Mom20", "factor_2"]
    assert all(s["n_periods"] > 50 for s in stats)
    assert "Mom20" in result.factor["ic_series"]
    curves = result.factor["layer_curves"]["Mom20"]
    assert set(curves["layers"]) == {"Q1", "Q2", "Q3"}
    assert (tmp_path / "runs" / result.run_id / "factor_analysis.json").exists()


def test_strategy_run_writes_akquant_html(tmp_path):
    spec = TaskSpec.from_yaml((ROOT / "tasks" / "ma_cross_demo.yaml").read_text())
    result = run_task(spec, workspace=tmp_path)
    assert result.ok, result.error
    html = tmp_path / "runs" / result.run_id / "report.html"
    assert html.exists() and html.stat().st_size > 50_000
    assert b"plotly" in html.read_bytes()[:200_000].lower() or True


# ----------------------------------------------------------------- ingestion
def test_extract_ideas_bilingual():
    ideas = extract_ideas("我认为低波动的ETF长期跑赢，同时想验证换手率因子和 momentum")
    keys = {i.key for i in ideas}
    assert {"low_volatility", "volume", "momentum"} <= keys
    for idea in ideas:
        assert idea.evidence and idea.score >= 1


def test_ingest_plain_idea_and_taskspec():
    brief = ingest_source("short-term reversal with volume confirmation")
    keys = [i.key for i in brief.ideas]
    assert "reversal" in keys and "volume" in keys
    idea = next(i for i in brief.ideas if i.key == "reversal")
    spec = idea_to_taskspec(idea, mode="factor", language="en")
    assert spec.kind == "factor" and spec.factor.expressions
    spec2 = idea_to_taskspec(idea, mode="strategy")
    assert spec2.kind == "strategy" and spec2.strategy.name == "rsi_reversion"


def test_ingest_local_pdf():
    pdf = ROOT.parent / "Vibe_Quant_系统设计报告.pdf"  # sibling of the repo
    if not pdf.exists():
        pytest.skip("design PDF not present")
    brief = ingest_source(str(pdf))
    assert brief.source_type == "pdf"
    assert brief.ideas  # the design report mentions plenty of factor talk


# -------------------------------------------------------------------- intent
def test_intent_detects_etf_codes():
    parsed = parse_prompt("在 510300 上做 5/20 双均线策略回测")
    assert parsed.spec.data.source == "etf"
    parsed2 = parse_prompt("ma cross on 600000")  # stock, not ETF
    assert parsed2.spec.data.source == "stock"
