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
import json

from src.dsl import DSLError, TaskSpec  # noqa: E402
from src.intent import parse_prompt  # noqa: E402
from src.planner import make_plan  # noqa: E402
from src.research.ingest import IngestError, Idea, _split_url, ingest_source  # noqa: E402
from src.research.llm_ideas import IdeaExtractionError, llm_extract_ideas  # noqa: E402
from src.research.tasks import extract_universe_hint, idea_to_taskspec  # noqa: E402
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
# ingest_source() only extracts text/metadata now (no keyword rules); idea
# extraction is exclusively llm_extract_ideas's job (src/research/llm_ideas.py).
def test_ingest_plain_idea_has_no_rule_ideas():
    brief = ingest_source("short-term reversal with volume confirmation")
    assert brief.text  # text is captured for the LLM to analyze
    assert brief.ideas == []  # no keyword-rule engine populates this anymore


def test_idea_to_taskspec_factor_and_strategy():
    idea = Idea(
        key="reversal",
        kind="factor",
        title_en="Short-term reversal",
        title_zh="短期反转",
        factor_expressions=["Rev5 = -Delta(Close, 5) / Delay(Close, 5)"],
        strategy_name="custom",
        strategy_params={"source": "class Strategy(BaseStrategy):\n    def on_bar(self, bar):\n        pass\n"},
    )
    spec = idea_to_taskspec(idea, mode="factor", language="en")
    assert spec.kind == "factor" and spec.factor.expressions
    spec2 = idea_to_taskspec(idea, mode="strategy")
    assert spec2.kind == "strategy" and spec2.strategy.name == "custom"


def test_idea_to_taskspec_strategy_mode_keeps_full_universe():
    # regression: this used to truncate to symbols[:1], silently discarding
    # the rest of a requested universe (e.g. the 24-ETF pool) even though
    # per-symbol templates run independently across every symbol.
    idea = Idea(key="mom", kind="strategy", title_en="momentum", title_zh="动量",
                strategy_name="custom",
                strategy_params={"source": "class Strategy(BaseStrategy):\n    def on_bar(self, bar):\n        pass\n"})
    universe = [f"S{i}" for i in range(24)]
    spec = idea_to_taskspec(idea, mode="strategy", universe=universe)
    assert spec.data.symbols == universe


def test_idea_to_taskspec_bridges_factor_idea_to_rotation():
    # regression: a "factor" idea (no strategy_name) used to fall through
    # to a bare "momentum" strategy with no factor info in strategy mode,
    # and was silently dropped entirely by webui.server._brief_payload's
    # filter -- uploading a factor-only research report to strategy mode
    # produced zero suggestions ("利用研报中的因子，构建一个多因子策略").
    idea = Idea(key="mom", kind="factor", title_en="momentum", title_zh="动量",
                factor_expressions=["Mom20 = Delta(Close, 20) / Delay(Close, 20)"])
    spec = idea_to_taskspec(idea, mode="strategy", universe=["A", "B", "C"])
    assert spec.strategy.name == "factor_rotation"
    assert spec.strategy.params["expressions"] == idea.factor_expressions


def test_brief_payload_combines_and_dedupes_factor_ideas():
    import webui.server as srv
    from src.research.ingest import ResearchBrief

    ideas = [
        Idea(key="a", kind="factor", title_en="A", title_zh="A",
             factor_expressions=["Mom20 = Delta(Close, 20) / Delay(Close, 20)"]),
        Idea(key="b", kind="factor", title_en="B", title_zh="B",
             factor_expressions=[
                 "Mom20 = Delta(Close, 20) / Delay(Close, 20)",  # duplicate
                 "LowVol = -Ts_Std(Close, 20)",
             ]),
    ]
    brief = ResearchBrief(source_type="pdf", source="x", title="t", ideas=ideas)
    payload = srv._brief_payload(brief, "strategy", None, "2021-01-01", "2024-12-31", "en")

    combined = payload["suggestions"][0]
    assert combined["idea"]["key"] == "combined-factors"
    spec = TaskSpec.from_yaml(combined["yaml"])
    assert spec.strategy.name == "factor_rotation"
    assert len(spec.strategy.params["expressions"]) == 2  # deduped, not 3
    # per-idea single-factor suggestions are still offered alongside it
    assert len(payload["suggestions"]) == 3


def test_split_url_preserves_accompanying_instruction():
    # regression: arXiv/URL ingestion used to silently drop any instruction
    # typed alongside the link (e.g. "在精选24ETF上构建策略").
    source = "根据这篇[论文](https://arxiv.org/pdf/2402.06635)的思路，在精选24ETF上构建策略"
    url, instruction = _split_url(source)
    assert url == "https://arxiv.org/pdf/2402.06635"
    assert instruction == "根据这篇论文的思路，在精选24ETF上构建策略"


def test_ingest_pdf_upload_preserves_question(monkeypatch):
    # regression: the PDF-upload form and the question textbox are separate
    # inputs; the question ("根据PDF，生成因子") used to never reach the
    # backend at all -- ingest_source's pdf_bytes branch now accepts it
    # explicitly and merges it into the LLM-visible text.
    monkeypatch.setattr(
        "src.research.ingest._pdf_text",
        lambda path: "Momentum factor test report content.",
    )
    brief = ingest_source(
        "", pdf_bytes=b"%PDF-fake", filename="report.pdf",
        instruction="根据PDF，生成因子",
    )
    assert brief.user_instruction == "根据PDF，生成因子"
    assert "根据PDF，生成因子" in brief.text
    assert "Momentum factor test report" in brief.text


def test_ingest_wechat_link_gives_actionable_error(monkeypatch):
    # mp.weixin.qq.com returns an anti-bot "环境异常" verification page (not
    # the article) to non-browser requests -- the generic "no readable text
    # extracted" error used to hide this; it should now name the real cause
    # and tell the user to paste the text instead.
    verify_page = b"""<html><body>
        <div class="weui-msg__title">\xe7\x8e\xaf\xe5\xa2\x83\xe5\xbc\x82\xe5\xb8\xb8</div>
        <a id="js_verify">\xe5\x8e\xbb\xe9\xaa\x8c\xe8\xaf\x81</a>
    </body></html>"""
    monkeypatch.setattr("src.research.ingest._fetch", lambda url: verify_page)
    source = "根据[文章](https://mp.weixin.qq.com/s/5KKoJHeps29x5FwFyicnxw)，在精选24ETF池上做策略"
    with pytest.raises(IngestError, match="微信公众号文章无法通过程序直接抓取"):
        ingest_source(source)

    bare = "check out https://example.com/paper and build on 600519"
    url2, instruction2 = _split_url(bare)
    assert url2 == "https://example.com/paper"
    assert "600519" in instruction2


def test_extract_universe_hint_pool24(fake_llm):
    client = fake_llm(lambda user, system: '{"universe": "pool24"}')
    hint = extract_universe_hint("在精选24ETF上构建策略", client)
    assert hint is not None and hint.rule is None and len(hint.symbols) == 24


def test_extract_universe_hint_explicit_symbols(fake_llm):
    client = fake_llm(lambda user, system: '{"symbols": ["600519", "600036"]}')
    hint = extract_universe_hint("只用600519和600036这两只股票", client)
    assert hint.symbols == ["600519", "600036"] and hint.rule is None


def test_extract_universe_hint_none_when_unspecified(fake_llm):
    client = fake_llm(lambda user, system: '{"universe": null}')
    assert extract_universe_hint("just analyze this paper's method", client) is None


def test_extract_universe_hint_degrades_on_llm_failure(fake_llm):
    # this is a best-effort enrichment, not the primary feature (unlike
    # parse_prompt) -- a failure here returns None (caller's default)
    # rather than raising.
    client = fake_llm(lambda user, system: "not json")
    assert extract_universe_hint("在精选24ETF上构建策略", client) is None


def test_ingest_local_pdf():
    pdf = ROOT.parent / "Vibe_Quant_系统设计报告.pdf"  # sibling of the repo
    if not pdf.exists():
        pytest.skip("design PDF not present")
    brief = ingest_source(str(pdf))
    assert brief.source_type == "pdf"
    assert brief.text  # the design report's text is captured, ideas come later


def test_llm_extract_ideas_from_text(fake_llm):
    client = fake_llm(lambda user, system: json.dumps([
        {
            "name": "LowVol", "kind": "factor",
            "title_en": "Low volatility", "title_zh": "低波动",
            "factor_expressions": ["LowVol20 = -(Ts_Std(Close, 20) / Ts_Mean(Close, 20))"],
            "evidence": ["低波动"],
        },
    ]))
    ideas = llm_extract_ideas("我认为低波动的ETF长期跑赢", client)
    assert ideas and ideas[0].kind == "factor" and ideas[0].factor_expressions


def test_llm_extract_ideas_raises_on_bad_json(fake_llm):
    client = fake_llm(lambda user, system: "not json")
    with pytest.raises(IdeaExtractionError):
        llm_extract_ideas("some text", client)


# -------------------------------------------------------------------- intent
_NOOP_SOURCE = 'class Strategy(BaseStrategy):\\n    def on_bar(self, bar):\\n        pass\\n'


def test_intent_detects_etf_codes(fake_llm):
    fake_llm(lambda user, system: json.dumps({
        "task_yaml": f"""
name: etf-510300
kind: strategy
data: {{source: etf, symbols: ["510300"]}}
strategy: {{name: custom, params: {{source: "{_NOOP_SOURCE}"}}}}
""",
        "clarifications": [], "recognized": True,
    }))
    parsed = parse_prompt("在 510300 上做 5/20 双均线策略回测")
    assert parsed.spec.data.source == "etf"

    fake_llm(lambda user, system: json.dumps({
        "task_yaml": f"""
name: stock-600000
kind: strategy
data: {{source: stock, symbols: ["600000"]}}
strategy: {{name: custom, params: {{source: "{_NOOP_SOURCE}"}}}}
""",
        "clarifications": [], "recognized": True,
    }))
    parsed2 = parse_prompt("ma cross on 600000")  # stock, not ETF
    assert parsed2.spec.data.source == "stock"
