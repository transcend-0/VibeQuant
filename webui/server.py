"""VibeQuant Web UI server (FastAPI).

Endpoints (grouped):
    UI          GET /
    research    POST /api/parse · /api/ingest · /api/ingest_pdf · /api/run
    runs        GET /api/runs · /api/runs/{id} · /api/runs/{id}/artifact/{name}
    library     GET /api/data · /api/data/{symbol} · /api/factors
    reference   GET /api/strategies · /api/factor_presets · /api/universes
    llm         GET/POST /api/llm/config · POST /api/llm/test
    deploy      GET/POST /api/deployments · POST .../{id}/run · .../{id}/toggle
                DELETE /api/deployments/{id}

A scheduler task runs every minute and fires enabled deployments after
their post-close time (Asia/Shanghai, weekdays) — signal emails only,
never orders. The pre-run risk gate still refuses paper/live tasks.
Binds to 127.0.0.1 by default.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src import live as live_mod
from src.config import raw_data_dir, workspace_dir
from src.dsl import DSLError, TaskSpec
from src.factors.registry import list_factors
from src.intent import IntentError, parse_prompt
from src.llm import LLMError, get_client, load_llm_config, save_llm_config, test_connection
from src.memory import MemoryStore
from src.planner import make_plan
from src.research import ingest_source
from src.research.ingest import Idea, IngestError
from src.research.llm_ideas import IdeaExtractionError, llm_extract_ideas
from src.research.tasks import (
    DEFAULT_ETF_UNIVERSE,
    extract_universe_hint,
    idea_to_taskspec,
)
from src.runner import run_task
from src.strategies import REGISTRY

logger = logging.getLogger(__name__)

app = FastAPI(title="VibeQuant", version="0.3.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"

_run_lock = asyncio.Lock()  # one engine run at a time keeps the box responsive


# ------------------------------------------------- deployment scheduler
async def _scheduler_loop() -> None:
    while True:
        try:
            fired = await asyncio.to_thread(live_mod.run_due_deployments)
            for result in fired:
                logger.info("scheduler: %s", result)
        except Exception as exc:  # never let the loop die
            logger.error("scheduler error: %s", exc)
        await asyncio.sleep(60)


@app.on_event("startup")
async def _start_scheduler() -> None:
    app.state.scheduler = asyncio.create_task(_scheduler_loop())


@app.on_event("shutdown")
async def _stop_scheduler() -> None:
    task = getattr(app.state, "scheduler", None)
    if task:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def _factor_universes() -> List[Dict[str, Any]]:
    """ETF pool presets for factor research (source: etf)."""
    from src.data_sources.etf_pool import CATEGORIES, pool_categories

    cats = pool_categories()
    presets = [
        {
            "key": "etf_pool",
            "label_en": f"Built-in ETF pool ({len(DEFAULT_ETF_UNIVERSE)})",
            "label_zh": f"内置 ETF 池（{len(DEFAULT_ETF_UNIVERSE)}只）",
            "source": "etf",
            "symbols": DEFAULT_ETF_UNIVERSE,
        }
    ]
    for key, (label_en, label_zh) in CATEGORIES.items():
        symbols = cats.get(key, [])
        if len(symbols) >= 2:
            presets.append(
                {
                    "key": f"pool_{key}",
                    "label_en": f"{label_en} ETFs ({len(symbols)})",
                    "label_zh": f"{label_zh} ETF（{len(symbols)}只）",
                    "source": "etf",
                    "symbols": symbols,
                }
            )
    presets.append(
        {"key": "custom", "label_en": "Custom…", "label_zh": "自定义…",
         "source": "etf", "symbols": []}
    )
    return presets


# strategy-research universes: indices and pools with their data source
STRATEGY_UNIVERSES = [
    {"key": "csi300", "label_en": "CSI 300 (index)", "label_zh": "沪深300（指数）",
     "source": "index", "symbols": ["sh000300"]},
    {"key": "csi1000", "label_en": "CSI 1000 (index)", "label_zh": "中证1000（指数）",
     "source": "index", "symbols": ["399852"]},
    {"key": "chinext", "label_en": "ChiNext (index)", "label_zh": "创业板指（指数）",
     "source": "index", "symbols": ["399006"]},
    {"key": "sp500", "label_en": "S&P 500 (index)", "label_zh": "标普500（指数）",
     "source": "us", "symbols": [".INX"]},
    {"key": "ndx100", "label_en": "NASDAQ 100 (index)", "label_zh": "纳斯达克100（指数）",
     "source": "us", "symbols": [".NDX"]},
    {"key": "hsi", "label_en": "Hang Seng (index)", "label_zh": "恒生指数",
     "source": "hk", "symbols": ["HSI"]},
    {"key": "etf_pool", "label_en": "Built-in ETF pool", "label_zh": "内置 ETF 池",
     "source": "etf", "symbols": None},  # filled at request time
    {"key": "custom", "label_en": "Custom…", "label_zh": "自定义…",
     "source": "etf", "symbols": []},
]

ALLOWED_ARTIFACTS = {
    "report.html",
    "report.md",
    "result.json",
    "task.yaml",
    "equity.csv",
    "factor_analysis.json",
    "error.txt",
}

FACTOR_PRESETS = [
    {
        "name_en": "Momentum 20d",
        "name_zh": "20日动量",
        "expr": "Mom20 = Delta(Close, 20) / Delay(Close, 20)",
    },
    {
        "name_en": "Reversal 5d",
        "name_zh": "5日反转",
        "expr": "Rev5 = -Delta(Close, 5) / Delay(Close, 5)",
    },
    {
        "name_en": "Low volatility 20d",
        "name_zh": "20日低波动",
        "expr": "LowVol20 = -(Ts_Std(Close, 20) / Ts_Mean(Close, 20))",
    },
    {
        "name_en": "Price-volume corr 10d",
        "name_zh": "10日量价相关",
        "expr": "PVCorr10 = Rank(Ts_Corr(Close, Volume, 10))",
    },
    {
        "name_en": "Volume surge",
        "name_zh": "放量因子",
        "expr": "VolSurge = Volume / Ts_Mean(Volume, 20)",
    },
    {
        "name_en": "Distance from 20d high",
        "name_zh": "距20日高点",
        "expr": "FromHigh20 = Close / Ts_Max(High, 20) - 1",
    },
]


class ParseRequest(BaseModel):
    prompt: str


class IngestRequest(BaseModel):
    source: str
    mode: str = "factor"  # which research mode is asking
    universe: Optional[List[str]] = None
    start: Optional[str] = None
    end: Optional[str] = None
    language: str = "en"


class RunRequest(BaseModel):
    task: Optional[Dict[str, Any]] = None
    yaml_text: Optional[str] = None


def _spec_from_request(req: RunRequest) -> TaskSpec:
    try:
        if req.yaml_text:
            return TaskSpec.from_yaml(req.yaml_text)
        if req.task:
            return TaskSpec.from_dict(req.task)
    except DSLError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise HTTPException(status_code=422, detail="provide 'task' or 'yaml_text'")


def _plan_payload(spec: TaskSpec) -> List[Dict[str, str]]:
    return [
        {"tool": s.tool, "title_en": s.title_en, "title_zh": s.title_zh}
        for s in make_plan(spec).steps
    ]


def _brief_payload(brief, req_mode: str, universe, start, end, language):
    """Brief + a ready-to-run task suggestion per extracted idea."""

    def _build(idea, mode, intent):
        return idea_to_taskspec(
            idea,
            mode=mode,
            universe=universe,
            start=start or "2021-01-01",
            end=end or "2024-12-31",
            language=language,
            intent=intent,
        )

    suggestions = []
    for idea in brief.ideas:
        # suggestions strictly match the requesting research mode
        mode = req_mode
        if mode == "factor" and not idea.factor_expressions:
            continue
        # a factor idea with real expressions is still usable in strategy
        # mode -- idea_to_taskspec bridges it into a factor_rotation
        # rather than being silently dropped for lacking a strategy_name.
        if mode == "strategy" and not idea.strategy_name and not idea.factor_expressions:
            continue
        try:
            spec = _build(idea, mode, f"[{brief.source_type}] {brief.title[:120]}")
        except DSLError:
            continue
        suggestions.append(
            {
                "idea": idea.to_dict(),
                "mode": mode,
                "yaml": spec.to_yaml(),
                "plan": _plan_payload(spec),
            }
        )

    # "利用研报中的因子构建一个多因子策略": combine every extracted factor
    # idea's expressions into ONE multi-factor rotation, in addition to the
    # per-idea single-factor suggestions above.
    if req_mode == "strategy":
        seen_names = set()
        all_exprs = []
        for idea in brief.ideas:
            for e in idea.factor_expressions:
                name = e.split("=", 1)[0].strip() if "=" in e else e
                if name in seen_names:
                    continue  # multiple ideas often propose the same factor
                seen_names.add(name)
                all_exprs.append(e)
        if len(all_exprs) >= 2:
            combined = Idea(
                key="combined-factors",
                kind="strategy",
                title_en="Combined multi-factor rotation",
                title_zh="组合多因子轮动",
                evidence=[i.title_en for i in brief.ideas if i.factor_expressions],
                strategy_name="factor_rotation",
                strategy_params={
                    "expressions": all_exprs,
                    "top_k": 5,
                    "rebalance_days": 20,
                },
            )
            try:
                spec = _build(
                    combined, "strategy",
                    f"[{brief.source_type}] {brief.title[:120]} (combined factors)",
                )
            except DSLError:
                pass
            else:
                suggestions.insert(0, {
                    "idea": combined.to_dict(),
                    "mode": "strategy",
                    "yaml": spec.to_yaml(),
                    "plan": _plan_payload(spec),
                })

    payload = brief.to_dict()
    payload["suggestions"] = suggestions
    payload["default_universe"] = DEFAULT_ETF_UNIVERSE
    return payload


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def _template_code(name: str) -> str:
    """Python source of a strategy template (adapter class for rotation)."""
    import importlib
    import inspect

    try:
        if name == "factor_rotation":
            from src.adapters import akquant_engine as eng

            return (
                inspect.getsource(eng._rotation_scores)
                + "\n\n"
                + inspect.getsource(eng._RotationStrategy)
            )
        module = importlib.import_module(f"src.strategies.{name}")
        return inspect.getsource(module)
    except Exception as exc:
        return f"# source unavailable: {exc}"


@app.get("/api/strategies")
def strategies() -> List[Dict[str, Any]]:
    return [
        {
            "name": name,
            "summary_en": tpl.summary_en,
            "summary_zh": tpl.summary_zh,
            "defaults": tpl.defaults,
            "code": _template_code(name),
        }
        for name, tpl in sorted(REGISTRY.items())
    ]


@app.get("/api/factor_presets")
def factor_presets() -> List[Dict[str, str]]:
    return FACTOR_PRESETS


@app.post("/api/parse")
def parse(req: ParseRequest) -> Dict[str, Any]:
    if not req.prompt.strip():
        raise HTTPException(status_code=422, detail="empty prompt")
    try:
        parsed = parse_prompt(req.prompt)
    except IntentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "task": parsed.spec.to_dict(),
        "yaml": parsed.spec.to_yaml(),
        "clarifications": parsed.clarifications,
        "recognized": parsed.recognized,
        "plan": _plan_payload(parsed.spec),
    }


async def _llm_ideas(brief) -> Optional[List[str]]:
    """Extract candidate ideas from the brief's text via the LLM (required).

    Also derives a universe/symbols override from any accompanying user
    instruction (e.g. "只用600519和600036" typed alongside a pasted paper
    link) — returns that hint (None if the instruction named nothing
    specific), so the caller can prefer it over its own default.
    """
    client = get_client()
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="LLM not configured — set up config/llm.yaml to extract ideas.",
        )
    if not brief.text:
        return None
    try:
        ideas = await asyncio.to_thread(llm_extract_ideas, brief.text, client)
    except IdeaExtractionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    brief.ideas = ideas
    if ideas:
        brief.notes.append(f"ideas extracted by LLM ({client.model_name})")
    else:
        brief.notes.append("LLM found no usable ideas in this text")

    universe_hint = None
    if brief.user_instruction:
        universe_hint = await asyncio.to_thread(
            extract_universe_hint, brief.user_instruction, client
        )
        if universe_hint:
            brief.notes.append(
                f"universe set from your instruction ({len(universe_hint)} symbols)"
            )
    return universe_hint


@app.post("/api/ingest")
async def ingest(req: IngestRequest) -> Dict[str, Any]:
    try:
        brief = await asyncio.to_thread(ingest_source, req.source)
    except IngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"failed to fetch source: {exc}"
        ) from exc
    universe_hint = await _llm_ideas(brief)
    return _brief_payload(
        brief, req.mode, req.universe or universe_hint, req.start, req.end,
        req.language,
    )


@app.post("/api/ingest_pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    mode: str = Form("factor"),
    language: str = Form("en"),
    start: str = Form("2021-01-01"),
    end: str = Form("2024-12-31"),
    question: str = Form(""),
) -> Dict[str, Any]:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="only .pdf uploads supported")
    content = await file.read()
    if len(content) > 30_000_000:
        raise HTTPException(status_code=422, detail="PDF larger than 30 MB")
    try:
        brief = await asyncio.to_thread(
            ingest_source, "", content, file.filename or "uploaded.pdf", question
        )
    except IngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    universe_hint = await _llm_ideas(brief)
    return _brief_payload(brief, mode, universe_hint, start, end, language)


@app.post("/api/run")
async def run(req: RunRequest) -> Dict[str, Any]:
    spec = _spec_from_request(req)
    async with _run_lock:
        result = await asyncio.to_thread(run_task, spec)
    payload = result.to_dict()
    payload["report_markdown"] = result.report_markdown
    detail = MemoryStore(workspace_dir()).load_run(result.run_id) or {}
    payload["equity_curve"] = detail.get("equity_curve")
    payload["has_html_report"] = "report.html" in result.artifacts
    return payload


@app.get("/api/runs")
def runs(limit: int = 50) -> List[Dict[str, Any]]:
    return MemoryStore(workspace_dir()).list_runs(limit=limit)


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> Dict[str, Any]:
    store = MemoryStore(workspace_dir())
    detail = store.load_run(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="run not found")
    run_dir = store.runs_dir / run_id
    detail["has_html_report"] = (run_dir / "report.html").exists()
    factor_file = run_dir / "factor_analysis.json"
    if factor_file.exists():
        import json

        detail["factor"] = json.loads(factor_file.read_text(encoding="utf-8"))
    return detail


@app.get("/api/runs/{run_id}/artifact/{name}")
def run_artifact(run_id: str, name: str) -> FileResponse:
    if name not in ALLOWED_ARTIFACTS:
        raise HTTPException(status_code=404, detail="unknown artifact")
    store = MemoryStore(workspace_dir())
    # resolve() + prefix check guards against path tricks in run_id
    path = (store.runs_dir / run_id / name).resolve()
    if not str(path).startswith(str(store.runs_dir.resolve())) or not path.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    media = {
        ".html": "text/html",
        ".md": "text/markdown",
        ".json": "application/json",
        ".yaml": "text/plain",
        ".csv": "text/csv",
        ".txt": "text/plain",
    }.get(path.suffix, "application/octet-stream")
    return FileResponse(path, media_type=media)


# ------------------------------------------------------------- reference
@app.get("/api/universes")
def universes(mode: str = "factor") -> List[Dict[str, Any]]:
    if mode == "strategy":
        out = []
        for u in STRATEGY_UNIVERSES:
            u = dict(u)
            if u["symbols"] is None:
                u["symbols"] = DEFAULT_ETF_UNIVERSE
            out.append(u)
        return out
    return _factor_universes()


# ------------------------------------------------------ library: data
@app.get("/api/data")
def data_index() -> List[Dict[str, Any]]:
    """Downloaded original bars (data/raw/*), one entry per cached file."""
    entries: List[Dict[str, Any]] = []
    root = raw_data_dir().parent  # data/raw
    for file in sorted(root.glob("*/*.csv")):
        try:
            stem_parts = file.stem.split("__")
            symbol = stem_parts[0].replace("_", ".")
            head = pd.read_csv(file, nrows=1)
            nrows = sum(1 for _ in file.open()) - 1
            entries.append(
                {
                    "symbol": symbol,
                    "source": file.parent.name,
                    "file": file.name,
                    "rows": nrows,
                    "start": stem_parts[1] if len(stem_parts) > 2 else None,
                    "end": stem_parts[2] if len(stem_parts) > 2 else None,
                    "columns": list(head.columns),
                    "size_kb": round(file.stat().st_size / 1024, 1),
                }
            )
        except Exception:
            continue
    # also expose bundled sample data
    sample_dir = raw_data_dir().parent.parent / "sample"
    for file in sorted(sample_dir.glob("*.csv")) if sample_dir.exists() else []:
        try:
            nrows = sum(1 for _ in file.open()) - 1
            entries.append(
                {
                    "symbol": file.stem,
                    "source": "sample",
                    "file": file.name,
                    "rows": nrows,
                    "start": None,
                    "end": None,
                    "columns": [],
                    "size_kb": round(file.stat().st_size / 1024, 1),
                }
            )
        except Exception:
            continue
    return entries


@app.get("/api/data/{source}/{file_name}")
def data_detail(source: str, file_name: str, max_points: int = 900) -> Dict[str, Any]:
    """Bars from one cached file, downsampled for charting."""
    if source == "sample":
        base = raw_data_dir().parent.parent / "sample"
    else:
        base = raw_data_dir(source)
    path = (base / Path(file_name).name).resolve()
    if not str(path).startswith(str(base.resolve())) or not path.exists():
        raise HTTPException(status_code=404, detail="data file not found")
    df = pd.read_csv(path, parse_dates=["date"])
    step = max(1, len(df) // max_points)
    ds = df.iloc[::step]
    if len(df) and (len(df) - 1) % step:
        ds = pd.concat([ds, df.iloc[[-1]]])
    return {
        "symbol": Path(file_name).stem.split("__")[0].replace("_", "."),
        "rows": int(len(df)),
        "start": str(df["date"].min().date()) if len(df) else None,
        "end": str(df["date"].max().date()) if len(df) else None,
        "dates": [str(d.date()) for d in ds["date"]],
        "close": [round(float(v), 4) for v in ds["close"]],
        "volume": [float(v) for v in ds["volume"]] if "volume" in ds else [],
        "tail": json.loads(
            df.tail(10).to_json(orient="records", date_format="iso")
        ),
    }


# --------------------------------------------------- library: factors
@app.get("/api/factors")
def factors_index(limit: int = 100) -> List[Dict[str, Any]]:
    return list_factors(limit=limit)


# ----------------------------------------------------------------- llm
class LLMConfigRequest(BaseModel):
    model: str
    api_key: str
    base_url: str


@app.get("/api/llm/config")
def llm_config_get() -> Dict[str, Any]:
    conf = load_llm_config()
    if conf is None:
        return {"configured": False}
    key = conf["api_key"]
    return {
        "configured": True,
        "model": conf["model"],
        "base_url": conf["base_url"],
        "api_key_masked": key[:6] + "…" + key[-4:] if len(key) > 12 else "***",
    }


@app.post("/api/llm/config")
def llm_config_set(req: LLMConfigRequest) -> Dict[str, Any]:
    if not (req.model.strip() and req.api_key.strip() and req.base_url.strip()):
        raise HTTPException(status_code=422, detail="all three fields are required")
    save_llm_config(req.model.strip(), req.api_key.strip(), req.base_url.strip())
    return {"saved": True}


@app.post("/api/llm/test")
async def llm_test() -> Dict[str, Any]:
    return await asyncio.to_thread(test_connection)


# -------------------------------------------------------------- deploy
class DeployRequest(BaseModel):
    yaml_text: str
    email_to: str = ""
    run_at: str = "16:30"
    name: str = ""


@app.get("/api/deployments")
def deployments_index() -> List[Dict[str, Any]]:
    return [d.to_dict() for d in live_mod.list_deployments()]


@app.post("/api/deployments")
def deployments_create(req: DeployRequest) -> Dict[str, Any]:
    try:
        dep = live_mod.create_deployment(
            req.yaml_text, email_to=req.email_to, run_at=req.run_at, name=req.name
        )
    except live_mod.deploy.DeployError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return dep.to_dict()


@app.post("/api/deployments/{dep_id}/run")
async def deployments_run(dep_id: str) -> Dict[str, Any]:
    try:
        async with _run_lock:
            return await asyncio.to_thread(
                live_mod.run_deployment, dep_id, True
            )
    except live_mod.deploy.DeployError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/deployments/{dep_id}/toggle")
def deployments_toggle(dep_id: str) -> Dict[str, Any]:
    deps = {d.id: d for d in live_mod.list_deployments()}
    if dep_id not in deps:
        raise HTTPException(status_code=404, detail="deployment not found")
    live_mod.set_enabled(dep_id, not deps[dep_id].enabled)
    return {"enabled": not deps[dep_id].enabled}


@app.delete("/api/deployments/{dep_id}")
def deployments_delete(dep_id: str) -> Dict[str, Any]:
    if not live_mod.delete_deployment(dep_id):
        raise HTTPException(status_code=404, detail="deployment not found")
    return {"deleted": True}


# --------------------------------------------------------------- refine
class RefineRequest(BaseModel):
    yaml_text: str
    question: str
    result_summary: Optional[Dict[str, Any]] = None
    language: str = "en"


@app.post("/api/refine")
async def refine(req: RefineRequest) -> Dict[str, Any]:
    from src.research.refine import refine_task

    if not req.question.strip():
        raise HTTPException(status_code=422, detail="empty question")
    try:
        out = await asyncio.to_thread(
            refine_task,
            req.yaml_text,
            req.question,
            req.result_summary,
            req.language,
        )
    except DSLError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        spec = TaskSpec.from_yaml(out["yaml"])
        out["plan"] = _plan_payload(spec)
    except DSLError:
        out["plan"] = []
    return out


# ------------------------------------------------- markets & universes v2
# One market -> universe hierarchy shared by BOTH research modes.
MARKETS = [
    {
        "key": "cn_etf", "source": "etf",
        "label_en": "A-share ETF", "label_zh": "A股 ETF",
        "universes": [
            {"key": "pool24", "label_en": "Curated 24 ETF",
             "label_zh": "精选24ETF", "symbols": DEFAULT_ETF_UNIVERSE},
            {"key": "all_top200", "label_en": "All ETF · top 200 by turnover",
             "label_zh": "全部 ETF · 成交额前200", "dynamic": True},
            {"key": "all_full", "label_en": "All ETF · complete (slow!)",
             "label_zh": "全部 ETF · 完整列表（慢！）", "dynamic": True},
            {"key": "custom", "label_en": "Custom…", "label_zh": "自定义…",
             "symbols": []},
        ],
    },
    {
        "key": "cn_stock", "source": "stock",
        "label_en": "A-share stocks", "label_zh": "A股 个股",
        "universes": [
            {"key": "hs300", "label_en": "CSI 300 constituents (PIT)",
             "label_zh": "沪深300 成分股（时点池）", "dynamic": True},
            {"key": "bluechip", "label_en": "Blue chips (10)",
             "label_zh": "蓝筹样本（10只）",
             "symbols": ["600519", "600036", "601318", "600900", "000333",
                          "000858", "601899", "002594", "300750", "600030"]},
            {"key": "custom", "label_en": "Custom…", "label_zh": "自定义…",
             "symbols": []},
        ],
    },
    {
        "key": "hk", "source": "hk",
        "label_en": "Hong Kong", "label_zh": "港股",
        "universes": [
            {"key": "hsi", "label_en": "Hang Seng index", "label_zh": "恒生指数",
             "symbols": ["HSI"]},
            {"key": "hk_bigtech", "label_en": "Big tech (6)",
             "label_zh": "科技龙头（6只）",
             "symbols": ["00700", "09988", "03690", "01810", "09618", "09999"]},
            {"key": "custom", "label_en": "Custom…", "label_zh": "自定义…",
             "symbols": []},
        ],
    },
    {
        "key": "us", "source": "us",
        "label_en": "US stocks", "label_zh": "美股",
        "universes": [
            {"key": "sp500", "label_en": "S&P 500 (index)",
             "label_zh": "标普500（指数）", "symbols": [".INX"]},
            {"key": "ndx100", "label_en": "NASDAQ 100 (index)",
             "label_zh": "纳斯达克100（指数）", "symbols": [".NDX"]},
            {"key": "mega7", "label_en": "Magnificent 7", "label_zh": "七巨头",
             "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]},
            {"key": "custom", "label_en": "Custom…", "label_zh": "自定义…",
             "symbols": []},
        ],
    },
    {
        "key": "crypto", "source": "crypto",
        "label_en": "Crypto", "label_zh": "加密货币",
        "universes": [
            {"key": "top8", "label_en": "Top coins (8)", "label_zh": "主流币（8）",
             "symbols": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "LTC"]},
            {"key": "custom", "label_en": "Custom…", "label_zh": "自定义…",
             "symbols": []},
        ],
    },
]


@app.get("/api/markets")
def markets() -> List[Dict[str, Any]]:
    return MARKETS


@app.get("/api/universe_symbols")
async def universe_symbols(market: str, key: str) -> Dict[str, Any]:
    """Resolve a dynamic universe (e.g. the all-ETF list) to symbols."""
    from src.config import raw_data_dir
    from src.data_sources.market import MarketDataError, fetch_all_etf_list

    if market == "cn_etf" and key in ("all_top200", "all_full"):
        top = 200 if key == "all_top200" else None
        try:
            entries = await asyncio.to_thread(
                fetch_all_etf_list, raw_data_dir("etf"), top
            )
        except MarketDataError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"symbols": [e["code"] for e in entries], "count": len(entries)}

    if market == "cn_stock":
        from src.data_sources.constituents import available_pools, pool_symbols

        if key in available_pools():
            symbols = await asyncio.to_thread(pool_symbols, key)
            return {
                "symbols": symbols,
                "count": len(symbols),
                "note": "current membership; runs mask history point-in-time",
            }
    raise HTTPException(status_code=404, detail="unknown dynamic universe")


# -------------------------------------------------------- auto-optimize
class AutoOptimizeRequest(BaseModel):
    yaml_text: str
    rounds: int = 3
    language: str = "en"


@app.post("/api/auto_optimize")
async def auto_optimize_endpoint(req: AutoOptimizeRequest) -> Dict[str, Any]:
    from src.research.auto_optimize import auto_optimize

    try:
        async with _run_lock:
            return await asyncio.to_thread(
                auto_optimize, req.yaml_text, req.rounds, req.language
            )
    except DSLError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
