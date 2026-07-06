"""LLM-backed research-idea extraction (optional upgrade over keyword rules).

Given source text (paper abstract, forum post, idea), asks the configured
LLM for factor/strategy candidates in strict JSON, then validates every
factor expression against the akquant expression grammar before accepting
it. Anything invalid is dropped; any failure falls back to the keyword
rules — the LLM can only add quality, never break the pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from ..llm import LLMClient, LLMError
from .ingest import Idea

logger = logging.getLogger(__name__)

# Function names understood by akquant's expression parser + data columns.
_ALLOWED_TOKENS = {
    "ts_mean", "ts_std", "ts_max", "ts_min", "ts_sum", "ts_corr", "ts_cov",
    "ts_argmax", "ts_argmin", "ts_rank", "delay", "delta", "rank", "scale",
    "log", "abs", "sign", "signedpower", "if",
    "open", "high", "low", "close", "volume",
}
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_STRATEGY_NAMES = {"ma_cross", "rsi_reversion", "momentum", "bollinger", "buy_hold"}

_SYSTEM = """You are a quantitative research assistant. You read research text and
propose testable ideas for a daily-frequency A-share ETF universe with only
price/volume data (columns: Open, High, Low, Close, Volume).

Factor expressions must use ONLY these functions:
Ts_Mean(x,d), Ts_Std(x,d), Ts_Max(x,d), Ts_Min(x,d), Ts_Sum(x,d),
Ts_Corr(x,y,d), Ts_Cov(x,y,d), Ts_Rank(x,d), Ts_ArgMax(x,d), Ts_ArgMin(x,d),
Delay(x,d), Delta(x,d), Rank(x), Scale(x), Log(x), Abs(x), Sign(x), If(c,a,b)
plus +,-,*,/ and numeric constants. Higher factor value should predict higher
forward return (negate if the paper's signal is inverted).

Reply with ONLY a JSON array (no markdown fence), each element:
{"name": "ShortName", "kind": "factor" or "strategy",
 "title_en": "...", "title_zh": "...",
 "factor_expressions": ["Name = Expr", ...],   // for kind=factor, 1-2 items
 "strategy_name": "ma_cross|rsi_reversion|momentum|bollinger|buy_hold" or null,
 "evidence": ["short quote or concept from the text", ...]}
At most 5 ideas, ordered most to least promising. Only propose what the text
actually supports; if it needs fundamental data, skip it."""


def _expression_valid(raw: str) -> bool:
    body = raw.split("=", 1)[-1]
    return all(t.lower() in _ALLOWED_TOKENS for t in _TOKEN_RE.findall(body))


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text


def llm_extract_ideas(text: str, client: LLMClient) -> Optional[List[Idea]]:
    """Ideas from the LLM, validated. None means: use the rule fallback."""
    try:
        reply = client.query(
            f"Research text:\n\n{text[:12000]}", system_prompt=_SYSTEM
        )
        items = json.loads(_strip_fence(reply))
    except (LLMError, json.JSONDecodeError, TypeError) as exc:
        logger.warning("LLM idea extraction unavailable: %s", exc)
        return None
    if not isinstance(items, list):
        return None

    ideas: List[Idea] = []
    for rank, item in enumerate(items[:5]):
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        exprs = [
            str(e) for e in (item.get("factor_expressions") or [])
            if _expression_valid(str(e))
        ][:2]
        strategy = item.get("strategy_name")
        if strategy not in _STRATEGY_NAMES:
            strategy = None
        if kind == "factor" and not exprs:
            continue  # every expression it proposed was invalid
        if kind == "strategy" and not strategy:
            continue
        if kind not in ("factor", "strategy"):
            continue
        name = re.sub(r"[^\w-]", "", str(item.get("name", f"idea{rank+1}")))[:24]
        ideas.append(
            Idea(
                key=f"llm_{name or rank + 1}",
                kind=kind,
                title_en=str(item.get("title_en", name))[:120],
                title_zh=str(item.get("title_zh", name))[:120],
                evidence=[str(e)[:80] for e in (item.get("evidence") or [])][:4],
                score=len(items) - rank,  # preserve the LLM's ordering
                factor_expressions=exprs,
                strategy_name=strategy,
            )
        )
    return ideas or None
