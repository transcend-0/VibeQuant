"""LLM-backed research-idea extraction.

Given source text (paper abstract, forum post, idea), asks the configured
LLM for factor/strategy candidates in strict JSON, then validates every
factor expression against akquant's real expression grammar (sourced from
`src.adapters.akquant_factor`, not a hand-copied list) before accepting
it. Anything invalid is dropped from the result. If the reply is
malformed or every proposed idea fails validation, the request is retried
(with the error fed back) up to 3 times via `query_structured`; if it
still fails, `IdeaExtractionError` is raised — there is no rule-based
fallback.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List

from ..adapters.akquant_factor import known_expression_functions
from ..llm import LLMClient, LLMError, query_structured
from .ingest import Idea

logger = logging.getLogger(__name__)


class IdeaExtractionError(RuntimeError):
    pass


# Function names understood by akquant's expression parser (single source of
# truth: src.adapters.akquant_factor.known_expression_functions) + data columns.
_ALLOWED_TOKENS = {f.lower() for f in known_expression_functions()} | {
    "open", "high", "low", "close", "volume",
}
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_FUNCTION_CATALOG = ", ".join(sorted(known_expression_functions()))

_SYSTEM = f"""You are a quantitative research assistant. You read research text and
propose testable ideas for a daily-frequency A-share ETF universe with only
price/volume data (columns: Open, High, Low, Close, Volume).

Factor expressions must use ONLY these functions: {_FUNCTION_CATALOG}
plus +,-,*,/ and numeric constants. Higher factor value should predict higher
forward return (negate if the paper's signal is inverted).

Reply with ONLY a JSON array (no markdown fence), each element:
{{"name": "ShortName", "kind": "factor" or "strategy",
 "title_en": "...", "title_zh": "...",
 "factor_expressions": ["Name = Expr", ...],   // for kind=factor, 1-2 items
 "strategy_name": "custom" or null,   // "custom" if kind=strategy, else null
 "strategy_source": "def signal(closes, position):\\n    ..." or null,
   // required when strategy_name="custom": complete Python implementing
   // the paper's rule. closes: chronological floats. position: currently
   // held QUANTITY (0 if flat, not a weight). Return a target weight in
   // [0,1] or None. Plain Python only (len/sum/min/max/range) -- no imports.
 "strategy_warmup": <int> or null,  // leading bars to skip, e.g. 20
 "evidence": ["short quote or concept from the text", ...]}}
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


def _parse_ideas(reply: str) -> List[Idea]:
    try:
        items = json.loads(_strip_fence(reply))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"reply was not valid JSON: {exc}") from exc
    if not isinstance(items, list):
        raise ValueError("reply JSON must be a list")

    ideas: List[Idea] = []
    for rank, item in enumerate(items[:5]):
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        exprs = [
            str(e) for e in (item.get("factor_expressions") or [])
            if _expression_valid(str(e))
        ][:2]
        strategy = None
        strategy_params: dict = {}
        if item.get("strategy_name") == "custom":
            source = str(item.get("strategy_source") or "")
            if "def signal(" in source:  # light sanity check; custom is unsandboxed
                strategy = "custom"
                warmup = item.get("strategy_warmup")
                strategy_params = {
                    "source": source,
                    "warmup": int(warmup) if isinstance(warmup, (int, float)) else 0,
                }
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
                strategy_params=strategy_params,
            )
        )
    if items and not ideas:
        # the model tried but every candidate failed validation — worth a
        # retry with feedback, unlike a genuine "nothing here" ([] items).
        raise ValueError(
            "none of the proposed ideas passed validation "
            "(invalid kind, strategy_name, or factor expression tokens)"
        )
    return ideas


def llm_extract_ideas(text: str, client: LLMClient) -> List[Idea]:
    """Ideas from the LLM, validated. Raises IdeaExtractionError on failure."""
    user_prompt = f"Research text:\n\n{text[:12000]}"
    try:
        return query_structured(client, user_prompt, _SYSTEM, _parse_ideas)
    except LLMError as exc:
        raise IdeaExtractionError(f"LLM idea extraction failed: {exc}") from exc
    except ValueError as exc:
        raise IdeaExtractionError(
            f"LLM idea extraction failed after 3 attempts: {exc}"
        ) from exc
