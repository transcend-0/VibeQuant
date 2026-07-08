"""Turn extracted research Ideas into runnable TaskSpecs."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..dsl import DataSpec, FactorSpec, ReportSpec, StrategySpec, TaskSpec
from ..llm import LLMClient, LLMError, query_structured
from ..strategies.custom import DEFAULT_SOURCE as CUSTOM_DEFAULT_SOURCE
from ..strategies.factor_rotation import DEFAULT_SOURCE as ROTATION_DEFAULT_SOURCE
from .ingest import Idea
from .llm_ideas import _strip_fence

# The built-in categorized ETF pool (abroad/commodity/bond/index/industry)
# is the default cross-section when the user names no universe.
from ..data_sources.etf_pool import POOL_SYMBOLS as DEFAULT_ETF_UNIVERSE  # noqa: E402

logger = logging.getLogger(__name__)


def idea_to_taskspec(
    idea: Idea,
    mode: str,  # "factor" | "strategy" — the UI mode requesting the task
    universe: Optional[List[str]] = None,
    start: str = "2025-01-01",
    end: str = "2025-12-31",
    language: str = "en",
    intent: str = "",
) -> TaskSpec:
    """Build a TaskSpec for an Idea in the requested research mode.

    An idea whose kind doesn't match the mode is still usable when it has
    the other side's material (e.g. a factor idea backed by a strategy
    template); otherwise a sensible default is chosen.
    """
    symbols = [str(s) for s in (universe or DEFAULT_ETF_UNIVERSE)]
    # no explicit override, or the override happens to equal the pool
    # (extract_universe_hint returns this exact list when the user names
    # the pool by name) -- either way this IS the curated pool, tag it.
    universe_key = "pool24" if symbols == list(DEFAULT_ETF_UNIVERSE) else "custom"
    slug = re.sub(r"[^a-zA-Z0-9-]+", "-", idea.key)[:30].strip("-")

    if mode == "factor":
        expressions = idea.factor_expressions or _fallback_expressions(idea)
        spec = TaskSpec(
            name=f"factor-{slug}",
            kind="factor",
            intent=intent or idea.title_en,
            data=DataSpec(
                source="etf", universe=universe_key, symbols=symbols,
                start=start, end=end,
            ),
            factor=FactorSpec(expressions=expressions),
            report=ReportSpec(language=language),
        )
    else:
        if idea.strategy_name:
            strategy_name = idea.strategy_name
            params: Dict[str, Any] = dict(idea.strategy_params)
        elif idea.factor_expressions:
            # no strategy template was proposed, but real factor
            # expressions were (e.g. a factor-heavy research report) --
            # bridge them into a multi-factor rotation (the standard
            # top-K/FACTOR_SCORES skeleton, fully editable from here) rather
            # than silently falling back to a template-less "momentum" guess.
            strategy_name = "factor_rotation"
            params = {
                "expressions": list(idea.factor_expressions),
                "source": ROTATION_DEFAULT_SOURCE,
            }
        else:
            # no strategy_name and no factor_expressions to bridge from --
            # only reachable via direct calls that bypass _brief_payload's
            # filter. Fall back to a no-op skeleton rather than crashing.
            strategy_name = "custom"
            params = {"source": CUSTOM_DEFAULT_SOURCE}
        spec = TaskSpec(
            name=f"strategy-{slug}",
            kind="strategy",
            intent=intent or idea.title_en,
            # a "custom" per-symbol signal runs independently on every
            # symbol via the adapter — no reason to truncate to one when
            # the caller asked for the whole universe.
            data=DataSpec(
                source="etf", universe=universe_key, symbols=symbols,
                start=start, end=end,
            ),
            strategy=StrategySpec(name=strategy_name, params=params),
            report=ReportSpec(language=language),
        )
    spec.validate()
    return spec


def _fallback_expressions(idea: Idea) -> List[str]:
    """Factor mode was requested but the idea carries no expressions."""
    return ["Mom20 = Delta(Close, 20) / Delay(Close, 20)"]


_UNIVERSE_SYSTEM = """You read a short instruction that may specify which trading
universe to use. Reply with ONLY JSON (no markdown fence), exactly one shape:
{"universe": "pool24"}          the text asks for the curated/built-in
                                 24-ETF pool (精选24ETF池 / 内置24ETF池 / 24ETF池)
{"symbols": ["<ticker>", ...]}  the text names specific tickers — quote
                                 every one as a string; A-share codes are
                                 6 digits and may have leading zeros, do
                                 not treat them as numbers
{"universe": null}              no specific universe/symbols mentioned"""


def _parse_universe_reply(reply: str) -> Dict[str, Any]:
    try:
        payload = json.loads(_strip_fence(reply))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"reply was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("reply JSON must be an object")
    return payload


def extract_universe_hint(text: str, client: LLMClient) -> Optional[List[str]]:
    """Explicit universe/symbols named in `text`, else None.

    None means "use the caller's own default" — whether because the text
    genuinely names nothing, or because this best-effort enrichment step
    itself failed. Unlike parse_prompt/llm_extract_ideas (the primary
    feature, which must never degrade silently), this is a secondary hint
    on top of an ingestion that already succeeded, so a failure here logs
    a warning and falls back to the caller's default rather than failing
    the whole request.
    """
    if not text.strip():
        return None
    try:
        payload = query_structured(
            client, text[:2000], _UNIVERSE_SYSTEM, _parse_universe_reply,
            max_attempts=2,
        )
    except (LLMError, ValueError) as exc:
        logger.warning("universe hint extraction failed, using default: %s", exc)
        return None

    if payload.get("universe") == "pool24":
        return list(DEFAULT_ETF_UNIVERSE)
    symbols = payload.get("symbols")
    if isinstance(symbols, list) and symbols:
        return [str(s) for s in symbols]
    return None
