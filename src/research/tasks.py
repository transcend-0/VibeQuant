"""Turn extracted research Ideas into runnable TaskSpecs."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..data_sources import universe_builder
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
    universe_rule: Optional[Dict[str, Any]] = None,
    start: str = "2025-01-01",
    end: str = "2025-12-31",
    language: str = "en",
    intent: str = "",
) -> TaskSpec:
    """Build a TaskSpec for an Idea in the requested research mode.

    An idea whose kind doesn't match the mode is still usable when it has
    the other side's material (e.g. a factor idea backed by a strategy
    template); otherwise a sensible default is chosen.

    `universe_rule`: a validated rule dict (extract_universe_hint's
    UniverseHint.rule). It is embedded as data.universe_rule so the
    runner's build_universe step constructs the pool at execution time —
    the suggested task's symbols stay a placeholder until then, and
    data.universe carries the deterministic pool id the build will use.
    """
    if universe_rule:
        symbols = ["DEMO"]  # placeholder; build_universe resolves members
        rule_obj = universe_builder.UniverseRule.from_dict(dict(universe_rule))
        universe_key = rule_obj.pool_id()
        universe_rule = rule_obj.to_dict()  # embed the canonical form
    else:
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
                start=start, end=end, universe_rule=universe_rule,
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
                start=start, end=end, universe_rule=universe_rule,
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
{"universe_rule": {...}}        the text describes a PROCEDURAL/DYNAMIC rule
                                 for selecting the universe (e.g. "daily pick
                                 ETFs covering the top 80% of turnover,
                                 excluding leveraged/inverse/newly-listed/
                                 suspended names") rather than naming explicit
                                 tickers. Shape: {"asset_type": "etf",
                                 "base_pool": "all_etf",
                                 "rebalance": {"freq": "daily"} or
                                 {"freq": "annual", "month": 5, "day": 1},
                                 "filters": [{"type": "<name>", ...params}]}
                                 using ONLY these filter types, never invent
                                 one, and never include a type flagged
                                 [NOT IMPLEMENTED] (leave it out instead):
                                 {rule_catalog}
{"universe": null}              no specific universe/symbols/rule mentioned"""


def _parse_universe_reply(reply: str, rule_start: str) -> Dict[str, Any]:
    try:
        payload = json.loads(_strip_fence(reply))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"reply was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("reply JSON must be an object")
    raw_rule = payload.get("universe_rule")
    if raw_rule:
        # the model isn't asked for a start date (the caller's own `start`
        # governs how far back the rule builds) -- fill it in before
        # validating, or the "start is required" check would reject every
        # otherwise-valid reply
        raw_rule.setdefault("start", rule_start)
        try:
            universe_builder.UniverseRule.from_dict(raw_rule)  # validate only
        except universe_builder.UniverseRuleError as exc:
            raise ValueError(f"universe_rule invalid: {exc}") from exc
    return payload


@dataclass
class UniverseHint:
    symbols: Optional[List[str]] = None
    rule: Optional[Dict[str, Any]] = None  # validated universe_rule dict
    note: str = ""


def extract_universe_hint(
    text: str, client: LLMClient, start: str = "2025-01-01", end: str = "2025-12-31",
) -> Optional[UniverseHint]:
    """Explicit universe/symbols/rule named in `text`, else None.

    None means "use the caller's own default" — whether because the text
    genuinely names nothing, or because this best-effort enrichment step
    itself failed. Unlike parse_prompt/llm_extract_ideas (the primary
    feature, which must never degrade silently), this is a secondary hint
    on top of an ingestion that already succeeded, so a failure here logs
    a warning and falls back to the caller's default rather than failing
    the whole request.

    A rule-based hint is VALIDATED but not built here (analysis proposes,
    execution builds): the rule dict rides along in `rule`, the caller
    embeds it as data.universe_rule, and the runner's build_universe step
    constructs the pool only after the user confirms the suggested task.
    """
    if not text.strip():
        return None
    try:
        # .replace, not .format: the template is full of literal JSON-example
        # braces that would all need doubling for str.format to leave alone
        system_prompt = _UNIVERSE_SYSTEM.replace(
            "{rule_catalog}", universe_builder.filter_catalog_prompt()
        )
        payload = query_structured(
            client, text[:2000], system_prompt,
            lambda reply: _parse_universe_reply(reply, start), max_attempts=2,
        )
    except (LLMError, ValueError) as exc:
        logger.warning("universe hint extraction failed, using default: %s", exc)
        return None

    if payload.get("universe") == "pool24":
        return UniverseHint(symbols=list(DEFAULT_ETF_UNIVERSE))
    symbols = payload.get("symbols")
    if isinstance(symbols, list) and symbols:
        return UniverseHint(symbols=[str(s) for s in symbols])

    raw_rule = payload.get("universe_rule")
    if raw_rule:
        raw_rule.setdefault("start", start)  # _parse_universe_reply already
        try:                                  # did this; kept for direct callers
            rule = universe_builder.UniverseRule.from_dict(raw_rule)
        except universe_builder.UniverseRuleError as exc:
            # best-effort enrichment: log and fall back, don't fail ingestion
            logger.warning("universe rule invalid, using default: %s", exc)
            return None
        return UniverseHint(
            rule=rule.to_dict(),  # canonical form, not the model's raw shape
            note=universe_builder.describe_rule(
                rule, rule.pool_id(), None, "zh", end=end
            ),
        )
    return None
