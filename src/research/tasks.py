"""Turn extracted research Ideas into runnable TaskSpecs."""

from __future__ import annotations

import re
from typing import List, Optional

from ..dsl import DataSpec, FactorSpec, ReportSpec, StrategySpec, TaskSpec
from .ingest import Idea

# The built-in categorized ETF pool (abroad/commodity/bond/index/industry)
# is the default cross-section when the user names no universe.
from ..data_sources.etf_pool import POOL_SYMBOLS as DEFAULT_ETF_UNIVERSE  # noqa: E402


def idea_to_taskspec(
    idea: Idea,
    mode: str,  # "factor" | "strategy" — the UI mode requesting the task
    universe: Optional[List[str]] = None,
    start: str = "2021-01-01",
    end: str = "2024-12-31",
    language: str = "en",
    intent: str = "",
) -> TaskSpec:
    """Build a TaskSpec for an Idea in the requested research mode.

    An idea whose kind doesn't match the mode is still usable when it has
    the other side's material (e.g. a factor idea backed by a strategy
    template); otherwise a sensible default is chosen.
    """
    symbols = [str(s) for s in (universe or DEFAULT_ETF_UNIVERSE)]
    slug = re.sub(r"[^a-zA-Z0-9-]+", "-", idea.key)[:30].strip("-")

    if mode == "factor":
        expressions = idea.factor_expressions or _fallback_expressions(idea)
        spec = TaskSpec(
            name=f"factor-{slug}",
            kind="factor",
            intent=intent or idea.title_en,
            data=DataSpec(source="etf", symbols=symbols, start=start, end=end),
            factor=FactorSpec(expressions=expressions),
            report=ReportSpec(language=language),
        )
    else:
        strategy_name = idea.strategy_name or "momentum"
        spec = TaskSpec(
            name=f"strategy-{slug}",
            kind="strategy",
            intent=intent or idea.title_en,
            data=DataSpec(
                source="etf", symbols=symbols[:1], start=start, end=end
            ),
            strategy=StrategySpec(
                name=strategy_name, params=dict(idea.strategy_params)
            ),
            report=ReportSpec(language=language),
        )
    spec.validate()
    return spec


def _fallback_expressions(idea: Idea) -> List[str]:
    """Factor mode was requested but the idea carries no expressions."""
    return ["Mom20 = Delta(Close, 20) / Delay(Close, 20)"]
