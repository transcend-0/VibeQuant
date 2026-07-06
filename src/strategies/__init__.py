"""Strategy templates.

Each template is *pure Python*: a params schema plus a signal function
over a rolling close-price window. No akquant imports here — the adapter
wraps a signal function into an engine strategy. This keeps templates
unit-testable and engine-agnostic.

Signal contract:
    signal(closes: list[float], position: float) -> float | None
      - closes: chronological closes up to and including the current bar
      - position: current quantity held for this symbol
      - returns a target weight in [0, 1] (fraction of equity) or None
        for "no change".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

SignalFn = Callable[[List[float], float], Optional[float]]


@dataclass
class StrategyTemplate:
    name: str
    summary_en: str
    summary_zh: str
    defaults: Dict[str, Any] = field(default_factory=dict)
    build: Callable[[Dict[str, Any]], SignalFn] = None  # type: ignore[assignment]
    warmup: Callable[[Dict[str, Any]], int] = lambda p: 0


REGISTRY: Dict[str, StrategyTemplate] = {}


def register(template: StrategyTemplate) -> StrategyTemplate:
    REGISTRY[template.name] = template
    return template


def get_template(name: str) -> StrategyTemplate:
    if name not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown strategy {name!r}; available: {known}")
    return REGISTRY[name]


def resolve_params(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Merge user params over template defaults, rejecting unknown keys."""
    tpl = get_template(name)
    unknown = set(params) - set(tpl.defaults)
    if unknown:
        raise KeyError(
            f"unknown params for {name!r}: {sorted(unknown)}; "
            f"accepted: {sorted(tpl.defaults)}"
        )
    merged = dict(tpl.defaults)
    merged.update(params)
    return merged


def build_signal(name: str, params: Dict[str, Any]) -> SignalFn:
    tpl = get_template(name)
    return tpl.build(resolve_params(name, params))


def warmup_bars(name: str, params: Dict[str, Any]) -> int:
    tpl = get_template(name)
    return tpl.warmup(resolve_params(name, params))


# Import concrete templates so registration runs on package import.
from . import (  # noqa: E402,F401
    bollinger,
    buy_hold,
    factor_rotation,
    ma_cross,
    momentum,
    rsi_reversion,
)
