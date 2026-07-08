"""Strategy skeletons for the workbench.

There is exactly one execution contract (see
`src/adapters/akquant_engine.py:_load_user_strategy`): `strategy.params["source"]`
is Python source defining a class that inherits `Strategy` (an alias for
akquant's own `aq.Strategy` injected into the exec scope), using akquant's
real API directly -- `on_bar`, `self.get_position(symbol)`, `self.buy` /
`self.sell` / `self.close_position`, `self.order_target_percent`,
`self.order_target_weights`, `self.get_history(count, symbol, field)`, and
so on. There is no restrictive `signal(closes, position)` callback layer
any more: a per-symbol rule strategy and a cross-sectional multi-factor
rotation both just look like a normal akquant `Strategy` subclass, with
the same freedom you'd have writing directly against akquant.

This module therefore no longer builds executable signal functions -- it
only holds default "starter code" skeletons shown in the workbench, so a
fresh task has something concrete to read and edit rather than a blank
textarea. Execution never goes through `REGISTRY`; the adapter execs
whatever source ends up in `params["source"]` regardless of `name`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

REGISTRY: Dict[str, "StrategySkeleton"] = {}


@dataclass
class StrategySkeleton:
    name: str
    summary_en: str
    summary_zh: str
    source: str  # default params["source"] -- a full akquant Strategy subclass
    params: Dict[str, Any] = field(default_factory=dict)  # extra companion params


def register(skeleton: StrategySkeleton) -> StrategySkeleton:
    REGISTRY[skeleton.name] = skeleton
    return skeleton


def get_skeleton(name: str) -> StrategySkeleton:
    if name not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown strategy skeleton {name!r}; available: {known}")
    return REGISTRY[name]


# Import concrete skeletons so registration runs on package import.
from . import (  # noqa: E402,F401
    custom,
    factor_rotation,
)
