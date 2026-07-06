"""Buy and hold (买入持有) — the baseline every strategy must beat."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import StrategyTemplate, register


def build(params: Dict[str, Any]):
    def signal(closes: List[float], position: float) -> Optional[float]:
        if position <= 0:
            return 1.0
        return None

    return signal


register(
    StrategyTemplate(
        name="buy_hold",
        summary_en="Buy on the first bar and hold. Baseline benchmark.",
        summary_zh="首根K线买入并持有，作为基准。",
        defaults={},
        build=build,
        warmup=lambda p: 0,
    )
)
