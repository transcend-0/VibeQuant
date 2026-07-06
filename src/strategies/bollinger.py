"""Bollinger band reversion (布林带回归)."""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from . import StrategyTemplate, register


def build(params: Dict[str, Any]):
    period = int(params["period"])
    num_std = float(params["num_std"])

    def signal(closes: List[float], position: float) -> Optional[float]:
        if len(closes) < period:
            return None
        window = closes[-period:]
        mid = sum(window) / period
        std = statistics.pstdev(window)
        if std == 0:
            return None
        lower = mid - num_std * std
        price = closes[-1]
        if position <= 0 and price < lower:
            return 1.0  # below lower band -> buy the dip
        if position > 0 and price >= mid:
            return 0.0  # revert to mid -> take profit
        return None

    return signal


register(
    StrategyTemplate(
        name="bollinger",
        summary_en="Bollinger reversion: buy below lower band, exit at the middle band.",
        summary_zh="布林带回归：跌破下轨买入，回到中轨平仓。",
        defaults={"period": 20, "num_std": 2.0},
        build=build,
        warmup=lambda p: int(p["period"]),
    )
)
