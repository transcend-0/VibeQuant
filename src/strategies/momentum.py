"""Time-series momentum (动量策略)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import StrategyTemplate, register


def build(params: Dict[str, Any]):
    lookback = int(params["lookback"])
    threshold = float(params["threshold"])

    def signal(closes: List[float], position: float) -> Optional[float]:
        if len(closes) < lookback + 1:
            return None
        ret = closes[-1] / closes[-1 - lookback] - 1.0
        if position <= 0 and ret > threshold:
            return 1.0
        if position > 0 and ret < 0:
            return 0.0
        return None

    return signal


register(
    StrategyTemplate(
        name="momentum",
        summary_en="Time-series momentum: long when trailing return beats threshold.",
        summary_zh="时间序列动量：回看收益超过阈值做多，转负平仓。",
        defaults={"lookback": 20, "threshold": 0.02},
        build=build,
        warmup=lambda p: int(p["lookback"]) + 1,
    )
)
