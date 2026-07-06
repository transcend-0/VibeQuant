"""Dual moving-average crossover (双均线金叉死叉)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import StrategyTemplate, register


def _sma(closes: List[float], n: int) -> float:
    return sum(closes[-n:]) / n


def build(params: Dict[str, Any]):
    fast, slow = int(params["fast"]), int(params["slow"])
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")

    def signal(closes: List[float], position: float) -> Optional[float]:
        if len(closes) < slow + 1:
            return None
        fast_now, slow_now = _sma(closes, fast), _sma(closes, slow)
        fast_prev = _sma(closes[:-1], fast)
        slow_prev = _sma(closes[:-1], slow)
        if fast_prev <= slow_prev and fast_now > slow_now:
            return 1.0  # golden cross -> long
        if fast_prev >= slow_prev and fast_now < slow_now:
            return 0.0  # death cross -> flat
        return None

    return signal


register(
    StrategyTemplate(
        name="ma_cross",
        summary_en="Dual SMA crossover: long on golden cross, flat on death cross.",
        summary_zh="双均线策略：金叉买入，死叉平仓。",
        defaults={"fast": 5, "slow": 20},
        build=build,
        warmup=lambda p: int(p["slow"]) + 1,
    )
)
