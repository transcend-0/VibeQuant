"""RSI mean reversion (RSI 超卖反弹)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import StrategyTemplate, register


def _rsi(closes: List[float], period: int) -> float:
    window = closes[-(period + 1):]
    gains = losses = 0.0
    for prev, cur in zip(window, window[1:]):
        delta = cur - prev
        if delta >= 0:
            gains += delta
        else:
            losses -= delta
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100.0 - 100.0 / (1.0 + rs)


def build(params: Dict[str, Any]):
    period = int(params["period"])
    oversold = float(params["oversold"])
    overbought = float(params["overbought"])
    if not oversold < overbought:
        raise ValueError("oversold must be < overbought")

    def signal(closes: List[float], position: float) -> Optional[float]:
        if len(closes) < period + 1:
            return None
        rsi = _rsi(closes, period)
        if position <= 0 and rsi < oversold:
            return 1.0
        if position > 0 and rsi > overbought:
            return 0.0
        return None

    return signal


register(
    StrategyTemplate(
        name="rsi_reversion",
        summary_en="RSI mean reversion: buy oversold, exit overbought.",
        summary_zh="RSI 均值回归：超卖买入，超买卖出。",
        defaults={"period": 14, "oversold": 30.0, "overbought": 70.0},
        build=build,
        warmup=lambda p: int(p["period"]) + 1,
    )
)
