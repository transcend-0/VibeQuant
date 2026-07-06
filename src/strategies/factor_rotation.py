"""Multi-factor rotation (多因子轮动) — the factor->strategy bridge.

Unlike the other templates this is NOT a per-symbol signal function:
scores are cross-sectional (top-K by combined factor score, rebalanced
every M days), so the akquant adapter implements it directly
(`_RotationStrategy`); `build` here exists only to satisfy the template
interface and must never be called.
"""

from __future__ import annotations

from typing import Any, Dict

from . import StrategyTemplate, register


def build(params: Dict[str, Any]):
    raise RuntimeError(
        "factor_rotation is cross-sectional and runs via the akquant "
        "adapter, not as a per-symbol signal function"
    )


register(
    StrategyTemplate(
        name="factor_rotation",
        summary_en="Multi-factor rotation: hold top-K symbols by combined "
        "factor score, rebalance every M days.",
        summary_zh="多因子轮动：按因子综合得分持有前K只，每M日调仓。",
        defaults={
            "expressions": ["Mom20 = Delta(Close, 20) / Delay(Close, 20)"],
            "top_k": 5,
            "rebalance_days": 5,
        },
        build=build,
        warmup=lambda p: 60,
    )
)
