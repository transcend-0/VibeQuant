"""Multi-factor rotation (多因子轮动) skeleton -- cross-sectional, editable.

Unlike the old fixed adapter-side implementation, this is now ordinary
Python: the class body, the ranking rule, the rebalance cadence are all
user/LLM-editable. The one piece of scaffolding the adapter still
provides is `FACTOR_SCORES` -- when `params["expressions"]` is set (a
list of WorldQuant-style factor formulas, the same grammar used in factor
research), the adapter precomputes a `{date_str: {symbol: score}}` map
from real historical bars before the backtest starts and injects it as a
global, so a validated alpha expression can seed the ranking below
without hand-translating it into `get_history` calls. Everything past
that point -- how scores turn into positions -- is free-form code.
"""

from __future__ import annotations

from . import StrategySkeleton, register

DEFAULT_SOURCE = '''class Strategy(BaseStrategy):
    """Hold the top-K symbols by FACTOR_SCORES, rebalance every N days."""

    top_k = 5
    rebalance_days = 5
    max_position_pct = 0.95

    def __init__(self):
        super().__init__()
        self.symbols = list(SYMBOLS)
        self.day_count = -1
        self.pending_target = None
        self.held = set()

    def on_daily_rebalance(self, trading_date, timestamp):
        # two-phase rebalance: exit names leaving the portfolio today,
        # enter new names tomorrow -- exit proceeds settle before the buys
        # are submitted, which avoids cash-rejected orders.
        if self.pending_target is not None:
            self.order_target_weights(
                target_weights=self.pending_target,
                liquidate_unmentioned=False,
                rebalance_tolerance=0.01,
            )
            self.held = set(self.pending_target)
            self.pending_target = None

        self.day_count += 1
        if self.day_count % self.rebalance_days:
            return
        day_scores = FACTOR_SCORES.get(str(trading_date)[:10]) or {}
        if len(day_scores) < self.top_k:
            return  # warmup: not enough valid scores yet
        ranked = sorted(day_scores, key=day_scores.get, reverse=True)
        weight = self.max_position_pct / self.top_k
        target = {s: weight for s in ranked[: self.top_k]}

        for symbol in self.held - set(target):
            if float(self.get_position(symbol)) > 0:
                self.close_position(symbol)
        self.pending_target = target
'''

register(
    StrategySkeleton(
        name="factor_rotation",
        summary_en="Multi-factor rotation: hold top-K symbols by combined "
        "factor score, rebalance every N days. Fully editable.",
        summary_zh="多因子轮动：按因子综合得分持有前K只，每N日调仓；代码完全可编辑。",
        source=DEFAULT_SOURCE,
        params={"expressions": ["Mom20 = Delta(Close, 20) / Delay(Close, 20)"]},
    )
)
