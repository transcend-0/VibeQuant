"""Thin adapter over akquant.

This is the ONLY module in VibeQuant that imports akquant. Everything
above it speaks in TaskSpec / signal functions / plain dataclasses, so
akquant stays untouched and swappable.

The adapter:
  1. wraps a pure-Python signal function into a generic akquant Strategy,
  2. calls akquant.run_backtest with cost/risk settings from the DSL,
  3. flattens the result into an engine-agnostic BacktestOutput.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

import akquant as aq

from ..dsl import TaskSpec
from ..strategies import SignalFn, build_signal, warmup_bars


@dataclass
class BacktestOutput:
    """Engine-agnostic backtest result consumed by report/risk/memory."""

    metrics: Dict[str, Optional[float]]
    equity_curve: pd.Series  # index: datetime, values: total equity
    trades: pd.DataFrame
    num_trades: int
    initial_cash: float
    engine: str = "akquant"
    engine_version: str = ""
    raw: Any = field(default=None, repr=False)  # akquant BacktestResult

    def to_summary(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "engine_version": self.engine_version,
            "initial_cash": self.initial_cash,
            "num_trades": self.num_trades,
            "metrics": self.metrics,
        }


class _SignalStrategy(aq.Strategy):
    """Generic bridge: rolling closes -> signal fn -> target-percent orders."""

    def __init__(
        self,
        signal_fn: SignalFn = None,  # type: ignore[assignment]
        max_position_pct: float = 0.95,
        history_cap: int = 512,
    ) -> None:
        super().__init__()
        self._signal_fn = signal_fn
        self._max_position_pct = max_position_pct
        self._history_cap = history_cap
        self._closes: Dict[str, List[float]] = {}

    def on_bar(self, bar: Any) -> None:  # noqa: D102
        symbol = bar.symbol
        closes = self._closes.setdefault(symbol, [])
        closes.append(float(bar.close))
        if len(closes) > self._history_cap:
            del closes[: len(closes) - self._history_cap]

        position = float(self.get_position(symbol))
        target = self._signal_fn(closes, position)
        if target is None:
            return
        target = max(0.0, min(float(target), 1.0)) * self._max_position_pct
        if target == 0.0 and position <= 0:
            return
        self.order_target_percent(target_percent=target, symbol=symbol)


class _RotationStrategy(aq.Strategy):
    """Cross-sectional rotation: hold the top-K symbols by factor score.

    Scores are precomputed from the same bars the engine replays (factor
    value at date t uses data up to t's close; orders fill at the next
    open, so there is no lookahead). Rebalances every `rebalance_days`
    trading days.
    """

    def __init__(
        self,
        scores: Dict[str, Dict[str, float]] = None,  # date -> {symbol: score}
        top_k: int = 5,
        rebalance_days: int = 5,
        max_position_pct: float = 0.95,
    ) -> None:
        super().__init__()
        self._scores = scores or {}
        self._top_k = max(1, int(top_k))
        self._rebalance_days = max(1, int(rebalance_days))
        self._max_position_pct = max_position_pct
        self._day_count = -1
        self._pending_target: Optional[Dict[str, float]] = None
        self._held: set = set()

    def on_daily_rebalance(self, trading_date, timestamp) -> None:  # noqa: D102
        # phase 2: entries queued at the previous session — exit proceeds
        # have settled by now, so buys cannot be rejected for cash
        if self._pending_target is not None:
            self.order_target_weights(
                target_weights=self._pending_target,
                liquidate_unmentioned=False,
                rebalance_tolerance=0.01,
            )
            self._held = set(self._pending_target)
            self._pending_target = None

        self._day_count += 1
        if self._day_count % self._rebalance_days:
            return
        key = str(trading_date)[:10]
        day_scores = self._scores.get(key) or {}
        if len(day_scores) < self._top_k:
            return  # warmup: not enough valid scores yet
        ranked = sorted(day_scores, key=day_scores.get, reverse=True)
        weight = self._max_position_pct / self._top_k
        target = {s: weight for s in ranked[: self._top_k]}

        # phase 1: exit names leaving the portfolio today; enter tomorrow
        # (two-phase rebalance, as live rotation desks do)
        for symbol in self._held - set(target):
            if float(self.get_position(symbol)) > 0:
                self.close_position(symbol)
        self._pending_target = target


def _rotation_scores(
    expressions: List[str], data: Dict[str, pd.DataFrame]
) -> Dict[str, Dict[str, float]]:
    """Precompute per-date combined factor scores (z-scored mean)."""
    from . import akquant_factor

    panel = akquant_factor.compute_factors(expressions, data)
    names = [
        akquant_factor.split_named_expression(raw, i)[0]
        for i, raw in enumerate(expressions)
    ]
    for name in names:  # per-date z-score, then average across factors
        grouped = panel.groupby("date")[name]
        std = grouped.transform("std").replace(0.0, pd.NA)
        panel[name] = (panel[name] - grouped.transform("mean")) / std
    panel["_score"] = panel[names].mean(axis=1)

    scores: Dict[str, Dict[str, float]] = {}
    for (date, symbol), value in panel.set_index(["date", "symbol"])["_score"].items():
        if pd.isna(value):
            continue
        scores.setdefault(str(date)[:10], {})[symbol] = float(value)
    return scores


ROTATION_DEFAULTS = {
    "expressions": ["Mom20 = Delta(Close, 20) / Delay(Close, 20)"],
    "top_k": 5,
    "rebalance_days": 5,
}


def _metric(metrics: Any, name: str) -> Optional[float]:
    try:
        value = getattr(metrics, name)
    except AttributeError:
        return None
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def run_backtest(spec: TaskSpec, data: Dict[str, pd.DataFrame]) -> BacktestOutput:
    """Run one backtest for a TaskSpec over pre-loaded per-symbol bars."""
    symbols = list(data.keys())

    if spec.strategy.name == "factor_rotation":
        params = {**ROTATION_DEFAULTS, **spec.strategy.params}
        strategy: Any = _RotationStrategy(
            scores=_rotation_scores(
                [str(e) for e in params["expressions"]], data
            ),
            top_k=int(params["top_k"]),
            rebalance_days=int(params["rebalance_days"]),
            max_position_pct=spec.risk.max_position_pct,
        )
    else:
        signal_fn = build_signal(spec.strategy.name, spec.strategy.params)
        warmup = warmup_bars(spec.strategy.name, spec.strategy.params)
        strategy = _SignalStrategy(
            signal_fn=signal_fn,
            max_position_pct=spec.risk.max_position_pct / max(len(symbols), 1),
            history_cap=max(warmup + 64, 256),
        )

    kwargs: Dict[str, Any] = dict(
        data=data if len(symbols) > 1 else next(iter(data.values())),
        strategy=strategy,
        symbols=symbols if len(symbols) > 1 else symbols[0],
        initial_cash=spec.execution.initial_cash,
        commission_rate=spec.execution.commission_rate,
        stamp_tax_rate=spec.execution.stamp_tax_rate,
        slippage={
            "type": "percent",
            "value": spec.execution.slippage_bps / 10_000.0,
        },
        t_plus_one=spec.execution.t_plus_one,
        show_progress=False,
    )
    if spec.risk.max_order_value:
        kwargs["strategy_max_order_value"] = {
            "default": float(spec.risk.max_order_value)
        }

    result = aq.run_backtest(**kwargs)

    metrics = result.metrics
    trades = result.trades_df
    num_trades = 0 if trades is None or trades.empty else int(len(trades))

    flat = {
        "total_return_pct": _metric(metrics, "total_return_pct"),
        "annualized_return": _metric(metrics, "annualized_return"),
        "sharpe_ratio": _metric(metrics, "sharpe_ratio"),
        "sortino_ratio": _metric(metrics, "sortino_ratio"),
        "max_drawdown_pct": _metric(metrics, "max_drawdown_pct"),
        "win_rate": _metric(metrics, "win_rate"),
    }

    return BacktestOutput(
        metrics=flat,
        equity_curve=result.equity_curve,
        trades=trades if trades is not None else pd.DataFrame(),
        num_trades=num_trades,
        initial_cash=spec.execution.initial_cash,
        engine_version=getattr(aq, "__version__", ""),
        raw=result,
    )


def write_html_report(
    output: BacktestOutput,
    path: str,
    title: str = "VibeQuant Strategy Report",
    market_data: Optional[Dict[str, pd.DataFrame]] = None,
    benchmark: Optional[pd.Series] = None,
) -> Optional[str]:
    """Render akquant's native HTML report (plotly). Best-effort.

    With a benchmark return series the report adds the benchmark block
    (excess return, alpha/beta, information ratio, tracking error).
    """
    try:
        output.raw.report(
            title=title,
            filename=path,
            show=False,
            market_data=market_data,
            benchmark=benchmark,
        )
        return path
    except Exception:
        return None
