"""Thin adapter over akquant.

This is the ONLY module in VibeQuant that imports akquant. Everything
above it speaks in TaskSpec / plain dataclasses, so akquant stays
untouched and swappable.

Execution contract: `spec.strategy.params["source"]` is Python source
defining a class that inherits `Strategy` (an alias for `aq.Strategy`
injected into the exec namespace, alongside `BaseStrategy`, `np`, `pd`,
`SYMBOLS`/`START`/`END`, and -- when `spec.strategy.params["expressions"]`
is set -- a precomputed `FACTOR_SCORES` map). The adapter execs that
source, hands the resulting class straight to `aq.run_backtest`, and
flattens the result into an engine-agnostic BacktestOutput. There is no
per-symbol callback layer any more: a single-symbol rule and a
cross-sectional rotation both just look like an ordinary akquant Strategy
subclass, with the freedom that implies (and the same accepted,
unsandboxed-exec risk as `src/strategies/custom.py` documents).
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import akquant as aq

from ..dsl import TaskSpec


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


class _DateKeyedScores(dict):
    """FACTOR_SCORES, tolerant of how the strategy code looks a date up.

    Keys are stored as "YYYY-MM-DD" strings, but `on_daily_rebalance`'s
    `trading_date` argument is a `datetime.date` (and LLM-authored code
    understandably sometimes looks it up directly, or via a pandas
    Timestamp, instead of converting with `str(trading_date)[:10]` first
    as the docs ask). A plain dict silently returns the default ({}) on a
    type-mismatched key -- no exception, just an empty score dict every
    single day, so the strategy quietly never rebalances and returns 0%
    with 0 trades. Normalizing the lookup key here turns that into a
    reliable match regardless of what the caller passes.
    """

    @staticmethod
    def _norm(key: Any) -> str:
        return str(key)[:10]

    def get(self, key: Any, default=None) -> Any:  # noqa: D102
        return super().get(self._norm(key), default)

    def __getitem__(self, key: Any) -> Any:
        return super().__getitem__(self._norm(key))

    def __contains__(self, key: Any) -> bool:  # noqa: D105
        return super().__contains__(self._norm(key))


def _rotation_scores(
    expressions: List[str], data: Dict[str, pd.DataFrame]
) -> Dict[str, Dict[str, float]]:
    """Precompute per-date combined factor scores (z-scored mean).

    Bridges validated WorldQuant-style factor expressions (same grammar as
    factor research) into a strategy: the score at date t only uses data
    up to t's close, so a strategy reading FACTOR_SCORES has no lookahead.
    """
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

    scores: Dict[str, Dict[str, float]] = _DateKeyedScores()
    for (date, symbol), value in panel.set_index(["date", "symbol"])["_score"].items():
        if pd.isna(value):
            continue
        scores.setdefault(str(date)[:10], {})[symbol] = float(value)
    return scores


def load_user_strategy(
    source: str,
    *,
    symbols: List[str],
    start: Optional[str],
    end: Optional[str],
    factor_scores: Optional[Dict[str, Dict[str, float]]] = None,
):
    """Exec `source` and return the akquant Strategy subclass it defines.

    `source` must define a class inheriting `Strategy`/`BaseStrategy`
    (both aliases for `aq.Strategy` in the exec namespace). If more than
    one such class is defined, the last one *in source order* wins (helper
    base classes defined earlier in the same source don't get picked over
    the final concrete strategy) -- source order, not namespace dict
    iteration order, because rebinding a name that already exists in the
    namespace (e.g. a class literally named `Strategy`) keeps its original
    dict position instead of moving to the end.
    """
    namespace: Dict[str, Any] = {
        "aq": aq,
        "akquant": aq,
        "Strategy": aq.Strategy,
        "BaseStrategy": aq.Strategy,
        "np": np,
        "pd": pd,
        "SYMBOLS": list(symbols),
        "START": start,
        "END": end,
        "FACTOR_SCORES": factor_scores if factor_scores is not None else _DateKeyedScores(),
    }
    compiled = compile(source, "<strategy_source>", "exec")
    exec(compiled, namespace)  # noqa: S102 -- intentionally unsandboxed, see strategies/custom.py

    candidates = {
        k: v for k, v in namespace.items()
        if isinstance(v, type) and issubclass(v, aq.Strategy) and v is not aq.Strategy
    }
    if not candidates:
        raise ValueError(
            "strategy source must define a class inheriting from Strategy "
            "(e.g. 'class Strategy(BaseStrategy): def on_bar(self, bar): ...')"
        )
    order = [
        node.name for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name in candidates
    ]
    strategy_cls = candidates[order[-1]] if order else next(iter(candidates.values()))

    # self.symbols/.start/.end as a convenience: LLM-authored code
    # naturally reaches for `self.symbols` (mirroring akquant's own
    # examples, where it's an explicit constructor arg) even though the
    # documented mechanism is the SYMBOLS/START/END globals -- rather than
    # relying on prompt wording alone, guarantee the attributes exist
    # before the class's own __init__ runs, so it works either way.
    # Anything the user's own __init__ sets afterward simply overrides
    # this default, same as if they'd set it themselves.
    #
    # Signature must stay a bare `(self)`, matching the documented no-arg
    # __init__ contract: akquant inspects the constructor signature and,
    # if it sees `**kwargs`, forwards its OWN context kwargs (e.g.
    # `symbols=`) straight through unfiltered -- which then blows up
    # against the user's original no-arg `__init__`. A plain `(self)`
    # keeps akquant's kwarg-filtering as "nothing accepted", exactly like
    # an unwrapped no-arg __init__.
    orig_init = strategy_cls.__init__

    def _init_with_defaults(self):
        self.symbols = list(symbols)
        self.start = start
        self.end = end
        orig_init(self)

    strategy_cls.__init__ = _init_with_defaults
    return strategy_cls


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
    params = spec.strategy.params or {}

    factor_scores = None
    expressions = params.get("expressions")
    if expressions:
        factor_scores = _rotation_scores([str(e) for e in expressions], data)

    strategy_cls = load_user_strategy(
        str(params.get("source") or "class Strategy(BaseStrategy):\n    pass\n"),
        symbols=symbols,
        start=spec.data.start,
        end=spec.data.end,
        factor_scores=factor_scores,
    )

    kwargs: Dict[str, Any] = dict(
        data=data if len(symbols) > 1 else next(iter(data.values())),
        strategy=strategy_cls,
        symbols=symbols if len(symbols) > 1 else symbols[0],
        initial_cash=spec.execution.initial_cash,
        commission_rate=spec.execution.commission_rate,
        stamp_tax_rate=spec.execution.stamp_tax_rate,
        slippage={
            "type": "percent",
            "value": spec.execution.slippage_bps / 10_000.0,
        },
        t_plus_one=spec.execution.t_plus_one,
        history_depth=int(params.get("history_depth", 300)),
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
