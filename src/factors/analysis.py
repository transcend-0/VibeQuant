"""Factor analysis: IC/IR, quantile layering, long-short spread.

Pure pandas — no engine imports. Input is the wide factor panel from
adapters.akquant_factor plus the original price frames; output is a
FactorReport of plain dicts/lists ready for JSON and the Web UI.

Methodology notes (kept deliberately standard):
- Forward return over `forward_days`, close-to-close, per symbol.
- IC = per-date cross-sectional Spearman rank correlation between the
  factor value and the forward return (computed on non-overlapping dates,
  stepping by forward_days, to avoid autocorrelated observations).
- Layered backtest: on each rebalance date, symbols are bucketed into
  `quantiles` groups by factor value; each layer earns the equal-weighted
  forward return of its bucket. Long-short = top layer minus bottom layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class FactorStats:
    name: str
    ic_mean: Optional[float] = None
    ic_std: Optional[float] = None
    icir: Optional[float] = None
    ic_positive_rate: Optional[float] = None
    ic_t_stat: Optional[float] = None
    n_periods: int = 0
    long_short_total_return: Optional[float] = None
    top_layer_total_return: Optional[float] = None
    bottom_layer_total_return: Optional[float] = None
    rank_autocorr: Optional[float] = None  # proxy for turnover (high = stable)
    constrained_ls_return: Optional[float] = None  # LS with max_position/max_trade
    mean_turnover: Optional[float] = None  # of the constrained LS portfolio

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class FactorReport:
    stats: List[FactorStats] = field(default_factory=list)
    # per-factor plotting payloads for the UI:
    #   ic_series[name] = {"dates": [...], "ic": [...]}
    #   layer_curves[name] = {"dates": [...], "layers": {"Q1": [...], ...},
    #                          "long_short": [...]}
    ic_series: Dict[str, Dict[str, list]] = field(default_factory=dict)
    layer_curves: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": [s.to_dict() for s in self.stats],
            "ic_series": self.ic_series,
            "layer_curves": self.layer_curves,
            "warnings": self.warnings,
        }


def _forward_returns(
    frames: Dict[str, pd.DataFrame], forward_days: int
) -> pd.DataFrame:
    parts = []
    for symbol, df in frames.items():
        closes = df.sort_values("date")[["date", "close"]].copy()
        closes["fwd_ret"] = (
            closes["close"].shift(-forward_days) / closes["close"] - 1.0
        )
        closes["symbol"] = symbol
        parts.append(closes[["date", "symbol", "fwd_ret"]])
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def apply_factor_ops(
    panel: pd.DataFrame,
    factor_names: List[str],
    truncation: float = 0.0,
    neutralization: str = "none",
    decay: int = 0,
    groups: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """WorldQuant-Brain style post-processing, in canonical order.

    1. truncation      — winsorize each date's cross-section at the given
                         fraction per tail (limits single-name dominance)
    2. neutralization  — per-date cross-section: demean (market-neutral),
                         industry (demean within groups), zscore, or
                         rank (uniform [0,1])
    3. decay           — per-symbol linear decay over N days: weighted
                         average with weights N..1 (turnover smoothing)

    groups: symbol -> industry/category key, required for
    neutralization="industry" (unknown symbols share group "other").
    """
    out = panel.copy()
    if neutralization == "industry":
        mapping = groups or {}
        out["_group"] = out["symbol"].map(lambda s: mapping.get(str(s), "other"))
    for name in factor_names:
        col = out[name]

        if truncation > 0:
            def _trunc(s: pd.Series) -> pd.Series:
                lo, hi = s.quantile(truncation), s.quantile(1 - truncation)
                return s.clip(lo, hi)

            col = col.groupby(out["date"]).transform(_trunc)

        if neutralization == "demean":
            col = col - col.groupby(out["date"]).transform("mean")
        elif neutralization == "industry":
            col = col - col.groupby([out["date"], out["_group"]]).transform("mean")
        elif neutralization == "zscore":
            grouped = col.groupby(out["date"])
            std = grouped.transform("std").replace(0.0, pd.NA)
            col = (col - grouped.transform("mean")) / std
        elif neutralization == "rank":
            col = col.groupby(out["date"]).rank(pct=True)

        if decay > 1:
            weights = list(range(decay, 0, -1))  # N..1, newest heaviest

            def _decay(s: pd.Series) -> pd.Series:
                arr = s.rolling(decay, min_periods=1).apply(
                    lambda w: (w[::-1] * weights[: len(w)]).sum()
                    / sum(weights[: len(w)]),
                    raw=True,
                )
                return arr

            out = out.sort_values(["symbol", "date"])
            col = col.reindex(out.index)
            col = col.groupby(out["symbol"]).transform(_decay)

        out[name] = col
    if "_group" in out.columns:
        out = out.drop(columns="_group")
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def constrained_long_short(
    sub: pd.DataFrame,  # columns: date, symbol, <factor>, fwd_ret (rebalance dates)
    name: str,
    quantiles: int,
    max_position: float = 0.0,
    max_trade: float = 0.0,
    adv: Optional[pd.DataFrame] = None,  # columns: date, symbol, adv20 ($ vol)
    book: float = 1_000_000.0,
) -> Dict[str, Any]:
    """Long-short portfolio with WorldQuant-style liquidity constraints.

    Target weights each rebalance: +1/n_top on the top factor quantile,
    -1/n_bottom on the bottom. Both caps are fractions of each name's
    ADV20 (20-day average dollar volume), converted to weight caps
    against the book size:

        |position notional| <= max_position * ADV20
            -> |w| <= max_position * ADV20 / book
        |traded notional per rebalance| <= max_trade * ADV20
            -> |dw| <= max_trade * ADV20 / book

    Names without ADV data are left uncapped. Caps bind when the book is
    large relative to a name's liquidity — with a small book on liquid
    ETFs they may never bind, which is correct behavior.
    """
    adv_map: Dict[tuple, float] = {}
    if adv is not None and not adv.empty:
        adv_map = {
            (row.date, row.symbol): float(row.adv20)
            for row in adv.itertuples()
            if pd.notna(row.adv20)
        }

    def cap_for(date, symbol, fraction: float) -> Optional[float]:
        if fraction <= 0:
            return None
        adv20 = adv_map.get((date, symbol))
        if adv20 is None or adv20 <= 0 or book <= 0:
            return None  # no liquidity data -> uncapped
        return fraction * adv20 / book

    weights: Dict[str, float] = {}
    curve: List[float] = []
    turnovers: List[float] = []
    binds = 0
    acc = 1.0
    for date, group in sub.groupby("date"):
        if len(group) < max(3, quantiles):
            continue
        try:
            buckets = pd.qcut(group[name], quantiles, labels=False, duplicates="drop")
        except ValueError:
            continue
        if buckets.nunique() < quantiles:
            continue
        top = group.loc[buckets == quantiles - 1, "symbol"]
        bottom = group.loc[buckets == 0, "symbol"]
        target = {s: 1.0 / len(top) for s in top}
        target.update({s: -1.0 / len(bottom) for s in bottom})

        new_weights: Dict[str, float] = {}
        turnover = 0.0
        for symbol in set(target) | set(weights):
            prev = weights.get(symbol, 0.0)
            want = target.get(symbol, 0.0)
            pos_cap = cap_for(date, symbol, max_position)
            if pos_cap is not None and abs(want) > pos_cap:
                want = max(-pos_cap, min(pos_cap, want))
                binds += 1
            step = want - prev
            trade_cap = cap_for(date, symbol, max_trade)
            if trade_cap is not None and abs(step) > trade_cap:
                step = max(-trade_cap, min(trade_cap, step))
                binds += 1
            w = prev + step
            turnover += abs(step)
            if abs(w) > 1e-12:
                new_weights[symbol] = w
        weights = new_weights
        turnovers.append(turnover / 2.0)

        ret_map = group.set_index("symbol")["fwd_ret"].to_dict()
        period_ret = sum(w * ret_map.get(s, 0.0) for s, w in weights.items())
        acc *= 1.0 + period_ret
        curve.append(round(acc, 6))

    return {
        "curve": curve,
        "total_return": round(curve[-1] - 1.0, 4) if curve else None,
        "mean_turnover": (
            round(sum(turnovers) / len(turnovers), 4) if turnovers else None
        ),
        "constraint_binds": binds,
    }


def adv20_panel(
    frames: Dict[str, pd.DataFrame], lot_multiplier: float = 1.0
) -> pd.DataFrame:
    """Per-symbol ADV20 dollar volume: rolling mean of close*volume*lot."""
    parts = []
    for symbol, df in frames.items():
        d = df.sort_values("date")[["date"]].copy()
        d["adv20"] = (
            (df["close"] * df["volume"] * lot_multiplier)
            .rolling(20, min_periods=5)
            .mean()
        )
        d["symbol"] = symbol
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def analyze_factor(
    factor_panel: pd.DataFrame,
    frames: Dict[str, pd.DataFrame],
    factor_names: List[str],
    forward_days: int = 1,
    quantiles: int = 5,
    truncation: float = 0.0,
    neutralization: str = "none",
    decay: int = 0,
    groups: Optional[Dict[str, str]] = None,
    max_position: float = 0.0,
    max_trade: float = 0.0,
    adv: Optional[pd.DataFrame] = None,
    book: float = 1_000_000.0,
) -> FactorReport:
    report = FactorReport()
    n_symbols = factor_panel["symbol"].nunique()
    if n_symbols < 5:
        report.warnings.append(
            f"universe has only {n_symbols} symbols — cross-sectional IC and "
            f"{quantiles}-quantile layering are noisy; prefer 5+ symbols"
        )
    quantiles = min(quantiles, n_symbols)
    constrained = max_position > 0 or max_trade > 0

    factor_panel = apply_factor_ops(
        factor_panel, factor_names, truncation, neutralization, decay, groups
    )

    merged = factor_panel.merge(
        _forward_returns(frames, forward_days), on=["date", "symbol"], how="left"
    )

    all_dates = sorted(merged["date"].unique())
    rebalance_dates = all_dates[::forward_days]  # non-overlapping windows

    for name in factor_names:
        sub = merged[["date", "symbol", name, "fwd_ret"]].dropna()
        sub = sub[sub["date"].isin(rebalance_dates)]

        ic_dates: List[str] = []
        ic_values: List[float] = []
        layer_returns: Dict[int, List[float]] = {q: [] for q in range(quantiles)}
        ls_returns: List[float] = []
        curve_dates: List[str] = []
        prev_ranks: Optional[pd.Series] = None
        autocorrs: List[float] = []

        for date, group in sub.groupby("date"):
            if len(group) < max(3, quantiles):
                continue
            # ---- IC (Spearman = Pearson on ranks)
            fr = group[name].rank()
            rr = group["fwd_ret"].rank()
            ic = fr.corr(rr)
            if pd.notna(ic):
                ic_dates.append(str(pd.Timestamp(date).date()))
                ic_values.append(round(float(ic), 6))

            # ---- quantile layers (Q1 = lowest factor, Qn = highest)
            try:
                buckets = pd.qcut(group[name], quantiles, labels=False, duplicates="drop")
            except ValueError:
                continue
            if buckets.nunique() < quantiles:
                continue
            means = group.groupby(buckets)["fwd_ret"].mean()
            curve_dates.append(str(pd.Timestamp(date).date()))
            for q in range(quantiles):
                layer_returns[q].append(float(means.get(q, 0.0)))
            ls_returns.append(float(means.get(quantiles - 1, 0.0) - means.get(0, 0.0)))

            # ---- rank stability (turnover proxy)
            ranks = group.set_index("symbol")[name].rank()
            if prev_ranks is not None:
                joined = pd.concat([prev_ranks, ranks], axis=1, join="inner")
                if len(joined) >= 3:
                    ac = joined.iloc[:, 0].corr(joined.iloc[:, 1])
                    if pd.notna(ac):
                        autocorrs.append(float(ac))
            prev_ranks = ranks

        stats = FactorStats(name=name, n_periods=len(ic_values))
        if ic_values:
            s = pd.Series(ic_values)
            stats.ic_mean = round(float(s.mean()), 4)
            stats.ic_std = round(float(s.std(ddof=1)), 4) if len(s) > 1 else None
            if stats.ic_std:
                stats.icir = round(stats.ic_mean / stats.ic_std, 4)
                stats.ic_t_stat = round(stats.icir * math.sqrt(len(s)), 4)
            stats.ic_positive_rate = round(float((s > 0).mean()), 4)
        if autocorrs:
            stats.rank_autocorr = round(sum(autocorrs) / len(autocorrs), 4)

        # cumulative layer curves (compounded, starting at 1.0)
        curves: Dict[str, List[float]] = {}
        for q in range(quantiles):
            acc, series = 1.0, []
            for r in layer_returns[q]:
                acc *= 1.0 + r
                series.append(round(acc, 6))
            curves[f"Q{q + 1}"] = series
        ls_curve, acc = [], 1.0
        for r in ls_returns:
            acc *= 1.0 + r
            ls_curve.append(round(acc, 6))
        if ls_curve:
            stats.long_short_total_return = round(ls_curve[-1] - 1.0, 4)
            stats.top_layer_total_return = round(
                curves[f"Q{quantiles}"][-1] - 1.0, 4
            )
            stats.bottom_layer_total_return = round(curves["Q1"][-1] - 1.0, 4)

        payload = {
            "dates": curve_dates,
            "layers": curves,
            "long_short": ls_curve,
        }
        if constrained:
            sim = constrained_long_short(
                sub, name, quantiles,
                max_position=max_position, max_trade=max_trade,
                adv=adv, book=book,
            )
            stats.constrained_ls_return = sim["total_return"]
            stats.mean_turnover = sim["mean_turnover"]
            payload["long_short_constrained"] = sim["curve"]

        report.stats.append(stats)
        report.ic_series[name] = {"dates": ic_dates, "ic": ic_values}
        report.layer_curves[name] = payload

    return report
