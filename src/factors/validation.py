"""Validation v1: is this result distinguishable from luck?

Two checks per kind, both computed from the single run's artifacts
(no re-running the engine):

factor
  1. Cross-sectional permutation test with TRIAL-COUNT DEFLATION:
     shuffle factor values within each date N times, measure how often
     the shuffled |IC mean| beats the observed one (p_raw). Because a
     researcher who tried T factors on the same universe will pass any
     fixed threshold by luck, the acceptance threshold is deflated to
     0.05 / T (Bonferroni-style, in the spirit of deflated-Sharpe /
     Reality Check). T comes from the experiment log.
  2. Sub-period consistency: yearly windows of the IC series — the sign
     consistency across windows. This is a REGIME-ROBUSTNESS check, not
     out-of-sample: everyone (including the researcher) has seen all of
     this history.

strategy
  1. Sub-period consistency of the single equity curve: split into
     sequential windows, per-window return/Sharpe, fraction positive.

Verdicts combine into overfit_risk: low | medium | high. None of this
protects against researcher-level snooping beyond the trial deflation —
the only clean out-of-sample data is data that did not exist when the
task was frozen.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PERMUTATIONS = 200
STRATEGY_WINDOWS = 4


# ---------------------------------------------------------------- factor
def permutation_test(
    factor_panel: pd.DataFrame,
    fwd_returns: pd.DataFrame,  # date, symbol, fwd_ret
    name: str,
    n_permutations: int = PERMUTATIONS,
    seed: int = 42,
) -> Dict[str, Any]:
    """Per-date cross-sectional shuffle of factor values -> null |IC mean|."""
    merged = factor_panel[["date", "symbol", name]].merge(
        fwd_returns, on=["date", "symbol"], how="inner"
    ).dropna(subset=[name, "fwd_ret"])

    groups = []
    for _date, g in merged.groupby("date"):
        if len(g) >= 3:
            # ranks once per date; IC = Pearson on ranks (Spearman)
            fr = g[name].rank().to_numpy(dtype=float)
            rr = g["fwd_ret"].rank().to_numpy(dtype=float)
            fr = (fr - fr.mean()) / (fr.std() or 1.0)
            rr = (rr - rr.mean()) / (rr.std() or 1.0)
            groups.append((fr, rr))
    if len(groups) < 20:
        return {"error": f"only {len(groups)} usable dates (<20)"}

    observed = float(np.mean([float(np.mean(fr * rr)) for fr, rr in groups]))

    rng = np.random.default_rng(seed)
    beats = 0
    for _ in range(n_permutations):
        acc = 0.0
        for fr, rr in groups:
            acc += float(np.mean(rng.permutation(fr) * rr))
        if abs(acc / len(groups)) >= abs(observed):
            beats += 1
    p_raw = (beats + 1) / (n_permutations + 1)
    return {
        "observed_ic_mean": round(observed, 4),
        "p_raw": round(p_raw, 4),
        "n_dates": len(groups),
        "n_permutations": n_permutations,
    }


def subperiod_consistency_ic(
    ic_dates: List[str], ic_values: List[float]
) -> Dict[str, Any]:
    """Yearly IC-sign consistency from the run's own IC series."""
    if len(ic_values) < 20:
        return {"error": "IC series too short"}
    s = pd.Series(ic_values, index=pd.to_datetime(ic_dates))
    yearly = s.groupby(s.index.year).mean()
    if len(yearly) < 2:
        # under two years: split the series in half instead
        half = len(s) // 2
        yearly = pd.Series(
            [s.iloc[:half].mean(), s.iloc[half:].mean()], index=["H1", "H2"]
        )
    overall_sign = 1.0 if s.mean() >= 0 else -1.0
    agree = float((np.sign(yearly) == overall_sign).mean())
    return {
        "windows": {str(k): round(float(v), 4) for k, v in yearly.items()},
        "sign_consistency": round(agree, 3),
        "n_windows": len(yearly),
    }


def validate_factor(
    factor_panel: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    name: str,
    ic_dates: List[str],
    ic_values: List[float],
    trial_count: int = 1,
) -> Dict[str, Any]:
    perm = permutation_test(factor_panel, fwd_returns, name)
    consistency = subperiod_consistency_ic(ic_dates, ic_values)

    trial_count = max(int(trial_count), 1)
    alpha_deflated = 0.05 / trial_count
    p_raw = perm.get("p_raw")
    significant_raw = p_raw is not None and p_raw < 0.05
    significant_deflated = p_raw is not None and p_raw < alpha_deflated
    agree = consistency.get("sign_consistency")

    if significant_deflated and (agree is None or agree >= 0.7):
        risk = "low"
    elif significant_raw and (agree is None or agree >= 0.5):
        risk = "medium"
    else:
        risk = "high"

    return {
        "kind": "factor",
        "name": name,
        "permutation": perm,
        "trial_count": trial_count,
        "alpha_deflated": round(alpha_deflated, 5),
        "significant_raw": significant_raw,
        "significant_deflated": significant_deflated,
        "consistency": consistency,
        "overfit_risk": risk,
    }


# -------------------------------------------------------------- strategy
def validate_strategy(
    equity_curve: pd.Series,
    num_trades: int = 0,
    n_windows: int = STRATEGY_WINDOWS,
) -> Dict[str, Any]:
    """Sequential-window consistency of one equity curve (no re-runs)."""
    if equity_curve is None or len(equity_curve) < n_windows * 20:
        return {
            "kind": "strategy",
            "error": "equity curve too short for windowing",
            "overfit_risk": "high",
        }
    returns = equity_curve.pct_change().dropna()
    size = len(returns) // n_windows
    windows = []
    for i in range(n_windows):
        chunk = returns.iloc[i * size: (i + 1) * size if i < n_windows - 1 else None]
        total = float((1 + chunk).prod() - 1)
        vol = float(chunk.std())
        sharpe = float(chunk.mean() / vol * math.sqrt(252)) if vol > 0 else 0.0
        windows.append({"return": round(total, 4), "sharpe": round(sharpe, 2)})
    positive = sum(1 for w in windows if w["return"] > 0) / n_windows

    if positive >= 0.75 and num_trades >= 10:
        risk = "low"
    elif positive >= 0.5:
        risk = "medium"
    else:
        risk = "high"
    return {
        "kind": "strategy",
        "windows": windows,
        "positive_fraction": round(positive, 3),
        "num_trades": num_trades,
        "overfit_risk": risk,
    }


def count_prior_trials(
    store, spec, kind: str = "factor"
) -> int:
    """Prior experiments on the same universe, from the experiment log.

    Matched by data source + universe key; custom universes match on the
    exact symbol set. The current run is not yet logged, so +1 for it.
    """
    try:
        entries = store.list_runs(limit=1000)
    except Exception:
        return 1
    symbols_key = tuple(sorted(spec.data.symbols))
    count = 0
    for e in entries:
        if (e.get("kind") or "strategy") != kind:
            continue
        if e.get("data_source") != spec.data.source:
            continue
        universe = e.get("universe") or "custom"
        if spec.data.universe != "custom":
            if universe == spec.data.universe:
                count += 1
        elif tuple(sorted(e.get("symbols") or [])) == symbols_key:
            count += 1
    return count + 1  # include the current attempt
