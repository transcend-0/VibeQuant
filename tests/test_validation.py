"""Validation v1: the validator itself must tell signal from noise."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.factors.validation import (  # noqa: E402
    permutation_test,
    subperiod_consistency_ic,
    validate_factor,
    validate_strategy,
)


def _panel(signal: bool, n_days=250, n_syms=20, seed=7):
    """Factor panel + forward returns; optionally plant a real signal."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    rows_f, rows_r = [], []
    for d in dates:
        fwd = rng.normal(0, 0.02, n_syms)
        noise = rng.normal(0, 1, n_syms)
        factor = 0.8 * (fwd / 0.02) + 0.6 * noise if signal else noise
        for i in range(n_syms):
            rows_f.append((d, f"S{i}", factor[i]))
            rows_r.append((d, f"S{i}", fwd[i]))
    fp = pd.DataFrame(rows_f, columns=["date", "symbol", "f"])
    fr = pd.DataFrame(rows_r, columns=["date", "symbol", "fwd_ret"])
    return fp, fr


def _ic_series(fp, fr):
    m = fp.merge(fr, on=["date", "symbol"])
    out = m.groupby("date").apply(
        lambda g: g["f"].rank().corr(g["fwd_ret"].rank()), include_groups=False
    )
    return [str(d.date()) for d in out.index], [float(v) for v in out]


def test_planted_signal_passes():
    fp, fr = _panel(signal=True)
    dates, ics = _ic_series(fp, fr)
    v = validate_factor(fp, fr, "f", dates, ics, trial_count=1)
    assert v["permutation"]["p_raw"] < 0.05
    assert v["significant_deflated"]
    assert v["overfit_risk"] == "low"


def test_pure_noise_fails():
    fp, fr = _panel(signal=False)
    dates, ics = _ic_series(fp, fr)
    v = validate_factor(fp, fr, "f", dates, ics, trial_count=1)
    assert v["overfit_risk"] != "low"


def test_trial_deflation_bites():
    # a mildly significant factor survives 1 trial but not 50
    fp, fr = _panel(signal=True)
    dates, ics = _ic_series(fp, fr)
    v1 = validate_factor(fp, fr, "f", dates, ics, trial_count=1)
    v50 = validate_factor(fp, fr, "f", dates, ics, trial_count=50)
    assert v50["alpha_deflated"] < v1["alpha_deflated"]
    assert v50["trial_count"] == 50
    # deflated significance can only get harder, never easier
    assert (not v50["significant_deflated"]) or v1["significant_deflated"]


def test_subperiod_consistency():
    dates = [str(d.date()) for d in pd.bdate_range("2021-01-04", periods=500)]
    steady = subperiod_consistency_ic(dates, [0.05] * 500)
    assert steady["sign_consistency"] == 1.0
    flip = subperiod_consistency_ic(dates, [0.1] * 250 + [-0.1] * 250)
    assert flip["sign_consistency"] < 1.0


def test_strategy_windows():
    idx = pd.bdate_range("2022-01-03", periods=400)
    rising = pd.Series(np.linspace(1e6, 1.5e6, 400), index=idx)
    v = validate_strategy(rising, num_trades=30)
    assert v["positive_fraction"] == 1.0 and v["overfit_risk"] == "low"

    rng = np.random.default_rng(3)
    flat = pd.Series(1e6 * np.cumprod(1 + rng.normal(0, 0.01, 400)), index=idx)
    v2 = validate_strategy(flat, num_trades=30)
    assert v2["overfit_risk"] in ("medium", "high")

    v3 = validate_strategy(rising.iloc[:30], num_trades=5)
    assert v3["overfit_risk"] == "high"  # too short to judge
