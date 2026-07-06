"""Tests for round 4: multi-kind data, ETF pool, factor ops v2, refine."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_sources.etf_pool import (  # noqa: E402
    DEFAULT_ETF_POOL,
    POOL_SYMBOLS,
    industry_groups,
    pool_categories,
)
from src.data_sources.market import (  # noqa: E402
    MarketDataError,
    canonical,
    normalize_symbol,
)
from src.dsl import DSLError, TaskSpec  # noqa: E402
from src.factors.analysis import (  # noqa: E402
    apply_factor_ops,
    constrained_long_short,
)
from src.research.refine import _heuristic_refine, _validate_yaml  # noqa: E402


# --------------------------------------------------- market normalization
def test_normalize_all_kinds():
    assert normalize_symbol("510300", "etf") == ("510300", "SH")
    assert normalize_symbol("600000", "stock") == ("600000", "SH")
    assert normalize_symbol("000001", "stock") == ("000001", "SZ")
    assert normalize_symbol("300750", "stock") == ("300750", "SZ")
    assert normalize_symbol("000300", "index") == ("000300", "SH")
    assert normalize_symbol("399006", "index") == ("399006", "SZ")
    assert normalize_symbol("00700", "hk") == ("00700", "HK")
    assert normalize_symbol("700", "hk") == ("00700", "HK")
    assert normalize_symbol("HSI", "hk") == ("HSI", "HK")
    assert normalize_symbol("AAPL", "us") == ("AAPL", "")
    assert normalize_symbol("AAPL.OQ", "us") == ("AAPL", "OQ")
    assert normalize_symbol(".NDX", "us") == ("NDX", "IDX")
    with pytest.raises(MarketDataError):
        normalize_symbol("!!!", "us")
    with pytest.raises(MarketDataError):
        normalize_symbol("600000", "nope")


def test_canonical_forms():
    assert canonical("600000", "stock") == "600000.SH"
    assert canonical("000300", "index") == "sh000300"
    assert canonical("700", "hk") == "hk00700"
    assert canonical(".INX", "us") == ".INX"
    assert canonical("AAPL", "us") == "AAPL"


# ------------------------------------------------------------- ETF pool
def test_pool_shape():
    assert len(POOL_SYMBOLS) == 24
    cats = pool_categories()
    assert set(cats) == {"abroad", "commodity", "bond", "index", "industry"}
    assert sum(len(v) for v in cats.values()) == 24
    assert DEFAULT_ETF_POOL["518880"][0] == "commodity"


def test_industry_groups_mapping():
    groups = industry_groups(["510300", "sh518880", "512800", "999999"])
    assert groups["510300"] == "index"
    assert groups["sh518880"] == "commodity"
    assert groups["512800"] == "industry"
    assert groups["999999"] == "other"


# --------------------------------------------------------- factor ops v2
def test_industry_neutralization_demeans_within_groups():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 4),
            "symbol": ["A", "B", "C", "D"],
            "f": [1.0, 3.0, 10.0, 14.0],
        }
    )
    groups = {"A": "g1", "B": "g1", "C": "g2", "D": "g2"}
    out = apply_factor_ops(panel, ["f"], neutralization="industry", groups=groups)
    vals = dict(zip(out["symbol"], out["f"]))
    assert vals["A"] == -1.0 and vals["B"] == 1.0  # demeaned within g1
    assert vals["C"] == -2.0 and vals["D"] == 2.0  # demeaned within g2


def test_constrained_long_short_caps():
    dates = pd.to_datetime(["2024-01-01"] * 4 + ["2024-01-02"] * 4)
    sub = pd.DataFrame(
        {
            "date": dates,
            "symbol": ["A", "B", "C", "D"] * 2,
            "f": [4, 3, 2, 1, 1, 2, 3, 4],
            "fwd_ret": [0.04, 0.03, 0.02, 0.01] * 2,
        }
    )
    # ADV20 == book for every name/date, so a fraction f caps weight at f
    adv = pd.DataFrame(
        {"date": dates, "symbol": ["A", "B", "C", "D"] * 2,
         "adv20": [1000.0] * 8}
    )
    out = constrained_long_short(
        sub, "f", quantiles=4, max_position=0.5, adv=adv, book=1000.0
    )
    assert out["total_return"] is not None
    # day1 targets: A +1 (top), D -1 (bottom) -> capped to ±0.5 of book
    # period 1 return = 0.5*0.04 - 0.5*0.01 = 0.015
    assert abs(out["curve"][0] - 1.015) < 1e-9
    assert out["constraint_binds"] > 0

    out2 = constrained_long_short(
        sub, "f", quantiles=4, max_trade=0.25, adv=adv, book=1000.0
    )
    # day1: weights move 0 -> ±0.25 only; return = 0.25*0.04 - 0.25*0.01
    assert abs(out2["curve"][0] - (1 + 0.25 * 0.03)) < 1e-9
    assert out2["mean_turnover"] is not None

    # no ADV data -> caps are inactive (uncapped targets ±1)
    out3 = constrained_long_short(sub, "f", quantiles=4, max_position=0.5)
    assert abs(out3["curve"][0] - (1 + 1.0 * 0.03)) < 1e-9


def test_factor_dsl_new_fields():
    spec = TaskSpec.from_dict(
        {
            "kind": "factor",
            "data": {"symbols": ["A", "B"]},
            "factor": {
                "expressions": ["Delta(Close, 1)"],
                "neutralization": "industry",
                "max_position": 0.3,
                "max_trade": 0.1,
            },
        }
    )
    assert spec.factor.forward_days == 1  # new default
    with pytest.raises(DSLError):
        TaskSpec.from_dict(
            {
                "kind": "factor",
                "data": {"symbols": ["A", "B"]},
                "factor": {"expressions": ["x"], "max_position": 1.5},
            }
        )


def test_multi_kind_sources_validate():
    for source, symbols in [
        ("stock", ["600000"]), ("index", ["000300"]),
        ("hk", ["00700"]), ("us", ["AAPL"]),
    ]:
        spec = TaskSpec.from_dict(
            {"kind": "strategy", "data": {"source": source, "symbols": symbols}}
        )
        assert spec.data.source == source


# ---------------------------------------------------------------- refine
def test_refine_heuristic_factor_progression():
    spec = TaskSpec.from_dict(
        {
            "kind": "factor",
            "data": {"symbols": ["A", "B", "C"]},
            "factor": {"expressions": ["Delta(Close, 1)"]},
        }
    )
    out = _heuristic_refine(spec, "improve it", "en")
    assert out["engine"] == "rules"
    revised = TaskSpec.from_yaml(out["yaml"])
    assert revised.factor.neutralization == "rank"  # first suggestion


def test_refine_validator_rejects_bad_yaml():
    assert _validate_yaml("kind: strategy\nexecution: {mode: live}") is None
    assert _validate_yaml("not: [valid") is None
    good = "kind: strategy\ndata: {symbols: [DEMO]}\n"
    assert _validate_yaml(good) is not None
