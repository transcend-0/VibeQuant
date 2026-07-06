"""Tests for round 6: PIT constituent pools and universe recording."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_sources.constituents import (  # noqa: E402
    Date2Symbols,
    available_pools,
    load_date2symbols,
    membership_mask,
    pool_symbols,
)
from src.dsl import TaskSpec  # noqa: E402


def test_date2symbols_as_of_lookup():
    d2s = Date2Symbols({20200101: {"A", "B"}, 20210101: {"B", "C"}})
    assert d2s["2020-06-15"] == {"A", "B"}      # previous reconstitution
    assert d2s["2021-01-01"] == {"B", "C"}      # exact date
    assert d2s["2025-12-31"] == {"B", "C"}      # after last
    assert d2s["1999-01-01"] == set()           # before first: unknown
    assert d2s.union() == {"A", "B", "C"}


def test_hs300_snapshots_load():
    assert "hs300" in available_pools()
    d2s = load_date2symbols("hs300")
    assert len(d2s.as_of("2021-03-15")) == 300
    current = pool_symbols("hs300")
    assert len(current) == 300 and "600519" in current
    union = pool_symbols("hs300", "2021-01-01", "2024-12-31")
    assert len(union) > 300  # reconstitutions accumulate


def test_membership_mask_point_in_time():
    d2s = load_date2symbols("hs300")
    member_now = sorted(d2s.as_of("2024-06-01"))[0]
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-06-03", "2024-06-03"]),
            "symbol": [f"{member_now}.SH", "999999.SZ"],
            "f": [1.0, 2.0],
        }
    )
    mask = membership_mask("hs300", panel)
    assert bool(mask.iloc[0]) is True
    assert bool(mask.iloc[1]) is False


def test_data_universe_field_roundtrip():
    spec = TaskSpec.from_dict(
        {
            "kind": "strategy",
            "data": {"source": "stock", "universe": "hs300",
                     "symbols": ["600519", "600036"]},
        }
    )
    assert spec.data.universe == "hs300"
    clone = TaskSpec.from_yaml(spec.to_yaml())
    assert clone.data.universe == "hs300"
