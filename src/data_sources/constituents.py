"""Point-in-time index constituent pools (沪深300 etc.).

Ported from the user's Strategy/backtest.py `Date2Symbols`: a dict of
{yyyymmdd int -> set(symbols)} where lookup with ANY date returns the
constituents as of the most recent reconstitution on or before it —
so backtests always see the membership that was actually in force.

Snapshot data lives in data/constituents/<pool>/<pool>_list_*.csv with
baostock's columns (updateDate, code, code_name; codes like "sh.600000").
Refreshing a pool = dropping in a new snapshot CSV from the same
baostock workflow (Strategy/data/download_stock_list.py); the loader
picks it up automatically.
"""

from __future__ import annotations

import bisect
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from ..config import data_dir


class Date2Symbols(dict):
    """{yyyymmdd -> set(symbols)} with as-of (previous reconstitution) lookup."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._sorted_keys: Optional[List[int]] = None
        if args or kwargs:
            self.update(*args, **kwargs)

    @staticmethod
    def _norm_date(value) -> int:
        if isinstance(value, int) and 19000101 <= value <= 21000101:
            return value
        if isinstance(value, datetime.datetime):
            value = value.date()
        if isinstance(value, datetime.date):
            return value.year * 10000 + value.month * 100 + value.day
        ts = pd.Timestamp(value)
        if getattr(ts, "tz", None) is not None:
            ts = ts.tz_convert(None)
        d = ts.normalize().date()
        return d.year * 10000 + d.month * 100 + d.day

    def _keys_sorted(self) -> List[int]:
        if self._sorted_keys is None:
            self._sorted_keys = sorted(dict.keys(self))
        return self._sorted_keys

    def effective_date(self, query_date) -> Optional[int]:
        q = self._norm_date(query_date)
        keys = self._keys_sorted()
        if not keys:
            raise KeyError("Date2Symbols is empty")
        i = bisect.bisect_right(keys, q) - 1
        return None if i < 0 else keys[i]

    def as_of(self, query_date) -> Set[str]:
        k = self.effective_date(query_date)
        if k is None:
            return set()  # before the first snapshot: unknown membership
        return dict.__getitem__(self, k)

    def __missing__(self, key):
        return self.as_of(key)

    def __setitem__(self, key, value) -> None:
        dict.__setitem__(self, self._norm_date(key), set(value))
        self._sorted_keys = None

    def update(self, *args, **kwargs) -> None:
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def union(self) -> Set[str]:
        out: Set[str] = set()
        for v in dict.values(self):
            out |= v
        return out


def _norm_code(raw: str) -> str:
    """'sh.600000' / 'sz.000001' / '600000' -> bare 6-digit code."""
    return str(raw).strip().split(".")[-1]


def constituents_dir() -> Path:
    path = data_dir() / "constituents"
    path.mkdir(parents=True, exist_ok=True)
    return path


def available_pools() -> List[str]:
    """Pools with at least one snapshot CSV (e.g. ['hs300'])."""
    return sorted(
        p.name for p in constituents_dir().iterdir()
        if p.is_dir() and any(p.glob("*.csv"))
    )


def load_date2symbols(pool: str) -> Date2Symbols:
    """Build the as-of membership map from a pool's snapshot CSVs."""
    pool_dir = constituents_dir() / pool
    paths = sorted(pool_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"no constituent snapshots under {pool_dir} — add "
            f"{pool}_list_*.csv files (columns: updateDate, code, code_name)"
        )
    merged: Dict[int, Set[str]] = {}
    for path in paths:
        df = pd.read_csv(path)
        dates = pd.to_datetime(df["updateDate"], errors="coerce").dt.normalize()
        codes = df["code"].astype(str).map(_norm_code)
        for d, group in codes.groupby(dates):
            if pd.isna(d):
                continue
            key = d.year * 10000 + d.month * 100 + d.day
            merged.setdefault(key, set()).update(group.tolist())
    return Date2Symbols(merged)


def pool_symbols(
    pool: str, start: Optional[str] = None, end: Optional[str] = None
) -> List[str]:
    """Symbols for a pool: union of memberships effective in [start, end].

    With no dates: the latest snapshot's membership (the current pool).
    """
    d2s = load_date2symbols(pool)
    if start is None and end is None:
        latest = max(dict.keys(d2s))
        return sorted(dict.__getitem__(d2s, latest))
    keys = sorted(dict.keys(d2s))
    start_key = d2s.effective_date(start or "1900-01-01")
    end_key = d2s._norm_date(end or "2100-01-01")
    out: Set[str] = set()
    for k in keys:
        if (start_key is None or k >= start_key) and k <= end_key:
            out |= dict.__getitem__(d2s, k)
    if start_key is not None:
        out |= dict.__getitem__(d2s, start_key)
    return sorted(out)


def membership_mask(
    pool: str, panel: pd.DataFrame
) -> pd.Series:
    """Boolean mask: was panel.symbol a member of the pool on panel.date?

    Symbols are compared on their bare 6-digit code (canonical forms like
    '600000.SH' match snapshot codes like 'sh.600000').
    """
    d2s = load_date2symbols(pool)
    dates = pd.to_datetime(panel["date"])
    date_keys = dates.dt.year * 10000 + dates.dt.month * 100 + dates.dt.day
    codes = panel["symbol"].astype(str).str.extract(r"(\d{6})")[0]

    members_cache: Dict[int, Set[str]] = {}

    def _is_member(key: int, code) -> bool:
        if pd.isna(code):
            return False
        if key not in members_cache:
            members_cache[key] = d2s.as_of(key)
        return code in members_cache[key]

    return pd.Series(
        [_is_member(k, c) for k, c in zip(date_keys, codes)],
        index=panel.index,
    )
