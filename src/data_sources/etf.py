"""A-share ETF daily bars — thin wrapper over the generalized market client.

Kept for API stability (tests, data.py); all logic lives in market.py,
which extends the same fallback-chain design to stocks, indices, HK and US.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from . import market
from .market import MarketDataError as ETFDataError  # noqa: F401  (back-compat)


def normalize_etf_symbol(raw: str) -> Tuple[str, str]:
    return market.normalize_symbol(raw, "etf")


def canonical(raw: str) -> str:
    return market.canonical(raw, "etf")


def fetch_etf_daily(
    raw_symbol: str,
    start: str,
    end: str,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    return market.fetch_daily(
        raw_symbol, "etf", start, end, cache_dir=cache_dir, use_cache=use_cache
    )
