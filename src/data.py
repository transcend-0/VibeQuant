"""Data loading: synthetic (offline, deterministic), CSV, optional akshare.

Returns {symbol: DataFrame} with columns date/open/high/low/close/volume
— the shape akquant.run_backtest accepts directly.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .dsl import DataSpec

REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}


class DataError(RuntimeError):
    pass


def load(spec: DataSpec) -> Dict[str, pd.DataFrame]:
    if spec.source == "synthetic":
        return {
            sym: synthetic_bars(sym, spec.start, spec.end, spec.seed)
            for sym in spec.symbols
        }
    if spec.source == "csv":
        return _load_csv(spec)
    if spec.source in MARKET_KINDS:
        return _load_market(spec)
    if spec.source == "akshare":
        return _load_akshare(spec)
    raise DataError(f"unknown data source {spec.source!r}")


# ------------------------------------------------- market (etf/stock/index/hk/us)
MARKET_KINDS = ("etf", "stock", "index", "hk", "us", "crypto")


def _load_market(spec: DataSpec) -> Dict[str, pd.DataFrame]:
    """Real daily bars via the eastmoney->tencent fallback chain, cached."""
    from .config import raw_data_dir
    from .data_sources import market

    kind = spec.source
    cache_dir = raw_data_dir(kind)
    start = spec.start or "2020-01-01"
    end = spec.end or _dt_today()
    frames: Dict[str, pd.DataFrame] = {}
    for raw in spec.symbols:
        symbol = market.canonical(raw, kind)
        df = market.fetch_daily(raw, kind, start, end, cache_dir=cache_dir)
        if len(df) < 30:
            raise DataError(
                f"{kind} {symbol}: fewer than 30 bars in {start}..{end}"
            )
        df = df.copy()
        df["symbol"] = symbol
        frames[symbol] = df
    return frames


def _dt_today() -> str:
    import datetime

    return datetime.date.today().isoformat()


# ------------------------------------------------------------- synthetic
def synthetic_bars(
    symbol: str,
    start: Optional[str],
    end: Optional[str],
    seed: int = 42,
    initial_price: float = 100.0,
) -> pd.DataFrame:
    """Deterministic geometric-random-walk daily bars (weekdays only).

    Seeded per (symbol, seed) so the same task always sees the same data —
    reproducibility is part of safe-by-default.
    """
    start = start or "2022-01-01"
    end = end or "2024-12-31"
    dates = pd.bdate_range(start=start, end=end)
    if len(dates) < 30:
        raise DataError(f"date range {start}..{end} too short (<30 trading days)")

    digest = hashlib.sha256(f"{symbol}:{seed}".encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))

    price = initial_price
    rows = []
    trend = 0.0
    for date in dates:
        # regime-switching drift keeps the series interesting for signals
        if rng.random() < 0.02:
            trend = rng.uniform(-0.002, 0.003)
        ret = rng.gauss(trend, 0.018)
        open_ = price
        close = max(price * (1.0 + ret), 0.5)
        high = max(open_, close) * (1.0 + abs(rng.gauss(0, 0.004)))
        low = min(open_, close) * (1.0 - abs(rng.gauss(0, 0.004)))
        volume = int(rng.uniform(1e6, 5e6))
        rows.append((date, open_, high, low, close, volume))
        price = close

    df = pd.DataFrame(
        rows, columns=["date", "open", "high", "low", "close", "volume"]
    )
    df["symbol"] = symbol
    return df


# ------------------------------------------------------------------ csv
def _load_csv(spec: DataSpec) -> Dict[str, pd.DataFrame]:
    path = Path(spec.path)  # type: ignore[arg-type]
    if not path.exists():
        raise DataError(f"csv path not found: {path}")

    frames: Dict[str, pd.DataFrame] = {}
    if path.is_dir():
        for sym in spec.symbols:
            file = path / f"{sym}.csv"
            if not file.exists():
                raise DataError(f"missing csv for symbol {sym}: {file}")
            frames[sym] = _read_one_csv(file, sym, spec)
    else:
        if len(spec.symbols) != 1:
            raise DataError(
                "single csv file given but multiple symbols requested; "
                "use a directory with one <symbol>.csv per symbol"
            )
        frames[spec.symbols[0]] = _read_one_csv(path, spec.symbols[0], spec)
    return frames


def _read_one_csv(file: Path, symbol: str, spec: DataSpec) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataError(f"{file}: missing columns {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if spec.start:
        df = df[df["date"] >= pd.Timestamp(spec.start)]
    if spec.end:
        df = df[df["date"] <= pd.Timestamp(spec.end)]
    if len(df) < 30:
        raise DataError(f"{file}: fewer than 30 bars after date filtering")
    df["symbol"] = symbol
    return df.reset_index(drop=True)


# -------------------------------------------------------------- akshare
def _load_akshare(spec: DataSpec) -> Dict[str, pd.DataFrame]:
    try:
        import akshare as ak  # noqa: WPS433
    except ImportError as exc:
        raise DataError(
            "data.source=akshare requires `pip install akshare` "
            "(or switch to source: synthetic / csv)"
        ) from exc

    frames: Dict[str, pd.DataFrame] = {}
    start = (spec.start or "2022-01-01").replace("-", "")
    end = (spec.end or "2026-12-31").replace("-", "")
    for sym in spec.symbols:
        df = ak.stock_zh_a_daily(
            symbol=sym, start_date=start, end_date=end, adjust=spec.adjust
        )
        if df is None or df.empty:
            raise DataError(f"akshare returned no data for {sym}")
        df = df.rename(columns=str.lower)
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = sym
        frames[sym] = df[
            ["date", "open", "high", "low", "close", "volume", "symbol"]
        ].reset_index(drop=True)
    return frames
