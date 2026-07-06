"""Thin adapter over akquant's factor expression engine.

Like akquant_engine.py, this lives in adapters/ because it imports akquant
(and polars, akquant's factor backend). It evaluates Alpha101-style
expressions on a {symbol: OHLCV DataFrame} panel and returns plain pandas —
factor *analysis* (IC, layering) stays engine-agnostic in src.factors.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import pandas as pd
import polars as pl

from akquant.factor import FactorEngine


class FactorExprError(ValueError):
    pass


_NAME_RE = re.compile(r"^\s*([A-Za-z_][\w.-]*)\s*=\s*(.+)$")


def split_named_expression(raw: str, index: int) -> Tuple[str, str]:
    """'Mom20 = -Delta(Close,20)' -> ('Mom20', '-Delta(Close,20)')."""
    match = _NAME_RE.match(raw)
    if match:
        return match.group(1), match.group(2).strip()
    return f"factor_{index + 1}", raw.strip()


def compute_factors(
    expressions: List[str], frames: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Evaluate expressions -> wide panel [date, symbol, <name>...]."""
    panel = pd.concat(frames.values(), ignore_index=True)
    panel = panel[["date", "symbol", "open", "high", "low", "close", "volume"]]
    lf = pl.from_pandas(panel).lazy()
    engine = FactorEngine(catalog=None)

    merged: pd.DataFrame | None = None
    for i, raw in enumerate(expressions):
        name, expr = split_named_expression(raw, i)
        try:
            out = engine.run_on_data(lf, expr).to_pandas()
        except Exception as exc:
            raise FactorExprError(f"expression {name!r} failed: {exc}") from exc
        out = out.rename(columns={"factor_value": name})
        merged = out if merged is None else merged.merge(
            out, on=["date", "symbol"], how="outer"
        )

    assert merged is not None
    merged["date"] = pd.to_datetime(merged["date"])
    return merged.sort_values(["date", "symbol"]).reset_index(drop=True)
