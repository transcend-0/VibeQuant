"""Tests for round 5: crypto kind, markets registry, ADV caps, auto-optimize."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_sources.market import (  # noqa: E402
    MarketDataError,
    canonical,
    normalize_symbol,
)
from src.dsl import TaskSpec  # noqa: E402
from src.research.auto_optimize import _objective_name, _pin_data  # noqa: E402


def test_crypto_normalization():
    assert normalize_symbol("BTC", "crypto") == ("BTC-USDT", "CRYPTO")
    assert normalize_symbol("eth/usdt", "crypto") == ("ETH-USDT", "CRYPTO")
    assert normalize_symbol("SOLUSDT", "crypto") == ("SOL-USDT", "CRYPTO")
    assert canonical("btc", "crypto") == "BTC-USDT"
    with pytest.raises(MarketDataError):
        normalize_symbol("!!", "crypto")


def test_crypto_source_validates():
    spec = TaskSpec.from_dict(
        {"kind": "strategy", "data": {"source": "crypto", "symbols": ["BTC"]}}
    )
    assert spec.data.source == "crypto"


def test_markets_registry_shape():
    from webui.server import MARKETS

    keys = [m["key"] for m in MARKETS]
    assert keys == ["cn_etf", "cn_stock", "hk", "us", "crypto"]
    cn_stock = next(m for m in MARKETS if m["key"] == "cn_stock")
    assert any(u["key"] == "hs300" and u.get("dynamic")
               for u in cn_stock["universes"])
    for market in MARKETS:
        assert market["source"] in ("etf", "stock", "index", "hk", "us", "crypto")
        uni_keys = [u["key"] for u in market["universes"]]
        assert "custom" in uni_keys
    cn_etf = MARKETS[0]
    assert any(u.get("dynamic") for u in cn_etf["universes"])  # all-ETF list
    assert any(u["key"] == "pool24" for u in cn_etf["universes"])


def test_auto_optimize_pins_data_section():
    ref = TaskSpec.from_dict(
        {"kind": "factor",
         "data": {"source": "synthetic", "symbols": ["A", "B", "C"]},
         "factor": {"expressions": ["Delta(Close, 1)"]}}
    )
    # candidate tries to switch symbols -> data section forced back
    candidate = ref.to_yaml().replace("- A", "- Z")
    pinned = _pin_data(candidate, ref)
    assert pinned is not None and "- A" in pinned and "- Z" not in pinned
    # candidate switching kind is rejected outright
    bad = ref.to_yaml().replace("kind: factor", "kind: strategy")
    assert _pin_data(bad, ref) is None
    assert _objective_name("factor") == "|ICIR|"
    assert _objective_name("strategy") == "Sharpe"
