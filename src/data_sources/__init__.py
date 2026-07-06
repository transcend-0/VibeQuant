"""External market-data clients.

Design borrowed from Vibe-Trading's loader stack (agent/backtest/loaders):
ordered fallback chains over free no-auth HTTP endpoints, per-host request
throttling, a normalized OHLCV contract, and an opt-out local cache.
Re-implemented minimally here (urllib only) so VibeQuant adds no heavy
data dependencies.
"""

from .etf import fetch_etf_daily, normalize_etf_symbol  # noqa: F401
