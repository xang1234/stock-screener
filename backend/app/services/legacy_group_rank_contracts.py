"""Shared contracts for the legacy Group ranking implementation."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


CACHE_MISS_TOLERANCE_RATIO = 0.05


class IncompleteGroupRankingCacheError(RuntimeError):
    """Raised when a cache-only same-day Group ranking run lacks inputs."""

    def __init__(self, stats: Dict[str, Any]):
        self.stats = stats
        benchmark_cached = stats.get("benchmark_cached", stats.get("spy_cached"))
        benchmark_symbol = str(stats.get("benchmark_symbol") or "SPY")
        market = str(stats.get("market") or "").strip().upper()
        market_suffix = f" for {market}" if market else ""
        reason = f"{benchmark_symbol} benchmark data is missing from cache{market_suffix}"
        if benchmark_cached:
            reason = (
                f"{stats.get('cache_miss_symbols', 0)} symbols are missing cached price data"
            )
        super().__init__(reason)


@dataclass(frozen=True)
class GroupRankPrefetchData:
    """Cached inputs shared by daily, backfill, and gap-fill Group rank runs."""

    benchmark_prices: Optional[pd.DataFrame]
    prices_by_symbol: Dict[str, Optional[pd.DataFrame]]
    active_symbols: set[str]
    market_caps: Dict[str, float]
    stats: Dict[str, Any]
    symbols_by_group: Dict[str, List[str]]


__all__ = [
    "CACHE_MISS_TOLERANCE_RATIO",
    "GroupRankPrefetchData",
    "IncompleteGroupRankingCacheError",
]
