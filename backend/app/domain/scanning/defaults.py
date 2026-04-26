"""Shared default scan profile used by backend scheduling and the frontend."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_SCAN_UNIVERSE = "all"
DEFAULT_SCAN_BENCHMARKS_BY_MARKET = {
    "US": "SPY",
    "HK": "^HSI",
    "IN": "^NSEI",
    "JP": "^N225",
    "TW": "^TWII",
}
DEFAULT_SCAN_SCREENERS = [
    "minervini",
    "canslim",
    "ipo",
    "custom",
    "volume_breakthrough",
    "setup_engine",
]
DEFAULT_SCAN_COMPOSITE_METHOD = "weighted_average"
DEFAULT_SCAN_CUSTOM_FILTERS = {
    "price_min": 20,
    "price_max": 500,
    "rs_rating_min": 75,
    "volume_min": 1_000_000,
    "market_cap_min": 1_000_000_000,
    "eps_growth_min": 20,
    "sales_growth_min": 15,
    "ma_alignment": True,
    "min_score": 70,
}
DEFAULT_SCAN_CRITERIA = {
    "include_vcp": True,
    "custom_filters": DEFAULT_SCAN_CUSTOM_FILTERS,
}


def get_default_scan_profile(market: str | None = None) -> dict[str, Any]:
    """Return a mutable copy of the backend-owned default scan profile.

    When ``market`` is omitted the legacy global profile is preserved. Market
    callers get a market-scoped universe plus the canonical benchmark used by
    relative-strength calculations.
    """
    profile: dict[str, Any] = {
        "universe": DEFAULT_SCAN_UNIVERSE,
        "screeners": list(DEFAULT_SCAN_SCREENERS),
        "composite_method": DEFAULT_SCAN_COMPOSITE_METHOD,
        "criteria": deepcopy(DEFAULT_SCAN_CRITERIA),
    }
    if market is None:
        return profile

    normalized_market = str(market or "US").strip().upper()
    try:
        benchmark_symbol = DEFAULT_SCAN_BENCHMARKS_BY_MARKET[normalized_market]
    except KeyError as exc:
        supported = ", ".join(sorted(DEFAULT_SCAN_BENCHMARKS_BY_MARKET))
        raise ValueError(f"Unsupported market for default scan profile: {market}. Supported: {supported}") from exc
    profile["universe"] = f"market:{normalized_market}"
    profile["benchmark_symbol"] = benchmark_symbol
    profile["criteria"]["benchmark_symbol"] = benchmark_symbol
    return profile
