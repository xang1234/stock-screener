"""Shared default scan profile used by backend scheduling and the frontend."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.domain.common.benchmarks import get_primary_benchmark_symbol, supported_benchmark_markets

DEFAULT_SCAN_UNIVERSE = "all"
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
        benchmark_symbol = get_primary_benchmark_symbol(normalized_market)
    except ValueError as exc:
        supported = ", ".join(supported_benchmark_markets())
        raise ValueError(f"Unsupported market for default scan profile: {market}. Supported: {supported}") from exc
    profile["universe"] = f"market:{normalized_market}"
    profile["benchmark_symbol"] = benchmark_symbol
    profile["criteria"]["benchmark_symbol"] = benchmark_symbol
    return profile
