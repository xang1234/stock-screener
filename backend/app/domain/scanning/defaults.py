"""Shared default scan profile used by backend scheduling and the frontend."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


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


def get_default_scan_profile() -> dict[str, Any]:
    """Return a mutable copy of the backend-owned default scan profile."""
    return {
        "universe": DEFAULT_SCAN_UNIVERSE,
        "screeners": list(DEFAULT_SCAN_SCREENERS),
        "composite_method": DEFAULT_SCAN_COMPOSITE_METHOD,
        "criteria": deepcopy(DEFAULT_SCAN_CRITERIA),
    }
