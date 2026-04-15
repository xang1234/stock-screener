"""Per-market alert thresholds + owner routing (bead asia.10.2).

Hardcoded constants — version-controlled, easy to evolve. Every (metric, market)
pair has a 2-level severity ladder: warning is the first breach threshold;
critical is the page-someone threshold.

Operators tuning these should:
- Edit the constant in this file (PR'd, reviewed)
- Re-deploy to pick up the change (no DB migration needed)

If/when CRUD becomes a real need, lift to an ``alert_policies`` table without
changing the evaluator API — it consumes a Mapping[(metric, market), Levels].
"""

from __future__ import annotations

from typing import Dict, Optional, TypedDict

from ...tasks.market_queues import SHARED_SENTINEL, SUPPORTED_MARKETS
from .schema import MetricKey


class Levels(TypedDict, total=False):
    """Numeric thresholds per severity. Comparison direction is metric-defined."""
    warning: float
    critical: float


# Thresholds. Comparison semantics are encoded per-metric in the evaluator
# (e.g. freshness_lag is "bigger is worse"; extraction_success is "smaller is
# worse"). Markets not present here use the SHARED_SENTINEL fallback.
THRESHOLDS: Dict[str, Dict[str, Levels]] = {
    # Seconds since last successful price refresh. Asia markets get more slack
    # because their refresh windows are offset and can legitimately lag.
    MetricKey.FRESHNESS_LAG: {
        "US": {"warning": 7200, "critical": 21600},      # 2h / 6h
        "HK": {"warning": 10800, "critical": 28800},     # 3h / 8h
        "JP": {"warning": 10800, "critical": 28800},
        "TW": {"warning": 14400, "critical": 36000},     # 4h / 10h
        SHARED_SENTINEL: {"warning": 10800, "critical": 28800},
    },
    # Seconds since SPY/benchmark cache was warmed. Daily warm cadence.
    MetricKey.BENCHMARK_AGE: {
        m: {"warning": 86400, "critical": 172800}        # 1d / 2d
        for m in (*SUPPORTED_MARKETS, SHARED_SENTINEL)
    },
    # Fraction of universe in the 0-25 completeness bucket. Big bucket = bad.
    MetricKey.COMPLETENESS_DISTRIBUTION: {
        m: {"warning": 0.30, "critical": 0.50}
        for m in (*SUPPORTED_MARKETS, SHARED_SENTINEL)
    },
    # |delta|/prior_size — universe shrinking or growing too fast.
    MetricKey.UNIVERSE_DRIFT: {
        m: {"warning": 0.05, "critical": 0.15}
        for m in (*SUPPORTED_MARKETS, SHARED_SENTINEL)
    },
    # Per-language extraction success ratio. Smaller = worse.
    MetricKey.EXTRACTION_SUCCESS: {
        SHARED_SENTINEL: {"warning": 0.85, "critical": 0.70},
    },
}


# Market → owner team for alert routing. Stored on the alert row at trigger
# time so a later owner-map change doesn't rewrite history.
OWNERS: Dict[str, str] = {
    "US": "us-ops",
    "HK": "asia-ops",
    "JP": "asia-ops",
    "TW": "asia-ops",
    SHARED_SENTINEL: "platform-ops",
}


def thresholds_for(metric_key: str, market: str) -> Optional[Levels]:
    """Return ``Levels`` for ``(metric, market)`` or None when no policy exists.

    Falls back to the SHARED policy when the market-specific entry is missing,
    so adding a new metric doesn't require touching every market.
    """
    by_market = THRESHOLDS.get(metric_key)
    if not by_market:
        return None
    return by_market.get(market) or by_market.get(SHARED_SENTINEL)


def owner_for(market: str) -> Optional[str]:
    return OWNERS.get(market) or OWNERS.get(SHARED_SENTINEL)
