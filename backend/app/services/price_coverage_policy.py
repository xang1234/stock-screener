"""Market-specific price coverage policy for cache-only workflows."""

from __future__ import annotations

from dataclasses import dataclass


CACHE_ONLY_MIN_PRICE_COVERAGE = 0.95
CACHE_ONLY_MIN_FUNDAMENTALS_COVERAGE = 0.95

# Price coverage thresholds are aligned to observed daily-price static bundle
# availability, rounded down to conservative 5-point floors. They gate whether
# cache-only workflows can publish without falling back to live fetches.
PRICE_MIN_COVERAGE_BY_MARKET: dict[str, float] = {
    "AU": 0.90,
    "CA": 0.75,
    "CN": 0.90,
    "DE": 0.90,
    "HK": 0.80,
    "IN": 0.50,
    "JP": 0.90,
    "KR": 0.95,
    "MY": 0.85,
    "SG": 0.60,
    "TW": 0.50,
    "US": 0.95,
}


@dataclass(frozen=True)
class PriceCoveragePolicy:
    market: str
    price_min_coverage: float
    fundamentals_min_coverage: float


def normalize_market_code(market: str | None) -> str:
    return str(market or "US").strip().upper() or "US"


def price_coverage_policy_for_market(market: str | None) -> PriceCoveragePolicy:
    normalized_market = normalize_market_code(market)
    return PriceCoveragePolicy(
        market=normalized_market,
        price_min_coverage=PRICE_MIN_COVERAGE_BY_MARKET.get(
            normalized_market,
            CACHE_ONLY_MIN_PRICE_COVERAGE,
        ),
        fundamentals_min_coverage=CACHE_ONLY_MIN_FUNDAMENTALS_COVERAGE,
    )
