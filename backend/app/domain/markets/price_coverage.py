"""Market-specific price coverage rules for cache-only workflows."""

from __future__ import annotations

from dataclasses import dataclass


CACHE_ONLY_MIN_PRICE_COVERAGE = 0.95
CACHE_ONLY_MIN_FUNDAMENTALS_COVERAGE = 0.95

# Thresholds follow observed live/cache coverage availability, rounded down to
# conservative five-point floors.
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

BOOTSTRAP_PRICE_MIN_COVERAGE_BY_MARKET: dict[str, float] = {
    **PRICE_MIN_COVERAGE_BY_MARKET,
    # HK has many listed-but-stale symbols, so its durable GitHub bundle
    # coverage currently sits just above 75% during first-run bootstrap.
    "HK": 0.75,
}


@dataclass(frozen=True)
class PriceCoveragePolicy:
    market: str
    price_min_coverage: float
    fundamentals_min_coverage: float


def normalize_market_code(market: str | None) -> str:
    return str(market or "US").strip().upper() or "US"


def _price_coverage_policy_for_market(
    market: str | None,
    *,
    price_thresholds: dict[str, float],
) -> PriceCoveragePolicy:
    normalized_market = normalize_market_code(market)
    return PriceCoveragePolicy(
        market=normalized_market,
        price_min_coverage=price_thresholds.get(
            normalized_market,
            CACHE_ONLY_MIN_PRICE_COVERAGE,
        ),
        fundamentals_min_coverage=CACHE_ONLY_MIN_FUNDAMENTALS_COVERAGE,
    )


def price_coverage_policy_for_market(market: str | None) -> PriceCoveragePolicy:
    return _price_coverage_policy_for_market(
        market,
        price_thresholds=PRICE_MIN_COVERAGE_BY_MARKET,
    )


def bootstrap_price_coverage_policy_for_market(
    market: str | None,
) -> PriceCoveragePolicy:
    return _price_coverage_policy_for_market(
        market,
        price_thresholds=BOOTSTRAP_PRICE_MIN_COVERAGE_BY_MARKET,
    )


__all__ = [
    "BOOTSTRAP_PRICE_MIN_COVERAGE_BY_MARKET",
    "CACHE_ONLY_MIN_FUNDAMENTALS_COVERAGE",
    "CACHE_ONLY_MIN_PRICE_COVERAGE",
    "PRICE_MIN_COVERAGE_BY_MARKET",
    "PriceCoveragePolicy",
    "bootstrap_price_coverage_policy_for_market",
    "normalize_market_code",
    "price_coverage_policy_for_market",
]
