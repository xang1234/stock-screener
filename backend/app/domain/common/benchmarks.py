"""Canonical market benchmark symbols shared by domain defaults and services."""

from __future__ import annotations

from app.domain.markets import UnsupportedMarketError, market_registry


PRIMARY_BENCHMARK_SYMBOL_BY_MARKET = {
    profile.market.code: profile.primary_benchmark_symbol
    for profile in market_registry.profiles()
}


def normalize_benchmark_market(market: str | None) -> str:
    try:
        return market_registry.profile(market or "US").market.code
    except UnsupportedMarketError as exc:
        supported = ", ".join(supported_benchmark_markets())
        raise ValueError(
            f"Unsupported market for benchmark registry: {market}. Supported: {supported}"
        ) from exc


def get_primary_benchmark_symbol(market: str | None) -> str:
    return PRIMARY_BENCHMARK_SYMBOL_BY_MARKET[normalize_benchmark_market(market)]


def supported_benchmark_markets() -> list[str]:
    return list(market_registry.supported_market_codes())
