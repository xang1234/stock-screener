"""Canonical market benchmark symbols shared by domain defaults and services."""

from __future__ import annotations

PRIMARY_BENCHMARK_SYMBOL_BY_MARKET = {
    "US": "SPY",
    "HK": "^HSI",
    "IN": "^NSEI",
    "JP": "^N225",
    "KR": "^KS11",
    "TW": "^TWII",
}


def normalize_benchmark_market(market: str | None) -> str:
    normalized = (market or "US").strip().upper()
    if normalized not in PRIMARY_BENCHMARK_SYMBOL_BY_MARKET:
        supported = ", ".join(supported_benchmark_markets())
        raise ValueError(f"Unsupported market for benchmark registry: {market}. Supported: {supported}")
    return normalized


def get_primary_benchmark_symbol(market: str | None) -> str:
    return PRIMARY_BENCHMARK_SYMBOL_BY_MARKET[normalize_benchmark_market(market)]


def supported_benchmark_markets() -> list[str]:
    return sorted(PRIMARY_BENCHMARK_SYMBOL_BY_MARKET)
