"""Stable Market facts and lookup helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .catalog import MarketCatalog, get_market_catalog
from .market import Market
from .mic_aliases import mic_alias_registry


@dataclass(frozen=True, slots=True)
class BenchmarkFacts:
    """Benchmark compatibility facts not owned by Market Catalog."""

    primary_symbol: str
    fallback_symbol: str | None
    primary_kind: str
    fallback_kind: str | None


@dataclass(frozen=True, slots=True)
class MarketProfile:
    """Stable facts about one supported Market."""

    market: Market
    label: str
    currency: str
    timezone_name: str
    calendar_id: str
    provider_calendar_id: str | None
    exchanges: tuple[str, ...]
    primary_benchmark_symbol: str
    benchmark_fallback_symbol: str | None
    benchmark_primary_kind: str
    benchmark_fallback_kind: str | None

    @property
    def indexes(self) -> tuple[str, ...]:
        """Derived compatibility summary; IndexRegistry owns definitions."""
        from ..universe.indexes import index_registry

        return tuple(
            definition.key for definition in index_registry.definitions(self.market.code)
        )


class MarketRegistry:
    """Registry of stable Market facts.

    Runtime Preferences and Market Workload are separate concepts. This module
    owns stable identity and lookup facts only.
    """

    def __init__(self, profiles: Iterable[MarketProfile]) -> None:
        self._profiles = tuple(profiles)
        self._by_code: dict[str, MarketProfile] = {}

        for profile in self._profiles:
            code = profile.market.code
            if code in self._by_code:
                raise ValueError(f"Duplicate market profile: {code}")
            self._by_code[code] = profile

    @classmethod
    def from_catalog(
        cls,
        catalog: MarketCatalog,
        *,
        benchmark_facts: Mapping[str, BenchmarkFacts],
    ) -> "MarketRegistry":
        profiles: list[MarketProfile] = []
        catalog_codes = tuple(catalog.supported_market_codes())
        missing = sorted(set(catalog_codes) - set(benchmark_facts))
        if missing:
            raise ValueError(
                "Benchmark facts are missing for supported Markets: "
                + ", ".join(missing)
            )
        for code in catalog_codes:
            entry = catalog.get(code)
            benchmark = benchmark_facts[code]
            profiles.append(
                MarketProfile(
                    market=Market(entry.code),
                    label=entry.label,
                    currency=entry.default_currency,
                    timezone_name=entry.display_timezone,
                    calendar_id=entry.calendar_id,
                    provider_calendar_id=entry.provider_calendar_id,
                    exchanges=entry.exchanges,
                    primary_benchmark_symbol=benchmark.primary_symbol,
                    benchmark_fallback_symbol=benchmark.fallback_symbol,
                    benchmark_primary_kind=benchmark.primary_kind,
                    benchmark_fallback_kind=benchmark.fallback_kind,
                )
            )
        return cls(profiles)

    def profile(self, market: Market | str) -> MarketProfile:
        resolved = market if isinstance(market, Market) else Market.from_str(market)
        return self._by_code[resolved.code]

    def profiles(self) -> tuple[MarketProfile, ...]:
        return self._profiles

    def supported_markets(self) -> tuple[Market, ...]:
        return tuple(profile.market for profile in self._profiles)

    def supported_market_codes(self) -> tuple[str, ...]:
        return tuple(profile.market.code for profile in self._profiles)

    def market_for_exchange(self, exchange: str | None) -> Market | None:
        resolved = mic_alias_registry.resolve_global(exchange)
        if resolved is None:
            return None
        return Market(resolved.market)

    def mic_for_exchange(
        self, market: Market | str | None, exchange: str | None
    ) -> str | None:
        if market is None:
            return None
        market_code = market.code if isinstance(market, Market) else str(market)
        resolved = mic_alias_registry.resolve(market_code, exchange)
        return resolved.mic if resolved else None

    def market_for_index(self, index: str | None) -> Market | None:
        from ..universe.indexes import index_registry

        market_code = index_registry.market_for(index)
        if market_code is None:
            return None
        return Market(market_code)


_BENCHMARK_FACTS_BY_MARKET: Mapping[str, BenchmarkFacts] = {
    "US": BenchmarkFacts("SPY", "IVV", "etf", "etf"),
    "HK": BenchmarkFacts("^HSI", "2800.HK", "index", "etf"),
    "IN": BenchmarkFacts("^NSEI", "NIFTYBEES.NS", "index", "etf"),
    "JP": BenchmarkFacts("^N225", "1306.T", "index", "etf"),
    "KR": BenchmarkFacts("^KS11", "069500.KS", "index", "etf"),
    "TW": BenchmarkFacts("^TWII", "0050.TW", "index", "etf"),
    "CN": BenchmarkFacts("000300.SS", "000001.SS", "index", "index"),
    "CA": BenchmarkFacts("^GSPTSE", "XIU.TO", "index", "etf"),
    "DE": BenchmarkFacts("^GDAXI", "EXS1.DE", "index", "etf"),
    "SG": BenchmarkFacts("^STI", "ES3.SI", "index", "etf"),
    "AU": BenchmarkFacts("^AXJO", "IOZ.AX", "index", "etf"),
    "MY": BenchmarkFacts("^KLSE", None, "index", None),
}


market_registry = MarketRegistry.from_catalog(
    get_market_catalog(),
    benchmark_facts=_BENCHMARK_FACTS_BY_MARKET,
)
