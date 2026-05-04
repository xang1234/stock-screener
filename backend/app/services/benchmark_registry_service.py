"""Canonical benchmark mapping registry with deterministic fallback policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..domain.markets import market_registry
from ..domain.common.benchmarks import (
    get_primary_benchmark_symbol,
    normalize_benchmark_market,
    supported_benchmark_markets,
)


@dataclass(frozen=True)
class BenchmarkRegistryEntry:
    market: str
    primary_symbol: str
    primary_kind: str
    fallback_symbol: str | None
    fallback_kind: str | None
    notes: str


class BenchmarkRegistryService:
    """Operator-visible benchmark mapping table for US/HK/IN/JP/KR/TW/CN."""

    TABLE_VERSION = "2026-04-30.v1"

    _TABLE: Dict[str, BenchmarkRegistryEntry] = {
        "US": BenchmarkRegistryEntry(
            market="US",
            primary_symbol=get_primary_benchmark_symbol("US"),
            primary_kind=market_registry.profile("US").benchmark_primary_kind,
            fallback_symbol=market_registry.profile("US").benchmark_fallback_symbol,
            fallback_kind=market_registry.profile("US").benchmark_fallback_kind,
            notes="US baseline benchmark; ETF primary keeps behavior parity.",
        ),
        "HK": BenchmarkRegistryEntry(
            market="HK",
            primary_symbol=get_primary_benchmark_symbol("HK"),
            primary_kind=market_registry.profile("HK").benchmark_primary_kind,
            fallback_symbol=market_registry.profile("HK").benchmark_fallback_symbol,
            fallback_kind=market_registry.profile("HK").benchmark_fallback_kind,
            notes="Index-primary for market semantics; ETF fallback when index feed unavailable.",
        ),
        "IN": BenchmarkRegistryEntry(
            market="IN",
            primary_symbol=get_primary_benchmark_symbol("IN"),
            primary_kind=market_registry.profile("IN").benchmark_primary_kind,
            fallback_symbol=market_registry.profile("IN").benchmark_fallback_symbol,
            fallback_kind=market_registry.profile("IN").benchmark_fallback_kind,
            notes="NIFTY 50 index-primary with NSE ETF fallback.",
        ),
        "JP": BenchmarkRegistryEntry(
            market="JP",
            primary_symbol=get_primary_benchmark_symbol("JP"),
            primary_kind=market_registry.profile("JP").benchmark_primary_kind,
            fallback_symbol=market_registry.profile("JP").benchmark_fallback_symbol,
            fallback_kind=market_registry.profile("JP").benchmark_fallback_kind,
            notes="Nikkei index-primary with TOPIX ETF fallback.",
        ),
        "KR": BenchmarkRegistryEntry(
            market="KR",
            primary_symbol=get_primary_benchmark_symbol("KR"),
            primary_kind=market_registry.profile("KR").benchmark_primary_kind,
            fallback_symbol=market_registry.profile("KR").benchmark_fallback_symbol,
            fallback_kind=market_registry.profile("KR").benchmark_fallback_kind,
            notes="KOSPI index-primary with KODEX 200 ETF fallback.",
        ),
        "TW": BenchmarkRegistryEntry(
            market="TW",
            primary_symbol=get_primary_benchmark_symbol("TW"),
            primary_kind=market_registry.profile("TW").benchmark_primary_kind,
            fallback_symbol=market_registry.profile("TW").benchmark_fallback_symbol,
            fallback_kind=market_registry.profile("TW").benchmark_fallback_kind,
            notes="TAIEX index-primary with TW50 ETF fallback.",
        ),
        "CN": BenchmarkRegistryEntry(
            market="CN",
            primary_symbol=get_primary_benchmark_symbol("CN"),
            primary_kind=market_registry.profile("CN").benchmark_primary_kind,
            fallback_symbol=market_registry.profile("CN").benchmark_fallback_symbol,
            fallback_kind=market_registry.profile("CN").benchmark_fallback_kind,
            notes="CSI 300 index-primary with Shanghai Composite fallback.",
        ),
    }

    def normalize_market(self, market: str | None) -> str:
        return normalize_benchmark_market(market)

    def get_entry(self, market: str) -> BenchmarkRegistryEntry:
        return self._TABLE[self.normalize_market(market)]

    def get_primary_symbol(self, market: str) -> str:
        return self.get_entry(market).primary_symbol

    def get_candidate_symbols(self, market: str) -> List[str]:
        entry = self.get_entry(market)
        candidates = [entry.primary_symbol]
        if entry.fallback_symbol:
            candidates.append(entry.fallback_symbol)
        return candidates

    def supported_markets(self) -> List[str]:
        return supported_benchmark_markets()

    def mapping_table(self) -> Dict[str, Dict[str, str | None]]:
        table: Dict[str, Dict[str, str | None]] = {}
        for market in self.supported_markets():
            entry = self._TABLE[market]
            table[market] = {
                "primary_symbol": entry.primary_symbol,
                "primary_kind": entry.primary_kind,
                "fallback_symbol": entry.fallback_symbol,
                "fallback_kind": entry.fallback_kind,
                "notes": entry.notes,
                "version": self.TABLE_VERSION,
            }
        return table


benchmark_registry = BenchmarkRegistryService()
