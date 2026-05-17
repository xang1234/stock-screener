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


def _build_benchmark_entry(market: str, notes: str) -> BenchmarkRegistryEntry:
    profile = market_registry.profile(market)
    return BenchmarkRegistryEntry(
        market=market,
        primary_symbol=get_primary_benchmark_symbol(market),
        primary_kind=profile.benchmark_primary_kind,
        fallback_symbol=profile.benchmark_fallback_symbol,
        fallback_kind=profile.benchmark_fallback_kind,
        notes=notes,
    )


class BenchmarkRegistryService:
    """Operator-visible benchmark mapping table for US/HK/IN/JP/KR/TW/CN/CA/DE/SG."""

    TABLE_VERSION = "2026-05-09.v1"

    _NOTES_BY_MARKET: Dict[str, str] = {
        "US": "US baseline benchmark; ETF primary keeps behavior parity.",
        "HK": "Index-primary for market semantics; ETF fallback when index feed unavailable.",
        "IN": "NIFTY 50 index-primary with NSE ETF fallback.",
        "JP": "Nikkei index-primary with TOPIX ETF fallback.",
        "KR": "KOSPI index-primary with KODEX 200 ETF fallback.",
        "TW": "TAIEX index-primary with TW50 ETF fallback.",
        "CN": "CSI 300 index-primary with Shanghai Composite fallback.",
        "CA": "S&P/TSX Composite index-primary with iShares S&P/TSX 60 (XIU.TO) ETF fallback.",
        "DE": "DAX index-primary with iShares DAX UCITS ETF (EXS1.DE) fallback.",
        "SG": "Straits Times Index index-primary with SPDR STI ETF (ES3.SI) fallback.",
    }
    _TABLE: Dict[str, BenchmarkRegistryEntry] = {
        market: _build_benchmark_entry(market, notes)
        for market, notes in _NOTES_BY_MARKET.items()
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
