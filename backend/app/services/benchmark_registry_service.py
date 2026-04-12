"""Canonical benchmark mapping registry with deterministic fallback policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class BenchmarkRegistryEntry:
    market: str
    primary_symbol: str
    primary_kind: str
    fallback_symbol: str | None
    fallback_kind: str | None
    notes: str


class BenchmarkRegistryService:
    """Operator-visible benchmark mapping table for US/HK/JP/TW."""

    TABLE_VERSION = "2026-04-11.v1"

    _TABLE: Dict[str, BenchmarkRegistryEntry] = {
        "US": BenchmarkRegistryEntry(
            market="US",
            primary_symbol="SPY",
            primary_kind="etf",
            fallback_symbol="IVV",
            fallback_kind="etf",
            notes="US baseline benchmark; ETF primary keeps behavior parity.",
        ),
        "HK": BenchmarkRegistryEntry(
            market="HK",
            primary_symbol="^HSI",
            primary_kind="index",
            fallback_symbol="2800.HK",
            fallback_kind="etf",
            notes="Index-primary for market semantics; ETF fallback when index feed unavailable.",
        ),
        "JP": BenchmarkRegistryEntry(
            market="JP",
            primary_symbol="^N225",
            primary_kind="index",
            fallback_symbol="1306.T",
            fallback_kind="etf",
            notes="Nikkei index-primary with TOPIX ETF fallback.",
        ),
        "TW": BenchmarkRegistryEntry(
            market="TW",
            primary_symbol="^TWII",
            primary_kind="index",
            fallback_symbol="0050.TW",
            fallback_kind="etf",
            notes="TAIEX index-primary with TW50 ETF fallback.",
        ),
    }

    def normalize_market(self, market: str | None) -> str:
        normalized = (market or "US").strip().upper()
        if normalized not in self._TABLE:
            raise ValueError(f"Unsupported market for benchmark registry: {market}")
        return normalized

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
        return sorted(self._TABLE.keys())

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
