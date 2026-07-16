"""Typed data contracts for group-ranking input and output."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

import pandas as pd

from .group_rank_cache_policy import GroupRankCacheRequirement


@dataclass(frozen=True)
class GroupRankPrefetchStats:
    target_symbols: int
    symbols_with_prices: int
    cache_miss_symbols: int
    cache_miss_symbols_sample: tuple[str, ...]
    cache_coverage_ratio: float
    benchmark_available: bool
    benchmark_cached: bool
    benchmark_symbol: str
    benchmark_role: str
    market: str
    cache_only: bool
    skipped_unsupported_symbols: int
    cache_coverage_min: float | None = None
    cache_requirement_reason: str | None = None

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> "GroupRankPrefetchStats":
        target_symbols = int(values.get("target_symbols", 0) or 0)
        symbols_with_prices = int(
            values.get("symbols_with_prices", 0) or 0
        )
        coverage = values.get("cache_coverage_ratio")
        return cls(
            target_symbols=target_symbols,
            symbols_with_prices=symbols_with_prices,
            cache_miss_symbols=int(
                values.get("cache_miss_symbols", 0) or 0
            ),
            cache_miss_symbols_sample=tuple(
                values.get("cache_miss_symbols_sample", ())
            ),
            cache_coverage_ratio=(
                float(coverage)
                if coverage is not None
                else (
                    symbols_with_prices / target_symbols
                    if target_symbols
                    else 1.0
                )
            ),
            benchmark_available=bool(
                values.get(
                    "benchmark_available",
                    values.get("spy_cached", False),
                )
            ),
            benchmark_cached=bool(
                values.get("benchmark_cached", False)
            ),
            benchmark_symbol=str(
                values.get("benchmark_symbol", "SPY")
            ),
            benchmark_role=str(
                values.get("benchmark_role", "primary")
            ),
            market=str(values.get("market", "US")).upper(),
            cache_only=bool(values.get("cache_only", False)),
            skipped_unsupported_symbols=int(
                values.get("skipped_unsupported_symbols", 0) or 0
            ),
            cache_coverage_min=values.get("cache_coverage_min"),
            cache_requirement_reason=values.get(
                "cache_requirement_reason"
            ),
        )

    def with_cache_requirement(
        self,
        requirement: GroupRankCacheRequirement,
    ) -> "GroupRankPrefetchStats":
        if not requirement.enabled:
            return self
        return replace(
            self,
            cache_coverage_min=requirement.min_coverage,
            cache_requirement_reason=requirement.reason,
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "target_symbols": self.target_symbols,
            "symbols_with_prices": self.symbols_with_prices,
            "cache_miss_symbols": self.cache_miss_symbols,
            "cache_miss_symbols_sample": list(
                self.cache_miss_symbols_sample
            ),
            "cache_coverage_ratio": self.cache_coverage_ratio,
            "spy_cached": self.benchmark_available,
            "benchmark_cached": self.benchmark_cached,
            "benchmark_symbol": self.benchmark_symbol,
            "benchmark_role": self.benchmark_role,
            "market": self.market,
            "cache_only": self.cache_only,
            "skipped_unsupported_symbols": (
                self.skipped_unsupported_symbols
            ),
        }
        if self.cache_coverage_min is not None:
            result["cache_coverage_min"] = self.cache_coverage_min
        if self.cache_requirement_reason is not None:
            result["cache_requirement_reason"] = (
                self.cache_requirement_reason
            )
        return result


@dataclass(frozen=True)
class GroupRankPrefetchData:
    benchmark_prices: Optional[pd.DataFrame]
    prices_by_symbol: Mapping[str, Optional[pd.DataFrame]]
    active_symbols: frozenset[str]
    market_caps: Mapping[str, float]
    stats: GroupRankPrefetchStats
    symbols_by_group: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "active_symbols",
            frozenset(self.active_symbols),
        )
        if not isinstance(self.stats, GroupRankPrefetchStats):
            object.__setattr__(
                self,
                "stats",
                GroupRankPrefetchStats.from_mapping(self.stats),
            )
        object.__setattr__(
            self,
            "symbols_by_group",
            {
                group: tuple(symbols)
                for group, symbols in self.symbols_by_group.items()
            },
        )


@dataclass(frozen=True)
class GroupRankCalculationResult:
    rankings: tuple[Mapping[str, Any], ...]
    prefetch_stats: GroupRankPrefetchStats
