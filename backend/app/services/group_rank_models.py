"""Typed data contracts for group-ranking input and output."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
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
        if not isinstance(self.prices_by_symbol, Mapping):
            raise TypeError("prices_by_symbol must be a mapping")
        if not isinstance(self.active_symbols, frozenset):
            raise TypeError("active_symbols must be frozenset[str]")
        if not isinstance(self.market_caps, Mapping):
            raise TypeError("market_caps must be a mapping")
        if not isinstance(self.stats, GroupRankPrefetchStats):
            raise TypeError("stats must be GroupRankPrefetchStats")
        if not isinstance(self.symbols_by_group, Mapping):
            raise TypeError("symbols_by_group must be a mapping")
        for group, symbols in self.symbols_by_group.items():
            if not isinstance(symbols, tuple):
                raise TypeError(
                    f"symbols_by_group[{group!r}] must be tuple[str, ...]"
                )


@dataclass(frozen=True)
class GroupRanking:
    industry_group: str
    date: date
    rank: int
    avg_rs_rating: float
    median_rs_rating: float | None
    weighted_avg_rs_rating: float | None
    rs_std_dev: float | None
    num_stocks: int
    num_stocks_rs_above_80: int
    top_symbol: str | None
    top_rs_rating: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "industry_group": self.industry_group,
            "date": self.date,
            "rank": self.rank,
            "avg_rs_rating": self.avg_rs_rating,
            "median_rs_rating": self.median_rs_rating,
            "weighted_avg_rs_rating": self.weighted_avg_rs_rating,
            "rs_std_dev": self.rs_std_dev,
            "num_stocks": self.num_stocks,
            "num_stocks_rs_above_80": self.num_stocks_rs_above_80,
            "top_symbol": self.top_symbol,
            "top_rs_rating": self.top_rs_rating,
        }


@dataclass(frozen=True)
class GroupRankCalculationResult:
    rankings: tuple[GroupRanking, ...]
    prefetch_stats: GroupRankPrefetchStats
