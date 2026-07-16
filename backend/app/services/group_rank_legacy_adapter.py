"""Explicit compatibility adapter for legacy group-prefetch values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .group_rank_models import (
    GroupRankPrefetchData,
    GroupRankPrefetchStats,
)


@dataclass(frozen=True)
class LegacyGroupRankPrefetchAdapter:
    def adapt(self, value: object) -> GroupRankPrefetchData:
        if isinstance(value, GroupRankPrefetchData):
            return value
        if not isinstance(value, tuple) or len(value) != 5:
            raise TypeError(
                "unsupported legacy group prefetch value; expected typed "
                "GroupRankPrefetchData or five-item tuple"
            )
        benchmark, prices, active, market_caps, raw_stats = value
        if not isinstance(raw_stats, Mapping):
            raise TypeError("legacy group prefetch stats must be a mapping")
        return GroupRankPrefetchData(
            benchmark_prices=benchmark,
            prices_by_symbol=prices,
            active_symbols=frozenset(active),
            market_caps=market_caps,
            stats=self._stats(raw_stats),
            symbols_by_group={},
        )

    @staticmethod
    def _stats(values: Mapping[str, Any]) -> GroupRankPrefetchStats:
        target_symbols = int(values.get("target_symbols", 0) or 0)
        symbols_with_prices = int(
            values.get("symbols_with_prices", 0) or 0
        )
        coverage = values.get("cache_coverage_ratio")
        return GroupRankPrefetchStats(
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
