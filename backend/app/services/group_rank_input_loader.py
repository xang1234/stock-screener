"""Load benchmark and constituent inputs for group-ranking calculations."""

from __future__ import annotations

from dataclasses import replace
import logging
from typing import Optional, Sequence

import pandas as pd
from sqlalchemy.orm import Session

from ..domain.providers.price_symbol_support import (
    is_unsupported_yahoo_price_symbol,
)
from .benchmark_cache_service import BenchmarkCacheService
from .derived_data_execution_policy import DerivedDataExecutionPolicy
from .group_rank_input_sources import (
    GroupRankMarketCapSource,
    GroupRankTaxonomySource,
    GroupRankUniverseSource,
)
from .group_rank_models import (
    GroupRankPrefetchData,
    GroupRankPrefetchStats,
)
from .price_cache_service import PriceCacheService


logger = logging.getLogger(__name__)
CACHE_MISS_SYMBOL_SAMPLE_LIMIT = 20


class GroupRankInputLoader:
    def __init__(
        self,
        *,
        price_cache: PriceCacheService,
        benchmark_cache: BenchmarkCacheService,
        universe_source: GroupRankUniverseSource,
        taxonomy_source: GroupRankTaxonomySource,
        market_cap_source: GroupRankMarketCapSource,
    ) -> None:
        self.price_cache = price_cache
        self.benchmark_cache = benchmark_cache
        self.universe_source = universe_source
        self.taxonomy_source = taxonomy_source
        self.market_cap_source = market_cap_source

    def load(
        self,
        db: Session,
        *,
        market: str,
        policy: DerivedDataExecutionPolicy,
    ) -> GroupRankPrefetchData:
        normalized_market = (market or "US").upper()
        primary_benchmark = self.benchmark_cache.get_benchmark_symbol(
            normalized_market
        )
        benchmark_role = "primary"
        benchmark_symbol = primary_benchmark

        if policy.cache_only:
            benchmark_prices, benchmark_symbol, benchmark_role = (
                self._get_cached_benchmark(
                    normalized_market,
                    primary_benchmark,
                    period="2y",
                )
            )
        else:
            benchmark_prices = self.benchmark_cache.get_benchmark_data(
                market=normalized_market,
                period="2y",
            )

        if benchmark_prices is None or benchmark_prices.empty:
            return self._empty_prefetch(
                market=normalized_market,
                policy=policy,
                benchmark_symbol=benchmark_symbol,
                benchmark_role=benchmark_role,
            )

        active_symbols = self.universe_source.active_symbols(
            db,
            normalized_market,
        )
        groups = self.taxonomy_source.groups(
            db,
            normalized_market,
        )
        symbols_by_group = self._validated_symbols_by_group(
            db,
            market=normalized_market,
            group_names=groups,
            active_symbols=active_symbols,
        )
        symbols_to_fetch = {
            symbol
            for symbols in symbols_by_group.values()
            for symbol in symbols
        }
        unsupported_symbols = {
            symbol
            for group in groups
            for symbol in self.taxonomy_source.symbols_for_group(
                db,
                group,
                normalized_market,
            )
            if (
                symbol in active_symbols
                and is_unsupported_yahoo_price_symbol(symbol)
            )
        }

        ordered_symbols = sorted(symbols_to_fetch)
        if policy.cache_only:
            prices = self.price_cache.get_many_cached_only_fresh(
                ordered_symbols,
                period="2y",
            )
        else:
            prices = self.price_cache.get_many(
                ordered_symbols,
                period="2y",
            )

        missing = tuple(
            symbol
            for symbol in ordered_symbols
            if prices.get(symbol) is None or prices[symbol].empty
        )
        valid_count = len(ordered_symbols) - len(missing)
        market_caps = self.market_cap_source.market_caps(
            db,
            ordered_symbols,
        )
        stats = GroupRankPrefetchStats(
            target_symbols=len(ordered_symbols),
            symbols_with_prices=valid_count,
            cache_miss_symbols=len(missing),
            cache_miss_symbols_sample=missing[
                :CACHE_MISS_SYMBOL_SAMPLE_LIMIT
            ],
            cache_coverage_ratio=(
                valid_count / len(ordered_symbols)
                if ordered_symbols
                else 1.0
            ),
            benchmark_available=True,
            benchmark_cached=policy.cache_only,
            benchmark_symbol=benchmark_symbol,
            benchmark_role=benchmark_role,
            market=normalized_market,
            cache_only=policy.cache_only,
            skipped_unsupported_symbols=len(unsupported_symbols),
        )
        return GroupRankPrefetchData(
            benchmark_prices=benchmark_prices,
            prices_by_symbol=prices,
            active_symbols=active_symbols,
            market_caps=market_caps,
            stats=stats,
            symbols_by_group=symbols_by_group,
        )

    def complete_legacy_symbols(
        self,
        db: Session,
        *,
        market: str,
        group_names: Sequence[str],
        prefetch: GroupRankPrefetchData,
    ) -> GroupRankPrefetchData:
        if prefetch.symbols_by_group:
            return prefetch
        symbols_by_group = self._validated_symbols_by_group(
            db,
            market=market,
            group_names=group_names,
            active_symbols=prefetch.active_symbols,
        )
        return replace(prefetch, symbols_by_group=symbols_by_group)

    def _validated_symbols_by_group(
        self,
        db: Session,
        *,
        market: str,
        group_names: Sequence[str],
        active_symbols: frozenset[str],
    ) -> dict[str, tuple[str, ...]]:
        return {
            group: tuple(
                symbol
                for symbol in self.taxonomy_source.symbols_for_group(
                    db,
                    group,
                    market,
                )
                if (
                    symbol in active_symbols
                    and not is_unsupported_yahoo_price_symbol(symbol)
                )
            )
            for group in group_names
        }

    def _empty_prefetch(
        self,
        *,
        market: str,
        policy: DerivedDataExecutionPolicy,
        benchmark_symbol: str,
        benchmark_role: str,
    ) -> GroupRankPrefetchData:
        stats = GroupRankPrefetchStats(
            target_symbols=0,
            symbols_with_prices=0,
            cache_miss_symbols=0,
            cache_miss_symbols_sample=(),
            cache_coverage_ratio=0.0,
            benchmark_available=False,
            benchmark_cached=False,
            benchmark_symbol=benchmark_symbol,
            benchmark_role=benchmark_role,
            market=market,
            cache_only=policy.cache_only,
            skipped_unsupported_symbols=0,
        )
        return GroupRankPrefetchData(
            benchmark_prices=None,
            prices_by_symbol={},
            active_symbols=frozenset(),
            market_caps={},
            stats=stats,
            symbols_by_group={},
        )

    def _get_cached_benchmark(
        self,
        market: str,
        primary_symbol: str,
        *,
        period: str,
    ) -> tuple[Optional[pd.DataFrame], str, str]:
        resolved = [
            str(symbol)
            for symbol in self.benchmark_cache.get_benchmark_candidates(
                market
            )
            if symbol
        ]
        candidates = resolved or [primary_symbol]

        for index, candidate in enumerate(candidates):
            data = self.price_cache.get_cached_only_fresh(
                candidate,
                period=period,
            )
            if data is not None and not data.empty:
                role = "primary" if index == 0 else "fallback"
                if role == "fallback":
                    logger.info(
                        "Using cached fallback benchmark %s for market %s",
                        candidate,
                        market,
                    )
                return data, candidate, role
        return None, primary_symbol, "primary"
