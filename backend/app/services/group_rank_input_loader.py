"""Load benchmark and constituent inputs for group-ranking calculations."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import pandas as pd
from sqlalchemy.orm import Session

from ..domain.providers.price_symbol_support import (
    is_unsupported_yahoo_price_symbol,
)
from ..models.stock_universe import StockUniverse
from .benchmark_cache_service import BenchmarkCacheService
from .derived_data_execution_policy import DerivedDataExecutionPolicy
from .group_rank_models import (
    GroupRankPrefetchData,
    GroupRankPrefetchStats,
)
from .ibd_industry_service import IBDIndustryService
from .price_cache_service import PriceCacheService


logger = logging.getLogger(__name__)
CACHE_MISS_SYMBOL_SAMPLE_LIMIT = 20


class GroupRankInputLoader:
    def __init__(
        self,
        *,
        price_cache: PriceCacheService,
        benchmark_cache: BenchmarkCacheService,
        market_cap_loader: (
            Callable[[Session, list[str]], dict[str, float]] | None
        ) = None,
    ) -> None:
        self.price_cache = price_cache
        self.benchmark_cache = benchmark_cache
        self.market_cap_loader = market_cap_loader

    def load(
        self,
        db: Session,
        *,
        market: str,
        policy: DerivedDataExecutionPolicy,
    ) -> GroupRankPrefetchData:
        from ..wiring.bootstrap import get_stock_universe_service

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

        active_symbols = frozenset(
            get_stock_universe_service().get_active_symbols(
                db,
                market=normalized_market,
            )
        )
        groups = IBDIndustryService.get_all_groups(
            db,
            market=normalized_market,
        )
        symbols_by_group: dict[str, tuple[str, ...]] = {}
        symbols_to_fetch: set[str] = set()
        unsupported_symbols: set[str] = set()

        for group in groups:
            validated: list[str] = []
            for symbol in IBDIndustryService.get_group_symbols(
                db,
                group,
                market=normalized_market,
            ):
                if symbol not in active_symbols:
                    continue
                if is_unsupported_yahoo_price_symbol(symbol):
                    unsupported_symbols.add(symbol)
                    continue
                validated.append(symbol)
            symbols_by_group[group] = tuple(validated)
            symbols_to_fetch.update(validated)

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
        market_cap_loader = self.market_cap_loader or self._market_caps
        market_caps = market_cap_loader(db, ordered_symbols)
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
        candidates = [primary_symbol]
        candidate_fn = getattr(
            self.benchmark_cache,
            "get_benchmark_candidates",
            None,
        )
        if callable(candidate_fn):
            try:
                resolved = [
                    str(symbol)
                    for symbol in candidate_fn(market)
                    if symbol
                ]
                if resolved:
                    candidates = resolved
            except Exception:
                logger.debug(
                    "Could not resolve benchmark candidates for market=%s",
                    market,
                    exc_info=True,
                )

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

    @staticmethod
    def _market_caps(
        db: Session,
        symbols: list[str],
    ) -> dict[str, float]:
        if not symbols:
            return {}
        rows = db.query(
            StockUniverse.symbol,
            StockUniverse.market_cap,
        ).filter(StockUniverse.symbol.in_(symbols)).all()
        return {
            symbol: market_cap
            for symbol, market_cap in rows
            if market_cap and market_cap > 0
        }
