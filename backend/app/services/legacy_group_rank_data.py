"""Focused legacy Group ranking implementation extracted from the public facade."""

import logging
import math
import statistics
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Protocol

import pandas as pd
from sqlalchemy import and_
from sqlalchemy.orm import Session

from ..domain.providers.price_symbol_support import is_unsupported_yahoo_price_symbol
from ..domain.relative_strength import LEGACY_RS_FORMULA_VERSION
from ..models.industry import IBDGroupRank
from ..models.stock_universe import StockUniverse
from .ibd_industry_service import IBDIndustryService
from .legacy_group_rank_contracts import GroupRankPrefetchData

logger = logging.getLogger(__name__)


class ActiveSymbolsProvider(Protocol):
    def __call__(
        self,
        db: Session,
        *,
        market: str,
    ) -> List[str]: ...


class LegacyGroupRankingEngine:
    """Legacy RS data loader and calculator with explicit dependencies."""

    def __init__(
        self,
        *,
        price_cache,
        benchmark_cache,
        rs_calculator,
        active_symbols_provider: ActiveSymbolsProvider | None = None,
    ) -> None:
        self.price_cache = price_cache
        self.benchmark_cache = benchmark_cache
        self.rs_calculator = rs_calculator
        self.active_symbols_provider = (
            active_symbols_provider or self._load_active_symbols_from_database
        )

    @staticmethod
    def _load_active_symbols_from_database(
        db: Session,
        *,
        market: str,
    ) -> List[str]:
        rows = db.query(StockUniverse.symbol).filter(
            StockUniverse.active_filter(),
            StockUniverse.market == market,
        ).all()
        return [row[0] for row in rows]

    def validated_group_symbols(
        self,
        db: Session,
        group_name: str,
        active_symbols: set,
        *,
        market: str = "US",
    ) -> list:
        """
        Get symbols for a group, filtered to only active stocks in stock_universe.

        This ensures group ranking uses the same universe as bulk scans.
        """
        group_symbols = IBDIndustryService.get_group_symbols(
            db, group_name, market=market
        )
        return [
            symbol
            for symbol in group_symbols
            if symbol in active_symbols and not is_unsupported_yahoo_price_symbol(symbol)
        ]

    def market_caps_for_symbols(
        self,
        db: Session,
        symbols: List[str]
    ) -> Dict[str, float]:
        """
        Fetch market caps for a set of symbols from stock_universe.

        Returns:
            Dict mapping symbol -> market cap (only positive values included)
        """
        if not symbols:
            return {}

        rows = db.query(StockUniverse.symbol, StockUniverse.market_cap).filter(
            StockUniverse.symbol.in_(symbols)
        ).all()

        return {symbol: cap for symbol, cap in rows if cap and cap > 0}

    def calculate_pct_above_80(
        self,
        count_above_80: Optional[int],
        total_count: Optional[int]
    ) -> Optional[float]:
        """
        Calculate percent of group members with RS >= 80.
        """
        if not total_count:
            return None
        count = count_above_80 or 0
        return round((count / total_count) * 100, 1)

    @staticmethod
    def coerce_prefetch_data(prefetch: Any) -> GroupRankPrefetchData:
        """Accept legacy test tuples while the service uses a named prefetch object."""
        if isinstance(prefetch, GroupRankPrefetchData):
            return prefetch
        spy_data, all_prices, active_symbols, market_caps, stats = prefetch
        return GroupRankPrefetchData(
            benchmark_prices=spy_data,
            prices_by_symbol=all_prices,
            active_symbols=active_symbols,
            market_caps=market_caps,
            stats=stats,
            symbols_by_group={},
        )

    def symbols_by_group_for_run(
        self,
        db: Session,
        group_names: List[str],
        prefetch: GroupRankPrefetchData,
        *,
        market: str,
    ) -> Dict[str, List[str]]:
        """Return prefetched group symbols, or build them once for legacy prefetch tuples."""
        symbols_by_group: Dict[str, List[str]] = {}
        for group_name in group_names:
            if group_name in prefetch.symbols_by_group:
                symbols_by_group[group_name] = prefetch.symbols_by_group[group_name]
            else:
                symbols_by_group[group_name] = self.validated_group_symbols(
                    db,
                    group_name,
                    prefetch.active_symbols,
                    market=market,
                )
        return symbols_by_group

    def prefetch_all_data(
        self,
        db: Session,
        *,
        market: str | None = None,
        cache_only: bool = False,
    ) -> GroupRankPrefetchData:
        """
        Pre-fetch all data needed for backfill in one batch.

        This optimization fetches:
        1. SPY benchmark data once
        2. Active symbols from stock_universe (same source as bulk scans)
        3. All prices for all symbols across all groups in a single batch

        Returns:
            Tuple of (spy_prices_df, {symbol: prices_df}, active_symbols_set, {symbol: market_cap})
        """
        normalized_market = (market or "US").upper()
        benchmark_symbol = self.benchmark_cache.get_benchmark_symbol(normalized_market)
        logger.info(
            "Pre-fetching data for market=%s (benchmark=%s)...",
            normalized_market, benchmark_symbol,
        )

        # 1. Get benchmark data once (SPY for US, HSI/N225/TAIEX/NIFTY for Asia)
        benchmark_role = "primary"
        if cache_only:
            spy_data, benchmark_symbol, benchmark_role = self._get_cached_only_benchmark_data(
                normalized_market,
                benchmark_symbol,
                period="2y",
            )
        else:
            spy_data = self.benchmark_cache.get_benchmark_data(
                market=normalized_market, period="2y",
            )
        if spy_data is None or spy_data.empty:
            logger.error(
                "Failed to get benchmark data (%s) for market %s",
                benchmark_symbol, normalized_market,
            )
            return GroupRankPrefetchData(
                benchmark_prices=None,
                prices_by_symbol={},
                active_symbols=set(),
                market_caps={},
                stats={
                    "target_symbols": 0,
                    "symbols_with_prices": 0,
                    "cache_miss_symbols": 0,
                    "spy_cached": False,
                    "benchmark_cached": False,
                    "benchmark_symbol": benchmark_symbol,
                    "market": normalized_market,
                },
                symbols_by_group={},
            )

        logger.info(f"Fetched benchmark {benchmark_symbol}: {len(spy_data)} days")

        # 2. Get active symbols from stock_universe (same as bulk scans)
        active_symbols_list = self.active_symbols_provider(
            db,
            market=normalized_market,
        )
        active_symbols = set(active_symbols_list)
        logger.info(f"Found {len(active_symbols)} active symbols in stock_universe")

        # 3. Collect ALL unique symbols across ALL groups for this market
        memberships = IBDIndustryService.get_group_memberships(
            db,
            market=normalized_market,
        )
        all_groups = list(memberships)
        symbols_to_fetch = set()
        symbols_by_group: Dict[str, List[str]] = {}
        skipped_unsupported_symbols = set()

        for group in all_groups:
            group_symbols = memberships[group]
            validated = []
            for symbol in group_symbols:
                if symbol not in active_symbols:
                    continue
                if is_unsupported_yahoo_price_symbol(symbol):
                    skipped_unsupported_symbols.add(symbol)
                    continue
                validated.append(symbol)
            symbols_by_group[group] = validated
            symbols_to_fetch.update(validated)

        logger.info(
            f"Collecting prices for {len(symbols_to_fetch)} unique symbols "
            f"across {len(all_groups)} groups"
        )

        # 4. Batch fetch ALL prices in one call
        if cache_only:
            all_prices = self.price_cache.get_many_cached_only_fresh(
                list(symbols_to_fetch),
                period="2y",
            )
        else:
            all_prices = self.price_cache.get_many(list(symbols_to_fetch), period="2y")

        # 5. Fetch market caps for weighting
        market_caps = self.market_caps_for_symbols(db, list(symbols_to_fetch))

        valid_count = sum(
            1 for v in all_prices.values() if v is not None and not v.empty
        )
        logger.info(f"Pre-fetched prices: {valid_count}/{len(all_prices)} symbols have data")

        stats = {
            "target_symbols": len(symbols_to_fetch),
            "symbols_with_prices": valid_count,
            "cache_miss_symbols": len(symbols_to_fetch) - valid_count,
            "spy_cached": True,
            "skipped_unsupported_symbols": len(skipped_unsupported_symbols),
        }
        if benchmark_role != "primary":
            stats["benchmark_symbol"] = benchmark_symbol
            stats["benchmark_role"] = benchmark_role

        return GroupRankPrefetchData(
            benchmark_prices=spy_data,
            prices_by_symbol=all_prices,
            active_symbols=active_symbols,
            market_caps=market_caps,
            stats=stats,
            symbols_by_group=symbols_by_group,
        )

    def _get_cached_only_benchmark_data(
        self,
        market: str,
        primary_symbol: str,
        *,
        period: str,
    ) -> tuple[Optional[pd.DataFrame], str, str]:
        candidates = [primary_symbol]
        candidate_fn = getattr(self.benchmark_cache, "get_benchmark_candidates", None)
        if callable(candidate_fn):
            try:
                resolved = [str(symbol) for symbol in candidate_fn(market) if symbol]
                if resolved:
                    candidates = resolved
            except Exception:
                logger.debug("Could not resolve benchmark candidates for market=%s", market, exc_info=True)

        for idx, candidate in enumerate(candidates):
            role = "primary" if idx == 0 else "fallback"
            data = self.price_cache.get_cached_only_fresh(candidate, period=period)
            if data is not None and not data.empty:
                if role != "primary":
                    logger.info(
                        "Using cached fallback benchmark %s for market %s",
                        candidate,
                        market,
                    )
                return data, candidate, role

        return None, primary_symbol, "primary"

    def delete_rankings_for_range(
        self,
        db: Session,
        start_date: date,
        end_date: date,
        *,
        market: str = "US",
        formula_version: str = LEGACY_RS_FORMULA_VERSION,
    ) -> int:
        """Delete existing rankings for one market in a date range.

        Scoped by ``market`` — a US backfill must not wipe HK/JP/TW/IN rows
        that share the same date range.
        """
        normalized_market = (market or "US").upper()
        deleted = db.query(IBDGroupRank).filter(
            and_(
                IBDGroupRank.date >= start_date,
                IBDGroupRank.date <= end_date,
                IBDGroupRank.market == normalized_market,
                IBDGroupRank.rs_formula_version == formula_version,
            )
        ).delete(synchronize_session=False)

        db.commit()
        logger.info(
            "Deleted %d existing rankings for market=%s %s to %s",
            deleted, normalized_market, start_date, end_date,
        )
        return deleted

    def calculate_group_rs_from_cache(
        self,
        db: Session,
        group_name: str,
        spy_prices: pd.Series,
        all_prices: Dict[str, pd.DataFrame],
        active_symbols: set,
        market_caps: Dict[str, float],
        calculation_date: date,
        *,
        market: str = "US",
    ) -> Optional[Dict]:
        """
        Calculate average RS rating for a single industry group using pre-fetched prices.

        Only uses symbols that are active in stock_universe (intersection approach).
        ``spy_prices`` is the per-market benchmark series (SPY for US, HSI for HK, etc).
        """
        symbols = self.validated_group_symbols(
            db, group_name, active_symbols, market=market,
        )

        if not symbols:
            logger.debug(f"No validated symbols found for group: {group_name}")
            return None

        # Calculate RS for each symbol using cached prices
        rs_ratings = []
        top_symbol = None
        top_rs = -1
        rs_above_80_count = 0
        weighted_sum = 0.0
        weighted_total = 0.0

        for symbol in symbols:
            prices = all_prices.get(symbol)
            if prices is None or prices.empty:
                continue

            # Filter to calculation_date for accurate historical rankings
            if calculation_date < datetime.now().date():
                prices_filtered = prices[prices.index.date <= calculation_date]
            else:
                prices_filtered = prices

            # Skip if insufficient data after filtering
            if prices_filtered.empty:
                continue

            # Prepare stock prices (most recent first)
            stock_prices = prices_filtered['Close'].sort_index(ascending=False)

            # Need at least 252 days for full RS calculation
            if len(stock_prices) < 252:
                continue

            try:
                rs_result = self.rs_calculator.calculate_rs_rating(
                    symbol, stock_prices, spy_prices
                )
                rs_rating = rs_result.get('rs_rating', 0)

                if rs_rating > 0:
                    rs_ratings.append(rs_rating)

                    if rs_rating > top_rs:
                        top_rs = rs_rating
                        top_symbol = symbol

                    if rs_rating >= 80:
                        rs_above_80_count += 1

                    market_cap = market_caps.get(symbol)
                    if market_cap:
                        weighted_sum += rs_rating * market_cap
                        weighted_total += market_cap

            except Exception as e:
                logger.debug(f"Error calculating RS for {symbol}: {e}")
                continue

        # Need at least 3 stocks with valid RS to rank the group
        if len(rs_ratings) < 3:
            logger.debug(
                f"Insufficient data for group {group_name}: "
                f"only {len(rs_ratings)} stocks with valid RS"
            )
            return None

        # Calculate average RS
        avg_rs = sum(rs_ratings) / len(rs_ratings)
        median_rs = statistics.median(rs_ratings)
        rs_std_dev = statistics.pstdev(rs_ratings) if len(rs_ratings) > 1 else None
        weighted_avg_rs = (weighted_sum / weighted_total) if weighted_total > 0 else None

        return {
            'industry_group': group_name,
            'date': calculation_date,
            'avg_rs_rating': round(avg_rs, 2),
            'median_rs_rating': round(median_rs, 2),
            'weighted_avg_rs_rating': round(weighted_avg_rs, 2) if weighted_avg_rs is not None else None,
            'rs_std_dev': round(rs_std_dev, 2) if rs_std_dev is not None else None,
            'num_stocks': len(rs_ratings),
            'num_stocks_rs_above_80': rs_above_80_count,
            'top_symbol': top_symbol,
            'top_rs_rating': round(top_rs, 2) if top_rs > 0 else None,
        }

    def calculate_rs_by_symbol_for_dates(
        self,
        prefetch: GroupRankPrefetchData,
        calculation_dates: List[date],
    ) -> Dict[date, Dict[str, float]]:
        """Precompute RS ratings once per symbol/date for group aggregation."""
        ordered_dates = sorted(calculation_dates)
        result: Dict[date, Dict[str, float]] = {calc_date: {} for calc_date in ordered_dates}
        benchmark_close = self._close_series(prefetch.benchmark_prices)
        if benchmark_close is None:
            return result

        periods = self.rs_calculator.PERIODS
        max_period = max(periods.keys())
        benchmark_returns = self._period_returns(benchmark_close, periods.keys())
        benchmark_positions = self._positions_by_date(benchmark_close.index, ordered_dates)

        for symbol, prices in prefetch.prices_by_symbol.items():
            close = self._close_series(prices)
            if close is None:
                continue

            symbol_returns = self._period_returns(close, periods.keys())
            symbol_positions = self._positions_by_date(close.index, ordered_dates)
            for calc_date in ordered_dates:
                latest_idx = symbol_positions.get(calc_date, -1)
                if latest_idx < max_period - 1:
                    continue

                benchmark_idx = benchmark_positions.get(calc_date, -1)
                weighted_performance = 0.0
                total_weight = 0.0
                for period, weight in periods.items():
                    if benchmark_idx < period - 1:
                        continue
                    stock_return = self._series_value_at(symbol_returns[period], latest_idx)
                    benchmark_return = self._series_value_at(benchmark_returns[period], benchmark_idx)
                    if stock_return is None or benchmark_return is None:
                        continue
                    weighted_performance += (stock_return - benchmark_return) * weight
                    total_weight += weight

                if total_weight == 0:
                    continue

                relative_performance = weighted_performance / total_weight
                rs_rating = self._scale_relative_performance_to_rs(relative_performance)
                if rs_rating > 0:
                    result[calc_date][symbol] = rs_rating

        return result

    @staticmethod
    def _close_series(prices: Optional[pd.DataFrame]) -> Optional[pd.Series]:
        if prices is None or prices.empty or "Close" not in prices.columns:
            return None
        close = prices["Close"].sort_index()
        return close

    @staticmethod
    def _period_returns(close: pd.Series, periods) -> Dict[int, pd.Series]:
        return {
            period: close.pct_change(periods=period - 1, fill_method=None)
            for period in periods
        }

    @staticmethod
    def _positions_by_date(index, calculation_dates: List[date]) -> Dict[date, int]:
        positions: Dict[date, int] = {}
        for calc_date in calculation_dates:
            exclusive_end = pd.Timestamp(calc_date) + pd.Timedelta(days=1)
            if isinstance(index, pd.DatetimeIndex) and index.tz is not None and exclusive_end.tz is None:
                exclusive_end = exclusive_end.tz_localize(index.tz)
            positions[calc_date] = index.searchsorted(exclusive_end, side="left") - 1
        return positions

    @staticmethod
    def _series_value_at(series: pd.Series, index: int) -> Optional[float]:
        try:
            value = series.iloc[index]
        except IndexError:
            return None
        if pd.isna(value) or not math.isfinite(float(value)):
            return None
        return float(value)

    @staticmethod
    def _scale_relative_performance_to_rs(relative_performance: float) -> float:
        if relative_performance >= 0:
            rs_rating = min(100, 50 + (relative_performance * 100))
        else:
            rs_rating = max(0, 50 + (relative_performance * 100))
        return round(rs_rating, 2)

    def calculate_group_metrics_from_rs(
        self,
        group_name: str,
        symbols: List[str],
        rs_by_symbol: Dict[str, float],
        market_caps: Dict[str, float],
        calculation_date: date,
    ) -> Optional[Dict]:
        """Aggregate precomputed per-symbol RS ratings into one group metric row."""
        if not symbols:
            logger.debug(f"No validated symbols found for group: {group_name}")
            return None

        rs_ratings = []
        top_symbol = None
        top_rs = -1
        rs_above_80_count = 0
        weighted_sum = 0.0
        weighted_total = 0.0

        for symbol in symbols:
            rs_rating = rs_by_symbol.get(symbol)
            if rs_rating is None or rs_rating <= 0:
                continue

            rs_ratings.append(rs_rating)
            if rs_rating > top_rs:
                top_rs = rs_rating
                top_symbol = symbol
            if rs_rating >= 80:
                rs_above_80_count += 1

            market_cap = market_caps.get(symbol)
            if market_cap:
                weighted_sum += rs_rating * market_cap
                weighted_total += market_cap

        if len(rs_ratings) < 3:
            logger.debug(
                f"Insufficient data for group {group_name}: "
                f"only {len(rs_ratings)} stocks with valid RS"
            )
            return None

        avg_rs = sum(rs_ratings) / len(rs_ratings)
        median_rs = statistics.median(rs_ratings)
        rs_std_dev = statistics.pstdev(rs_ratings) if len(rs_ratings) > 1 else None
        weighted_avg_rs = (weighted_sum / weighted_total) if weighted_total > 0 else None

        return {
            'industry_group': group_name,
            'date': calculation_date,
            'avg_rs_rating': round(avg_rs, 2),
            'median_rs_rating': round(median_rs, 2),
            'weighted_avg_rs_rating': round(weighted_avg_rs, 2) if weighted_avg_rs is not None else None,
            'rs_std_dev': round(rs_std_dev, 2) if rs_std_dev is not None else None,
            'num_stocks': len(rs_ratings),
            'num_stocks_rs_above_80': rs_above_80_count,
            'top_symbol': top_symbol,
            'top_rs_rating': round(top_rs, 2) if top_rs > 0 else None,
        }
