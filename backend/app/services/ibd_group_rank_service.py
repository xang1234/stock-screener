"""
Service for calculating and managing IBD Industry Group Rankings.

Calculates daily rankings based on average RS rating of constituent stocks.
"""
import logging
import statistics
from typing import Any, Optional, Dict, List
from datetime import datetime, date, timedelta
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import and_

from ..models.industry import IBDGroupRank
from ..domain.providers.price_symbol_support import is_unsupported_yahoo_price_symbol
from .group_constituent_source import GroupConstituentSource
from .group_detail_payloads import constituent_stock_payloads_from_scan_items
from .group_ranking_history import build_group_detail_payload_from_parts
from .group_rank_cache_policy import GroupRankCacheRequirement
from .group_rank_models import (
    GroupRankCalculationResult,
    GroupRankPrefetchData,
    GroupRankPrefetchStats,
)
from .group_rank_input_loader import GroupRankInputLoader
from .group_rank_historical_calculator import (
    GroupRankHistoricalCalculator,
)
from .group_rank_legacy_adapter import LegacyGroupRankPrefetchAdapter
from .group_ranking_calculator import GroupRankingCalculator
from .group_ranking_repository import GroupRankingRepository
from .derived_data_execution_policy import DerivedDataExecutionPolicy
from .ibd_industry_service import IBDIndustryService
from .price_cache_service import PriceCacheService
from .benchmark_cache_service import BenchmarkCacheService
from ..scanners.criteria.relative_strength import RelativeStrengthCalculator

logger = logging.getLogger(__name__)
GROUP_RANK_CHANGE_CALENDAR_DAYS = {
    "1w": 7,
    "1m": 30,
    "3m": 90,
    "6m": 180,
}


class IncompleteGroupRankingCacheError(RuntimeError):
    """Raised when a cache-only same-day group ranking run lacks required inputs."""

    def __init__(self, stats: GroupRankPrefetchStats):
        self.stats = stats
        market_suffix = f" for {stats.market}" if stats.market else ""
        reason = (
            f"{stats.benchmark_symbol} benchmark data is missing from cache"
            f"{market_suffix}"
        )
        if stats.benchmark_available:
            reason = (
                f"{stats.cache_miss_symbols} symbols are missing cached price data"
            )
        super().__init__(reason)


class MissingIBDIndustryMappingsError(RuntimeError):
    """Raised when tracked IBD industry mappings have not been loaded."""

    def __init__(self) -> None:
        super().__init__("IBD industry mappings are not loaded")


class IBDGroupRankService:
    """
    Service for calculating and managing IBD Industry Group Rankings.

    Methodology:
    1. For each IBD industry group, get all constituent stock symbols
    2. Calculate RS rating for each stock using RelativeStrengthCalculator
    3. Compute average RS for the group (excluding stocks with insufficient data)
    4. Rank all groups from highest to lowest avg_rs
    5. Store daily snapshot in IBDGroupRank table
    """

    def __init__(
        self,
        *,
        price_cache: PriceCacheService,
        benchmark_cache: BenchmarkCacheService,
        rs_calculator: RelativeStrengthCalculator | None = None,
        group_constituent_source: GroupConstituentSource | None = None,
        input_loader: GroupRankInputLoader,
        ranking_calculator: GroupRankingCalculator | None = None,
        ranking_repository: GroupRankingRepository | None = None,
        historical_calculator: GroupRankHistoricalCalculator,
        legacy_prefetch_adapter: (
            LegacyGroupRankPrefetchAdapter | None
        ) = None,
    ):
        """Initialize the service."""
        self.price_cache = price_cache
        self.benchmark_cache = benchmark_cache
        self.rs_calculator = rs_calculator or RelativeStrengthCalculator()
        self.group_constituent_source = (
            group_constituent_source or GroupConstituentSource()
        )
        self.input_loader = input_loader
        self.ranking_calculator = (
            ranking_calculator
            or GroupRankingCalculator(self.rs_calculator)
        )
        self.ranking_repository = (
            ranking_repository or GroupRankingRepository()
        )
        self.historical_calculator = historical_calculator
        self.legacy_prefetch_adapter = (
            legacy_prefetch_adapter
            or LegacyGroupRankPrefetchAdapter()
        )

    def calculate_group_rankings(
        self,
        db: Session,
        calculation_date: date = None,
        *,
        market: str | None = None,
        policy: DerivedDataExecutionPolicy = (
            DerivedDataExecutionPolicy.provider_allowed()
        ),
        cache_requirement: GroupRankCacheRequirement = GroupRankCacheRequirement.disabled(),
    ) -> GroupRankCalculationResult:
        """
        Calculate and store rankings for all IBD groups for a given date.

        Args:
            db: Database session
            calculation_date: Date to calculate rankings for (default: today)

        Returns:
            List of ranked groups with metrics
        """
        if calculation_date is None:
            calculation_date = datetime.now().date()

        normalized_market = (market or "US").upper()
        logger.info(
            "Calculating industry group rankings for market=%s date=%s",
            normalized_market, calculation_date,
        )
        start_time = datetime.now()

        # Get all unique industry groups for this market
        all_groups = self.input_loader.taxonomy_source.groups(
            db,
            normalized_market,
        )
        if not all_groups:
            logger.error("No industry groups found for market %s", normalized_market)
            raise MissingIBDIndustryMappingsError()

        logger.info(
            "Found %d industry groups for market %s", len(all_groups), normalized_market,
        )

        # Pre-fetch all data upfront (benchmark + all stock prices).
        prefetch = self._coerce_prefetch_data(
            self._prefetch_all_data(
                db,
                market=normalized_market,
                policy=policy,
            )
        )
        prefetch_stats = prefetch.stats.with_cache_requirement(
            cache_requirement
        )

        if cache_requirement.enabled:
            if not prefetch_stats.benchmark_available:
                raise IncompleteGroupRankingCacheError(prefetch_stats)
            if (
                prefetch_stats.cache_coverage_ratio
                < cache_requirement.min_coverage
            ):
                raise IncompleteGroupRankingCacheError(prefetch_stats)
            if prefetch_stats.cache_miss_symbols > 0:
                logger.warning(
                    "Cache-only group ranking run has %d cache misses out of %d symbols "
                    "(coverage %.1f%% >= %.1f%%)",
                    prefetch_stats.cache_miss_symbols,
                    prefetch_stats.target_symbols,
                    prefetch_stats.cache_coverage_ratio * 100,
                    cache_requirement.min_coverage * 100,
                )

        if prefetch.benchmark_prices is None or prefetch.benchmark_prices.empty:
            logger.error(
                "Failed to get benchmark data for market %s", normalized_market,
            )
            return GroupRankCalculationResult(
                rankings=(),
                prefetch_stats=prefetch_stats,
            )

        prefetch = self.input_loader.complete_legacy_symbols(
            db,
            market=normalized_market,
            group_names=all_groups,
            prefetch=prefetch,
        )
        group_metrics = list(
            self.ranking_calculator.calculate_for_date(
                prefetch=prefetch,
                group_names=all_groups,
                calculation_date=calculation_date,
            )
        )

        if not group_metrics:
            logger.error("No valid group metrics calculated")
            return GroupRankCalculationResult(
                rankings=(),
                prefetch_stats=prefetch_stats,
            )

        # Store in database
        self.ranking_repository.store_rankings(
            db,
            calculation_date=calculation_date,
            rankings=group_metrics,
            market=normalized_market,
        )
        db.commit()

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Calculated rankings for {len(group_metrics)} groups in {duration:.1f}s"
        )

        return GroupRankCalculationResult(
            rankings=tuple(group_metrics),
            prefetch_stats=prefetch_stats,
        )

    def _calculate_group_rs(
        self,
        db: Session,
        group_name: str,
        spy_prices: pd.Series,
        calculation_date: date
    ) -> Optional[Dict]:
        """
        Calculate average RS rating for a single industry group.

        Args:
            db: Database session
            group_name: IBD industry group name
            spy_prices: SPY price series (most recent first)
            calculation_date: Date for calculation

        Returns:
            Dict with group metrics or None if insufficient data
        """
        # Get all symbols in this group
        symbols = IBDIndustryService.get_group_symbols(db, group_name)

        if not symbols:
            logger.debug(f"No symbols found for group: {group_name}")
            return None

        # Fetch prices for all symbols in batch
        prices_dict = self.price_cache.get_many(symbols, period="2y")

        # Market caps for weighted RS
        market_caps = self.input_loader.market_cap_source.market_caps(
            db,
            symbols,
        )

        # Calculate RS for each symbol
        rs_ratings = []
        top_symbol = None
        top_rs = -1
        rs_above_80_count = 0
        weighted_sum = 0.0
        weighted_total = 0.0

        for symbol in symbols:
            prices = prices_dict.get(symbol)
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

    def get_current_rankings(
        self,
        db: Session,
        limit: int = 197,
        calculation_date: date | None = None,
        *,
        market: str = "US",
    ) -> List[Dict]:
        """
        Get most recent group rankings with rank changes for one market.

        Rankings are partitioned by market — each market maintains its own
        rank ordering (both US and HK can have a rank #1), so every query
        must filter by market to avoid mixing them.
        """
        normalized_market = (market or "US").upper()
        rankings = self.ranking_repository.current_rank_rows(
            db,
            limit=limit,
            market=normalized_market,
            calculation_date=calculation_date,
        )

        if not rankings:
            return []
        latest_date = rankings[0].date

        # Calendar-day target offsets for rank changes. The lookup below picks
        # the closest stored ranking within a small window; these are not exact
        # trading-session offsets.
        period_days = dict(GROUP_RANK_CHANGE_CALENDAR_DAYS)

        # Batch fetch all historical ranks in ONE query instead of 197*4=788 queries
        group_names = [r.industry_group for r in rankings]
        historical_ranks = self.ranking_repository.historical_ranks_batch(
            db,
            group_names=group_names,
            current_date=latest_date,
            period_days=period_days,
            market=normalized_market,
        )

        # Build result with rank changes
        result = []
        for ranking in rankings:
            pct_above_80 = self.ranking_calculator._calculate_pct_above_80(
                ranking.num_stocks_rs_above_80, ranking.num_stocks
            )
            item = self._rank_record_payload(
                ranking,
                pct_rs_above_80=pct_above_80,
            )

            # Get pre-computed historical ranks from batch lookup
            for period_name in period_days.keys():
                historical_rank = historical_ranks.get(
                    (ranking.industry_group, period_name)
                )
                if historical_rank is not None:
                    # Positive change means rank improved (moved up)
                    item[f'rank_change_{period_name}'] = historical_rank - ranking.rank
                else:
                    item[f'rank_change_{period_name}'] = None

            result.append(item)

        self._annotate_top_symbol_names(db, result)
        return result

    @staticmethod
    def _rank_record_payload(
        ranking: IBDGroupRank,
        *,
        pct_rs_above_80: float | None,
        top_symbol_name: str | None = None,
    ) -> Dict:
        return {
            'industry_group': ranking.industry_group,
            'date': ranking.date.isoformat(),
            'rank': ranking.rank,
            'avg_rs_rating': ranking.avg_rs_rating,
            'median_rs_rating': ranking.median_rs_rating,
            'weighted_avg_rs_rating': ranking.weighted_avg_rs_rating,
            'rs_std_dev': ranking.rs_std_dev,
            'num_stocks': ranking.num_stocks,
            'num_stocks_rs_above_80': ranking.num_stocks_rs_above_80,
            'pct_rs_above_80': pct_rs_above_80,
            'top_symbol': ranking.top_symbol,
            'top_symbol_name': top_symbol_name,
            'top_rs_rating': ranking.top_rs_rating,
            'rank_change_1w': None,
            'rank_change_1m': None,
            'rank_change_3m': None,
            'rank_change_6m': None,
        }

    def _annotate_top_symbol_names(self, db: Session, rows: List[Dict]) -> None:
        name_map = self._get_symbol_name_map(
            db,
            [row.get("top_symbol") for row in rows],
        )
        for row in rows:
            row["top_symbol_name"] = name_map.get(row.get("top_symbol"))

    def _get_symbol_name_map(
        self,
        db: Session,
        symbols: List[str | None],
    ) -> Dict[str, str | None]:
        normalized_symbols = {
            str(symbol).strip()
            for symbol in symbols
            if str(symbol or "").strip()
        }
        if not normalized_symbols:
            return {}
        return self.input_loader.universe_source.symbol_names(
            db,
            sorted(normalized_symbols),
        )

    def get_historical_ranks_batch(
        self,
        db: Session,
        group_names: List[str],
        current_date: date,
        period_days: Dict[str, int],
        *,
        market: str = "US",
    ) -> Dict[tuple, int]:
        """Return closest historical ranks for each group and period."""
        return self.ranking_repository.historical_ranks_batch(
            db,
            group_names=group_names,
            current_date=current_date,
            period_days=period_days,
            market=market,
        )

    def get_group_history(
        self,
        db: Session,
        industry_group: str,
        days: int = 180,
        *,
        market: str = "US",
    ) -> Dict:
        """Get historical ranking data for a specific group, scoped by market."""
        normalized_market = (market or "US").upper()
        cutoff_date = datetime.now().date() - timedelta(days=days)

        records = self.ranking_repository.group_rank_rows(
            db,
            industry_group=industry_group,
            start_date=cutoff_date,
            market=normalized_market,
        )

        if not records:
            return {'industry_group': industry_group, 'history': []}

        # Current (most recent) data
        current = records[0]

        # Get rank changes using calendar-day target offsets with closest-record matching.
        period_days = dict(GROUP_RANK_CHANGE_CALENDAR_DAYS)
        rank_changes = {}

        historical_ranks = self.ranking_repository.historical_ranks_batch(
            db,
            group_names=[industry_group],
            current_date=current.date,
            period_days=period_days,
            market=normalized_market,
        )
        for period_name in period_days:
            historical_rank = historical_ranks.get((industry_group, period_name))
            if historical_rank is not None:
                rank_changes[f'rank_change_{period_name}'] = historical_rank - current.rank
            else:
                rank_changes[f'rank_change_{period_name}'] = None

        # Build history data points
        history = [
            {
                'date': r.date.isoformat(),
                'rank': r.rank,
                'avg_rs_rating': r.avg_rs_rating,
                'num_stocks': r.num_stocks,
            }
            for r in records
        ]

        # Get constituent stocks with metrics
        stocks = self._get_constituent_stocks(
            db,
            industry_group,
            market=normalized_market,
            as_of_date=current.date,
        )

        pct_above_80 = self.ranking_calculator._calculate_pct_above_80(
            current.num_stocks_rs_above_80, current.num_stocks
        )

        top_symbol_name = self._get_symbol_name_map(db, [current.top_symbol]).get(
            current.top_symbol
        )
        ranking_payload = self._rank_record_payload(
            current,
            pct_rs_above_80=pct_above_80,
            top_symbol_name=top_symbol_name,
        )
        ranking_payload.update(rank_changes)

        return build_group_detail_payload_from_parts(
            industry_group,
            ranking=ranking_payload,
            history=history,
            stocks=stocks,
        )

    def _get_constituent_stocks(
        self,
        db: Session,
        industry_group: str,
        *,
        market: str = "US",
        as_of_date: date | None = None,
    ) -> List[Dict]:
        """
        Get constituent stocks for an industry group with their metrics.

        Fetches from the most recent completed scan for the requested market.

        Args:
            db: Database session
            industry_group: IBD industry group name

        Returns:
            List of stocks with RS, earnings, and sales metrics
        """
        items = self.group_constituent_source.get_constituent_items(
            db,
            industry_group,
            market=market,
            as_of_date=as_of_date,
        )
        return constituent_stock_payloads_from_scan_items(items)

    def get_rank_movers(
        self,
        db: Session,
        period: str = '1w',
        limit: int = 20,
        calculation_date: date | None = None,
        *,
        market: str = "US",
    ) -> Dict:
        """Get groups with biggest rank changes over a period, scoped by market."""
        current_rankings = self.get_current_rankings(
            db,
            limit=197,
            calculation_date=calculation_date,
            market=market,
        )

        if not current_rankings:
            return {'period': period, 'gainers': [], 'losers': []}

        # Filter to groups with rank change data for this period
        change_key = f'rank_change_{period}'
        groups_with_change = [
            r for r in current_rankings
            if r.get(change_key) is not None
        ]

        # Split by sign so a market with only positive (or only negative) movers
        # never reports the smallest gainers as "losers" or vice versa.
        gainers = [r for r in groups_with_change if r[change_key] > 0]
        losers = [r for r in groups_with_change if r[change_key] < 0]
        gainers.sort(key=lambda r: r[change_key], reverse=True)
        losers.sort(key=lambda r: r[change_key])

        return {
            'period': period,
            'gainers': gainers[:limit],
            'losers': losers[:limit],
        }

    def _get_validated_group_symbols(
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
        group_symbols = self.input_loader.taxonomy_source.symbols_for_group(
            db,
            group_name,
            market,
        )
        return [
            symbol
            for symbol in group_symbols
            if symbol in active_symbols and not is_unsupported_yahoo_price_symbol(symbol)
        ]

    def _get_market_caps_for_symbols(
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

        return self.input_loader.market_cap_source.market_caps(
            db,
            symbols,
        )

    def _coerce_prefetch_data(self, prefetch: Any) -> GroupRankPrefetchData:
        """Adapt the legacy facade seam to the strict prefetch model."""
        return self.legacy_prefetch_adapter.adapt(prefetch)

    def _prefetch_all_data(
        self,
        db: Session,
        *,
        market: str | None = None,
        policy: DerivedDataExecutionPolicy = (
            DerivedDataExecutionPolicy.provider_allowed()
        ),
    ) -> GroupRankPrefetchData:
        return self.input_loader.load(
            db,
            market=(market or "US").upper(),
            policy=policy,
        )

    def backfill_rankings_optimized(
        self,
        db: Session,
        start_date: date,
        end_date: date,
        *,
        market: str = "US",
    ) -> Dict:
        return self.historical_calculator.backfill_rankings_optimized(
            db,
            start_date=start_date,
            end_date=end_date,
            market=market,
        )

    def backfill_rankings(
        self,
        db: Session,
        start_date: date,
        end_date: date,
        *,
        market: str = "US",
    ) -> Dict:
        return self.historical_calculator.backfill_rankings(
            db,
            start_date,
            end_date,
            market=market,
        )

    def find_missing_dates(
        self,
        db: Session,
        lookback_days: int = 365,
        *,
        market: str = "US",
        end_date: date | None = None,
    ) -> List[date]:
        return self.historical_calculator.find_missing_dates(
            db,
            lookback_days,
            market=market,
            end_date=end_date,
        )

    def fill_gaps(
        self,
        db: Session,
        missing_dates: List[date],
        *,
        market: str = "US",
    ) -> Dict:
        return self.historical_calculator.fill_gaps(
            db,
            missing_dates,
            market=market,
        )

    def fill_gaps_optimized(
        self,
        db: Session,
        missing_dates: List[date],
        *,
        market: str = "US",
        policy: DerivedDataExecutionPolicy = (
            DerivedDataExecutionPolicy.provider_allowed()
        ),
    ) -> Dict:
        return self.historical_calculator.fill_gaps_optimized(
            db,
            missing_dates,
            market=market,
            policy=policy,
        )

    def backfill_rankings_chunked(
        self,
        db: Session,
        start_date: date,
        end_date: date,
        chunk_size_days: int = 30,
        *,
        market: str = "US",
    ) -> Dict:
        return self.historical_calculator.backfill_rankings_chunked(
            db,
            start_date,
            end_date,
            chunk_size_days,
            market=market,
        )
