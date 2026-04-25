"""
Service for calculating and managing IBD Industry Group Rankings.

Calculates daily rankings based on average RS rating of constituent stocks.
"""
import logging
import statistics
from typing import Optional, Dict, List
from datetime import datetime, date, timedelta
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc

from ..models.industry import IBDGroupRank
from ..models.stock_universe import StockUniverse
from ..models.scan_result import Scan, ScanResult
from ..utils.symbol_support import is_unsupported_yahoo_price_symbol
from .ibd_industry_service import IBDIndustryService
from .price_cache_service import PriceCacheService
from .benchmark_cache_service import BenchmarkCacheService
from ..scanners.criteria.relative_strength import RelativeStrengthCalculator

logger = logging.getLogger(__name__)
CACHE_MISS_TOLERANCE_RATIO = 0.05  # Allow up to 5% cache misses in cache-only group ranking runs


class IncompleteGroupRankingCacheError(RuntimeError):
    """Raised when a cache-only same-day group ranking run lacks required inputs."""

    def __init__(self, stats: Dict[str, int | bool]):
        self.stats = stats
        reason = "SPY benchmark data is missing from cache"
        if stats.get("spy_cached"):
            reason = (
                f"{stats.get('cache_miss_symbols', 0)} symbols are missing cached price data"
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
    ):
        """Initialize the service."""
        self.price_cache = price_cache
        self.benchmark_cache = benchmark_cache
        self.rs_calculator = rs_calculator or RelativeStrengthCalculator()

    def calculate_group_rankings(
        self,
        db: Session,
        calculation_date: date = None,
        *,
        market: str | None = None,
        cache_only: bool = False,
        require_complete_cache: bool = False,
    ) -> List[Dict]:
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
        all_groups = IBDIndustryService.get_all_groups(db, market=normalized_market)
        if not all_groups:
            logger.error("No industry groups found for market %s", normalized_market)
            raise MissingIBDIndustryMappingsError()

        logger.info(
            "Found %d industry groups for market %s", len(all_groups), normalized_market,
        )

        # Pre-fetch all data upfront (benchmark + all stock prices).
        spy_data, all_prices, active_symbols, market_caps, prefetch_stats = (
            self._prefetch_all_data(
                db,
                market=normalized_market,
                cache_only=cache_only,
            )
        )

        if require_complete_cache:
            if not prefetch_stats.get("spy_cached"):
                raise IncompleteGroupRankingCacheError(prefetch_stats)
            cache_miss_symbols = prefetch_stats.get("cache_miss_symbols", 0)
            target_symbols = prefetch_stats.get("target_symbols", 0)
            miss_ratio = cache_miss_symbols / target_symbols if target_symbols > 0 else 0.0
            if miss_ratio > CACHE_MISS_TOLERANCE_RATIO:
                raise IncompleteGroupRankingCacheError(prefetch_stats)
            if cache_miss_symbols > 0:
                logger.warning(
                    "Cache-only group ranking run has %d cache misses out of %d symbols (%.1f%%) -- within tolerance",
                    cache_miss_symbols, target_symbols, miss_ratio * 100,
                )

        if spy_data is None or spy_data.empty:
            logger.error(
                "Failed to get benchmark data for market %s", normalized_market,
            )
            return []

        # Filter benchmark data to calculation_date for accurate historical rankings
        if calculation_date < datetime.now().date():
            spy_data_filtered = spy_data[spy_data.index.date <= calculation_date]
        else:
            spy_data_filtered = spy_data

        # Prepare benchmark prices series (most recent first for RS calculator)
        spy_prices = spy_data_filtered['Close'].sort_index(ascending=False)

        # Calculate RS for each group using pre-fetched cached data
        group_metrics = []

        for group_name in all_groups:
            try:
                metrics = self._calculate_group_rs_from_cache(
                    db, group_name, spy_prices, all_prices, active_symbols, market_caps, calculation_date,
                    market=normalized_market,
                )
                if metrics:
                    group_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Error calculating RS for group {group_name}: {e}")
                continue

        if not group_metrics:
            logger.error("No valid group metrics calculated")
            return []

        # Sort by avg_rs (descending) and assign ranks
        group_metrics.sort(key=lambda x: x['avg_rs_rating'], reverse=True)

        for rank, metrics in enumerate(group_metrics, start=1):
            metrics['rank'] = rank

        # Store in database
        self._store_rankings(db, calculation_date, group_metrics, market=normalized_market)

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Calculated rankings for {len(group_metrics)} groups in {duration:.1f}s"
        )

        return group_metrics

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
        market_caps = self._get_market_caps_for_symbols(db, symbols)

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

    def _store_rankings(
        self,
        db: Session,
        calculation_date: date,
        group_metrics: List[Dict],
        *,
        market: str = "US",
    ) -> None:
        """
        Store group rankings in database.

        Upserts records (updates if exists, inserts if new). Unique key is
        ``(industry_group, date, market)``.
        """
        try:
            for metrics in group_metrics:
                existing = db.query(IBDGroupRank).filter(
                    and_(
                        IBDGroupRank.industry_group == metrics['industry_group'],
                        IBDGroupRank.date == calculation_date,
                        IBDGroupRank.market == market,
                    )
                ).first()

                if existing:
                    existing.rank = metrics['rank']
                    existing.avg_rs_rating = metrics['avg_rs_rating']
                    existing.num_stocks = metrics['num_stocks']
                    existing.num_stocks_rs_above_80 = metrics['num_stocks_rs_above_80']
                    existing.top_symbol = metrics['top_symbol']
                    existing.top_rs_rating = metrics['top_rs_rating']
                    existing.median_rs_rating = metrics.get('median_rs_rating')
                    existing.weighted_avg_rs_rating = metrics.get('weighted_avg_rs_rating')
                    existing.rs_std_dev = metrics.get('rs_std_dev')
                else:
                    record = IBDGroupRank(
                        market=market,
                        industry_group=metrics['industry_group'],
                        date=calculation_date,
                        rank=metrics['rank'],
                        avg_rs_rating=metrics['avg_rs_rating'],
                        median_rs_rating=metrics.get('median_rs_rating'),
                        weighted_avg_rs_rating=metrics.get('weighted_avg_rs_rating'),
                        rs_std_dev=metrics.get('rs_std_dev'),
                        num_stocks=metrics['num_stocks'],
                        num_stocks_rs_above_80=metrics['num_stocks_rs_above_80'],
                        top_symbol=metrics['top_symbol'],
                        top_rs_rating=metrics['top_rs_rating'],
                    )
                    db.add(record)

            db.commit()
            logger.info(
                "Stored %d group rankings for market=%s date=%s",
                len(group_metrics), market, calculation_date,
            )

        except Exception as e:
            logger.error(f"Error storing rankings: {e}", exc_info=True)
            db.rollback()
            raise

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
        if calculation_date is not None:
            latest_date = calculation_date
        else:
            # Get most recent date with rankings for THIS market
            latest_record = db.query(IBDGroupRank).filter(
                IBDGroupRank.market == normalized_market,
            ).order_by(
                desc(IBDGroupRank.date)
            ).first()

            if not latest_record:
                return []

            latest_date = latest_record.date

        # Get all rankings for latest date + market
        rankings = db.query(IBDGroupRank).filter(
            IBDGroupRank.date == latest_date,
            IBDGroupRank.market == normalized_market,
        ).order_by(IBDGroupRank.rank).limit(limit).all()

        if not rankings:
            return []

        # Get historical dates for rank changes
        period_days = {
            '1w': 5,    # 5 trading days
            '1m': 21,   # ~1 month
            '3m': 63,   # ~3 months
            '6m': 126,  # ~6 months
        }

        # Batch fetch all historical ranks in ONE query instead of 197*4=788 queries
        group_names = [r.industry_group for r in rankings]
        historical_ranks = self._get_historical_ranks_batch(
            db, group_names, latest_date, period_days,
            market=normalized_market,
        )

        # Build result with rank changes
        result = []
        for ranking in rankings:
            pct_above_80 = self._calculate_pct_above_80(
                ranking.num_stocks_rs_above_80, ranking.num_stocks
            )
            item = {
                'industry_group': ranking.industry_group,
                'date': ranking.date.isoformat(),
                'rank': ranking.rank,
                'avg_rs_rating': ranking.avg_rs_rating,
                'median_rs_rating': ranking.median_rs_rating,
                'weighted_avg_rs_rating': ranking.weighted_avg_rs_rating,
                'rs_std_dev': ranking.rs_std_dev,
                'num_stocks': ranking.num_stocks,
                'num_stocks_rs_above_80': ranking.num_stocks_rs_above_80,
                'pct_rs_above_80': pct_above_80,
                'top_symbol': ranking.top_symbol,
                'top_rs_rating': ranking.top_rs_rating,
            }

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
    def _annotate_top_symbol_names(db: Session, rows: List[Dict]) -> None:
        name_map = IBDGroupRankService._get_symbol_name_map(
            db,
            [row.get("top_symbol") for row in rows],
        )
        for row in rows:
            row["top_symbol_name"] = name_map.get(row.get("top_symbol"))

    @staticmethod
    def _get_symbol_name_map(db: Session, symbols: List[str | None]) -> Dict[str, str | None]:
        normalized_symbols = {
            str(symbol).strip()
            for symbol in symbols
            if str(symbol or "").strip()
        }
        if not normalized_symbols:
            return {}
        return dict(
            db.query(StockUniverse.symbol, StockUniverse.name)
            .filter(StockUniverse.symbol.in_(normalized_symbols))
            .all()
        )

    def _get_historical_ranks_batch(
        self,
        db: Session,
        group_names: List[str],
        current_date: date,
        period_days: Dict[str, int],
        *,
        market: str = "US",
    ) -> Dict[tuple, int]:
        """Batch fetch historical ranks for all groups and periods in ONE query.

        Scoped by ``market`` so multi-market data doesn't cross-contaminate
        rank-change calculations.
        """
        if not group_names or not period_days:
            return {}

        # Calculate date range covering all periods (max period + 7-day buffer)
        max_days = max(period_days.values())
        earliest_date = current_date - timedelta(days=max_days + 7)

        # Single query: fetch ALL historical records within the date range
        all_records = db.query(
            IBDGroupRank.industry_group,
            IBDGroupRank.date,
            IBDGroupRank.rank
        ).filter(
            and_(
                IBDGroupRank.industry_group.in_(group_names),
                IBDGroupRank.date >= earliest_date,
                IBDGroupRank.date < current_date,  # Exclude current date
                IBDGroupRank.market == (market or "US").upper(),
            )
        ).all()

        # Build lookup: group_name -> list of (date, rank)
        group_history = {}
        for record in all_records:
            if record.industry_group not in group_history:
                group_history[record.industry_group] = []
            group_history[record.industry_group].append(
                (record.date, record.rank)
            )

        # For each group and period, find the closest historical rank
        result = {}
        for group_name in group_names:
            history = group_history.get(group_name, [])
            if not history:
                continue

            for period_name, days in period_days.items():
                target_date = current_date - timedelta(days=days)

                # Filter to records within 7-day window of target
                candidates = [
                    (d, r) for d, r in history
                    if abs((d - target_date).days) <= 7
                ]

                if not candidates:
                    continue

                # Pick closest date (prefer earlier if tied)
                def _distance_key(item):
                    d, _ = item
                    delta = d - target_date
                    return (abs(delta.days), delta.days > 0)

                closest_date, closest_rank = min(candidates, key=_distance_key)
                result[(group_name, period_name)] = closest_rank

        return result

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

        records = db.query(IBDGroupRank).filter(
            and_(
                IBDGroupRank.industry_group == industry_group,
                IBDGroupRank.date >= cutoff_date,
                IBDGroupRank.market == normalized_market,
            )
        ).order_by(IBDGroupRank.date.desc()).all()

        if not records:
            return {'industry_group': industry_group, 'history': []}

        # Current (most recent) data
        current = records[0]

        # Get rank changes
        period_days = {'1w': 5, '1m': 21, '3m': 63, '6m': 126}
        rank_changes = {}

        historical_ranks = self._get_historical_ranks_batch(
            db, [industry_group], current.date, period_days,
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
        stocks = self._get_constituent_stocks(db, industry_group)

        pct_above_80 = self._calculate_pct_above_80(
            current.num_stocks_rs_above_80, current.num_stocks
        )

        top_symbol_name = self._get_symbol_name_map(db, [current.top_symbol]).get(
            current.top_symbol
        )

        return {
            'industry_group': industry_group,
            'current_rank': current.rank,
            'current_avg_rs': current.avg_rs_rating,
            'current_median_rs': current.median_rs_rating,
            'current_weighted_avg_rs': current.weighted_avg_rs_rating,
            'current_rs_std_dev': current.rs_std_dev,
            'num_stocks': current.num_stocks,
            'pct_rs_above_80': pct_above_80,
            'top_symbol': current.top_symbol,
            'top_symbol_name': top_symbol_name,
            'top_rs_rating': current.top_rs_rating,
            **rank_changes,
            'history': history,
            'stocks': stocks,
        }

    def _get_constituent_stocks(
        self,
        db: Session,
        industry_group: str
    ) -> List[Dict]:
        """
        Get constituent stocks for an industry group with their metrics.

        Fetches from the most recent completed scan results.

        Args:
            db: Database session
            industry_group: IBD industry group name

        Returns:
            List of stocks with RS, earnings, and sales metrics
        """
        try:
            # Get the most recent completed scan
            latest_scan = db.query(Scan).filter(
                Scan.status == 'completed'
            ).order_by(desc(Scan.completed_at)).first()

            if not latest_scan:
                logger.warning("No completed scans found for constituent stocks")
                return []

            results = (
                db.query(ScanResult, StockUniverse.name.label("company_name"))
                .outerjoin(StockUniverse, StockUniverse.symbol == ScanResult.symbol)
                .filter(
                    and_(
                        ScanResult.scan_id == latest_scan.scan_id,
                        ScanResult.ibd_industry_group == industry_group,
                    )
                )
                .order_by(desc(ScanResult.rs_rating))
                .all()
            )

            stocks = []
            for r, company_name in results:
                stocks.append({
                    'symbol': r.symbol,
                    'company_name': company_name,
                    'price': r.price,
                    'rs_rating': r.rs_rating,
                    'rs_rating_1m': r.rs_rating_1m,
                    'rs_rating_3m': r.rs_rating_3m,
                    'rs_rating_12m': r.rs_rating_12m,
                    'eps_growth_qq': r.eps_growth_qq,
                    'eps_growth_yy': r.eps_growth_yy,
                    'sales_growth_qq': r.sales_growth_qq,
                    'sales_growth_yy': r.sales_growth_yy,
                    'composite_score': r.composite_score,
                    'stage': r.stage,
                    'price_sparkline_data': r.price_sparkline_data,
                    'price_trend': r.price_trend,
                    'price_change_1d': r.price_change_1d,
                    'rs_sparkline_data': r.rs_sparkline_data,
                    'rs_trend': r.rs_trend,
                })

            logger.info(f"Found {len(stocks)} stocks for group {industry_group}")
            return stocks

        except Exception as e:
            logger.error(f"Error fetching constituent stocks: {e}", exc_info=True)
            return []

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
        period_days_map = {
            '1w': 5,
            '1m': 21,
            '3m': 63,
            '6m': 126,
        }

        days = period_days_map.get(period, 5)

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

        # Sort by rank change
        # Positive change = rank improved (moved up), so sort descending for gainers
        groups_with_change.sort(key=lambda x: x[change_key], reverse=True)

        gainers = groups_with_change[:limit]
        losers = groups_with_change[-limit:][::-1]  # Reverse to show biggest losers first

        return {
            'period': period,
            'gainers': gainers,
            'losers': losers,
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
        group_symbols = IBDIndustryService.get_group_symbols(
            db, group_name, market=market
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

        rows = db.query(StockUniverse.symbol, StockUniverse.market_cap).filter(
            StockUniverse.symbol.in_(symbols)
        ).all()

        return {symbol: cap for symbol, cap in rows if cap and cap > 0}

    def _calculate_pct_above_80(
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

    def _prefetch_all_data(
        self,
        db: Session,
        *,
        market: str | None = None,
        cache_only: bool = False,
    ) -> tuple:
        """
        Pre-fetch all data needed for backfill in one batch.

        This optimization fetches:
        1. SPY benchmark data once
        2. Active symbols from stock_universe (same source as bulk scans)
        3. All prices for all symbols across all groups in a single batch

        Returns:
            Tuple of (spy_prices_df, {symbol: prices_df}, active_symbols_set, {symbol: market_cap})
        """
        from ..wiring.bootstrap import get_stock_universe_service

        normalized_market = (market or "US").upper()
        benchmark_symbol = self.benchmark_cache.get_benchmark_symbol(normalized_market)
        logger.info(
            "Pre-fetching data for market=%s (benchmark=%s)...",
            normalized_market, benchmark_symbol,
        )

        # 1. Get benchmark data once (SPY for US, HSI/N225/TAIEX/NIFTY for Asia)
        if cache_only:
            spy_data = self.price_cache.get_cached_only_fresh(benchmark_symbol, period="2y")
        else:
            spy_data = self.benchmark_cache.get_benchmark_data(
                market=normalized_market, period="2y",
            )
        if spy_data is None or spy_data.empty:
            logger.error(
                "Failed to get benchmark data (%s) for market %s",
                benchmark_symbol, normalized_market,
            )
            return None, {}, set(), {}, {
                "target_symbols": 0,
                "symbols_with_prices": 0,
                "cache_miss_symbols": 0,
                "spy_cached": False,
            }

        logger.info(f"Fetched benchmark {benchmark_symbol}: {len(spy_data)} days")

        # 2. Get active symbols from stock_universe (same as bulk scans)
        active_symbols_list = get_stock_universe_service().get_active_symbols(
            db,
            market=normalized_market,
        )
        active_symbols = set(active_symbols_list)
        logger.info(f"Found {len(active_symbols)} active symbols in stock_universe")

        # 3. Collect ALL unique symbols across ALL groups for this market
        all_groups = IBDIndustryService.get_all_groups(db, market=normalized_market)
        symbols_to_fetch = set()
        skipped_unsupported_symbols = set()

        for group in all_groups:
            group_symbols = IBDIndustryService.get_group_symbols(db, group, market=normalized_market)
            validated = []
            for symbol in group_symbols:
                if symbol not in active_symbols:
                    continue
                if is_unsupported_yahoo_price_symbol(symbol):
                    skipped_unsupported_symbols.add(symbol)
                    continue
                validated.append(symbol)
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
        market_caps = self._get_market_caps_for_symbols(db, list(symbols_to_fetch))

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

        return spy_data, all_prices, active_symbols, market_caps, stats

    def _delete_rankings_for_range(
        self,
        db: Session,
        start_date: date,
        end_date: date,
        *,
        market: str = "US",
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
            )
        ).delete(synchronize_session=False)

        db.commit()
        logger.info(
            "Deleted %d existing rankings for market=%s %s to %s",
            deleted, normalized_market, start_date, end_date,
        )
        return deleted

    def _calculate_group_rs_from_cache(
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
        symbols = self._get_validated_group_symbols(
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

    def backfill_rankings_optimized(
        self,
        db: Session,
        start_date: date,
        end_date: date
    ) -> Dict:
        """
        Optimized backfill that:
        1. Uses same universe as bulk scans (stock_universe intersection)
        2. Deletes existing rankings and recalculates (no skipping)
        3. Pre-fetches all data once for efficiency

        Args:
            db: Database session
            start_date: Start of backfill range
            end_date: End of backfill range

        Returns:
            Dict with backfill statistics
        """
        logger.info(f"Starting optimized backfill from {start_date} to {end_date}")
        start_time = datetime.now()

        # 1. Delete existing rankings in range
        deleted = self._delete_rankings_for_range(db, start_date, end_date)

        # 2. Pre-fetch ALL data upfront
        spy_data, all_prices, active_symbols, market_caps, _prefetch_stats = self._prefetch_all_data(db)

        if spy_data is None or spy_data.empty:
            logger.error("Cannot proceed without SPY data")
            return {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'deleted': deleted,
                'total_dates': 0,
                'processed': 0,
                'skipped': 0,
                'errors': 1,
                'error': 'Failed to fetch SPY benchmark data',
            }

        all_groups = IBDIndustryService.get_all_groups(db)

        # 3. Generate trading dates (weekdays)
        dates_to_process = []
        current = end_date
        while current >= start_date:
            if current.weekday() < 5:  # Monday=0, Friday=4
                dates_to_process.append(current)
            current -= timedelta(days=1)

        logger.info(
            f"Processing {len(dates_to_process)} trading days with "
            f"{len(all_prices)} symbols across {len(all_groups)} groups"
        )

        processed = 0
        errors = 0

        # 4. Process each date using cached data
        for calc_date in dates_to_process:
            try:
                # Filter SPY to calculation_date
                spy_filtered = spy_data[spy_data.index.date <= calc_date]
                if spy_filtered.empty:
                    logger.warning(f"No SPY data for {calc_date}")
                    errors += 1
                    continue

                spy_prices = spy_filtered['Close'].sort_index(ascending=False)

                # Calculate RS for each group from cache
                group_metrics = []
                for group_name in all_groups:
                    metrics = self._calculate_group_rs_from_cache(
                        db, group_name, spy_prices, all_prices, active_symbols, market_caps, calc_date
                    )
                    if metrics:
                        group_metrics.append(metrics)

                if group_metrics:
                    # Sort and rank
                    group_metrics.sort(key=lambda x: x['avg_rs_rating'], reverse=True)
                    for rank, metrics in enumerate(group_metrics, start=1):
                        metrics['rank'] = rank

                    # Store
                    self._store_rankings(db, calc_date, group_metrics)
                    processed += 1

                    if processed % 10 == 0:
                        logger.info(
                            f"Progress: {processed}/{len(dates_to_process)} dates "
                            f"({len(group_metrics)} groups for {calc_date})"
                        )
                else:
                    errors += 1
                    logger.warning(f"No valid groups for {calc_date}")

            except Exception as e:
                errors += 1
                logger.error(f"Error processing {calc_date}: {e}")

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Optimized backfill complete: {processed} processed, {errors} errors "
            f"in {duration:.1f}s"
        )

        return {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'deleted': deleted,
            'total_dates': len(dates_to_process),
            'processed': processed,
            'skipped': 0,  # No skipping in optimized version
            'errors': errors,
            'duration_seconds': round(duration, 2),
        }

    def backfill_rankings(
        self,
        db: Session,
        start_date: date,
        end_date: date
    ) -> Dict:
        """
        Backfill historical rankings from existing price data.

        Processes each trading day in the range.

        Args:
            db: Database session
            start_date: Start of backfill range
            end_date: End of backfill range

        Returns:
            Dict with backfill statistics
        """
        logger.info(f"Starting backfill from {start_date} to {end_date}")

        # Generate list of dates to process (weekdays only)
        current_date = end_date
        dates_to_process = []

        while current_date >= start_date:
            # Skip weekends
            if current_date.weekday() < 5:  # Monday=0, Friday=4
                dates_to_process.append(current_date)
            current_date -= timedelta(days=1)

        logger.info(f"Processing {len(dates_to_process)} trading days")

        processed = 0
        skipped = 0
        errors = 0

        for calc_date in dates_to_process:
            try:
                # Check if already calculated
                existing = db.query(IBDGroupRank).filter(
                    IBDGroupRank.date == calc_date
                ).first()

                if existing:
                    logger.debug(f"Skipping {calc_date} - already calculated")
                    skipped += 1
                    continue

                # Calculate rankings for this date
                results = self.calculate_group_rankings(db, calc_date)

                if results:
                    processed += 1
                    logger.info(f"Backfilled {calc_date}: {len(results)} groups")
                else:
                    errors += 1
                    logger.warning(f"No results for {calc_date}")

            except Exception as e:
                errors += 1
                logger.error(f"Error processing {calc_date}: {e}")

        logger.info(
            f"Backfill complete: {processed} processed, {skipped} skipped, {errors} errors"
        )

        return {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'total_dates': len(dates_to_process),
            'processed': processed,
            'skipped': skipped,
            'errors': errors,
        }

    def find_missing_dates(
        self,
        db: Session,
        lookback_days: int = 365,
        *,
        market: str = "US",
    ) -> List[date]:
        """Find missing trading dates in the ranking data for one market."""
        from sqlalchemy import func

        normalized_market = (market or "US").upper()
        today = datetime.now().date()
        start_date = today - timedelta(days=lookback_days)

        existing_dates = db.query(
            func.distinct(IBDGroupRank.date)
        ).filter(
            IBDGroupRank.date >= start_date,
            IBDGroupRank.market == normalized_market,
        ).all()

        existing_date_set = {d[0] for d in existing_dates}

        # Generate all weekdays in range
        missing_dates = []
        current_date = start_date

        while current_date < today:  # Exclude today
            # Skip weekends (Saturday=5, Sunday=6)
            if current_date.weekday() < 5:
                if current_date not in existing_date_set:
                    missing_dates.append(current_date)
            current_date += timedelta(days=1)

        logger.info(f"Found {len(missing_dates)} missing dates in last {lookback_days} days")
        return sorted(missing_dates)

    def fill_gaps(
        self,
        db: Session,
        missing_dates: List[date]
    ) -> Dict:
        """
        Fill specific missing dates (used by startup gap-fill).

        Args:
            db: Database session
            missing_dates: List of dates to calculate rankings for

        Returns:
            Statistics about the gap-fill operation
        """
        logger.info(f"Filling {len(missing_dates)} missing dates")

        stats = {
            'total_dates': len(missing_dates),
            'processed': 0,
            'skipped': 0,
            'errors': 0,
        }

        for calc_date in missing_dates:
            try:
                results = self.calculate_group_rankings(db, calc_date)

                if results:
                    stats['processed'] += 1
                    logger.debug(f"Filled gap for {calc_date}: {len(results)} groups")
                else:
                    stats['errors'] += 1
                    logger.warning(f"No results for {calc_date}")

            except Exception as e:
                stats['errors'] += 1
                logger.error(f"Error filling gap for {calc_date}: {e}")

        logger.info(
            f"Gap-fill complete: {stats['processed']} processed, "
            f"{stats['errors']} errors"
        )

        return stats

    def fill_gaps_optimized(
        self,
        db: Session,
        missing_dates: List[date],
        *,
        market: str = "US",
    ) -> Dict:
        """
        Fill specific missing dates using optimized approach.

        This optimized gap-fill:
        1. Pre-fetches all data once for efficiency
        2. Uses same universe as bulk scans (intersection of IBD groups and stock_universe)
        3. Processes all missing dates with cached data

        Args:
            db: Database session
            missing_dates: List of dates to calculate rankings for

        Returns:
            Statistics about the gap-fill operation
        """
        if not missing_dates:
            return {
                'total_dates': 0,
                'processed': 0,
                'skipped': 0,
                'errors': 0,
            }

        normalized_market = (market or "US").upper()
        logger.info(
            "Filling %d missing dates (optimized) for market=%s",
            len(missing_dates), normalized_market,
        )
        start_time = datetime.now()

        # Pre-fetch ALL data upfront
        spy_data, all_prices, active_symbols, market_caps, _prefetch_stats = (
            self._prefetch_all_data(db, market=normalized_market)
        )

        if spy_data is None or spy_data.empty:
            logger.error("Cannot proceed without SPY data")
            return {
                'total_dates': len(missing_dates),
                'processed': 0,
                'skipped': 0,
                'errors': len(missing_dates),
                'error': 'Failed to fetch SPY benchmark data',
            }

        all_groups = IBDIndustryService.get_all_groups(db, market=normalized_market)

        stats = {
            'total_dates': len(missing_dates),
            'processed': 0,
            'skipped': 0,
            'errors': 0,
        }

        for calc_date in missing_dates:
            try:
                # Filter SPY to calculation_date
                spy_filtered = spy_data[spy_data.index.date <= calc_date]
                if spy_filtered.empty:
                    logger.warning(f"No SPY data for {calc_date}")
                    stats['errors'] += 1
                    continue

                spy_prices = spy_filtered['Close'].sort_index(ascending=False)

                # Calculate RS for each group from cache
                group_metrics = []
                for group_name in all_groups:
                    metrics = self._calculate_group_rs_from_cache(
                        db,
                        group_name,
                        spy_prices,
                        all_prices,
                        active_symbols,
                        market_caps,
                        calc_date,
                        market=normalized_market,
                    )
                    if metrics:
                        group_metrics.append(metrics)

                if group_metrics:
                    # Sort and rank
                    group_metrics.sort(key=lambda x: x['avg_rs_rating'], reverse=True)
                    for rank, metrics in enumerate(group_metrics, start=1):
                        metrics['rank'] = rank

                    # Store
                    self._store_rankings(
                        db,
                        calc_date,
                        group_metrics,
                        market=normalized_market,
                    )
                    stats['processed'] += 1
                    logger.debug(f"Filled gap for {calc_date}: {len(group_metrics)} groups")
                else:
                    stats['errors'] += 1
                    logger.warning(f"No results for {calc_date}")

            except Exception as e:
                stats['errors'] += 1
                logger.error(f"Error filling gap for {calc_date}: {e}")

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Optimized gap-fill complete: {stats['processed']} processed, "
            f"{stats['errors']} errors in {duration:.1f}s"
        )

        stats['duration_seconds'] = round(duration, 2)
        return stats

    def backfill_rankings_chunked(
        self,
        db: Session,
        start_date: date,
        end_date: date,
        chunk_size_days: int = 30
    ) -> Dict:
        """
        Backfill rankings in chunks to avoid memory issues.

        Processes the date range in chunks of `chunk_size_days`.

        Args:
            db: Database session
            start_date: Start of backfill range
            end_date: End of backfill range
            chunk_size_days: Days per processing chunk

        Returns:
            Aggregate statistics from all chunks
        """
        logger.info(f"Starting chunked backfill from {start_date} to {end_date}")

        total_stats = {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'total_dates': 0,
            'processed': 0,
            'skipped': 0,
            'errors': 0,
        }

        # Process in reverse chronological order (most recent first)
        chunk_end = end_date

        while chunk_end >= start_date:
            chunk_start = max(start_date, chunk_end - timedelta(days=chunk_size_days - 1))

            logger.info(f"Processing chunk: {chunk_start} to {chunk_end}")

            try:
                chunk_result = self.backfill_rankings(db, chunk_start, chunk_end)

                # Aggregate stats
                total_stats['total_dates'] += chunk_result.get('total_dates', 0)
                total_stats['processed'] += chunk_result.get('processed', 0)
                total_stats['skipped'] += chunk_result.get('skipped', 0)
                total_stats['errors'] += chunk_result.get('errors', 0)

            except Exception as e:
                logger.error(f"Error processing chunk {chunk_start} to {chunk_end}: {e}")
                total_stats['errors'] += chunk_size_days  # Estimate

            # Move to previous chunk
            chunk_end = chunk_start - timedelta(days=1)

        logger.info(
            f"Chunked backfill complete: {total_stats['processed']} processed, "
            f"{total_stats['skipped']} skipped, {total_stats['errors']} errors"
        )

        return total_stats
