"""
Service for calculating and managing IBD Industry Group Rankings.

Calculates daily rankings based on average RS rating of constituent stocks.
"""
import logging
from typing import Optional, Dict, List
from datetime import datetime, date, timedelta
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc

from ..database import SessionLocal
from ..models.industry import IBDGroupRank
from ..models.scan_result import Scan, ScanResult
from .ibd_industry_service import IBDIndustryService
from .price_cache_service import PriceCacheService
from .benchmark_cache_service import BenchmarkCacheService
from ..scanners.criteria.relative_strength import RelativeStrengthCalculator

logger = logging.getLogger(__name__)


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

    _instance = None

    def __init__(self):
        """Initialize the service."""
        self.price_cache = PriceCacheService.get_instance()
        self.benchmark_cache = BenchmarkCacheService.get_instance()
        self.rs_calculator = RelativeStrengthCalculator()

    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def calculate_group_rankings(
        self,
        db: Session,
        calculation_date: date = None
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

        logger.info(f"Calculating IBD group rankings for {calculation_date}")
        start_time = datetime.now()

        # Get all unique IBD groups
        all_groups = IBDIndustryService.get_all_groups(db)
        if not all_groups:
            logger.error("No IBD industry groups found")
            return []

        logger.info(f"Found {len(all_groups)} IBD industry groups")

        # Pre-fetch ALL data upfront (SPY + all stock prices in single bulk fetch)
        spy_data, all_prices, active_symbols = self._prefetch_all_data(db)

        if spy_data is None or spy_data.empty:
            logger.error("Failed to get SPY benchmark data")
            return []

        # Filter SPY data to calculation_date for accurate historical rankings
        if calculation_date < datetime.now().date():
            spy_data_filtered = spy_data[spy_data.index.date <= calculation_date]
        else:
            spy_data_filtered = spy_data

        # Prepare SPY prices series (most recent first for RS calculator)
        spy_prices = spy_data_filtered['Close'].sort_index(ascending=False)

        # Calculate RS for each group using pre-fetched cached data
        group_metrics = []

        for group_name in all_groups:
            try:
                metrics = self._calculate_group_rs_from_cache(
                    db, group_name, spy_prices, all_prices, active_symbols, calculation_date
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
        self._store_rankings(db, calculation_date, group_metrics)

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

        # Calculate RS for each symbol
        rs_ratings = []
        top_symbol = None
        top_rs = -1
        rs_above_80_count = 0

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

        return {
            'industry_group': group_name,
            'date': calculation_date,
            'avg_rs_rating': round(avg_rs, 2),
            'num_stocks': len(rs_ratings),
            'num_stocks_rs_above_80': rs_above_80_count,
            'top_symbol': top_symbol,
            'top_rs_rating': round(top_rs, 2) if top_rs > 0 else None,
        }

    def _store_rankings(
        self,
        db: Session,
        calculation_date: date,
        group_metrics: List[Dict]
    ) -> None:
        """
        Store group rankings in database.

        Upserts records (updates if exists, inserts if new).
        """
        try:
            for metrics in group_metrics:
                # Check if record exists for this group and date
                existing = db.query(IBDGroupRank).filter(
                    and_(
                        IBDGroupRank.industry_group == metrics['industry_group'],
                        IBDGroupRank.date == calculation_date
                    )
                ).first()

                if existing:
                    # Update existing record
                    existing.rank = metrics['rank']
                    existing.avg_rs_rating = metrics['avg_rs_rating']
                    existing.num_stocks = metrics['num_stocks']
                    existing.num_stocks_rs_above_80 = metrics['num_stocks_rs_above_80']
                    existing.top_symbol = metrics['top_symbol']
                    existing.top_rs_rating = metrics['top_rs_rating']
                else:
                    # Insert new record
                    record = IBDGroupRank(
                        industry_group=metrics['industry_group'],
                        date=calculation_date,
                        rank=metrics['rank'],
                        avg_rs_rating=metrics['avg_rs_rating'],
                        num_stocks=metrics['num_stocks'],
                        num_stocks_rs_above_80=metrics['num_stocks_rs_above_80'],
                        top_symbol=metrics['top_symbol'],
                        top_rs_rating=metrics['top_rs_rating'],
                    )
                    db.add(record)

            db.commit()
            logger.info(f"Stored {len(group_metrics)} group rankings for {calculation_date}")

        except Exception as e:
            logger.error(f"Error storing rankings: {e}", exc_info=True)
            db.rollback()
            raise

    def get_current_rankings(
        self,
        db: Session,
        limit: int = 197
    ) -> List[Dict]:
        """
        Get most recent group rankings with rank changes.

        Args:
            db: Database session
            limit: Max number of groups to return

        Returns:
            List of groups with rankings and rank changes
        """
        # Get most recent date with rankings
        latest_record = db.query(IBDGroupRank).order_by(
            desc(IBDGroupRank.date)
        ).first()

        if not latest_record:
            return []

        latest_date = latest_record.date

        # Get all rankings for latest date
        rankings = db.query(IBDGroupRank).filter(
            IBDGroupRank.date == latest_date
        ).order_by(IBDGroupRank.rank).limit(limit).all()

        # Get historical dates for rank changes
        period_days = {
            '1w': 5,    # 5 trading days
            '1m': 21,   # ~1 month
            '3m': 63,   # ~3 months
            '6m': 126,  # ~6 months
        }

        # Build result with rank changes
        result = []
        for ranking in rankings:
            item = {
                'industry_group': ranking.industry_group,
                'date': ranking.date.isoformat(),
                'rank': ranking.rank,
                'avg_rs_rating': ranking.avg_rs_rating,
                'num_stocks': ranking.num_stocks,
                'num_stocks_rs_above_80': ranking.num_stocks_rs_above_80,
                'top_symbol': ranking.top_symbol,
                'top_rs_rating': ranking.top_rs_rating,
            }

            # Calculate rank changes for each period
            for period_name, days in period_days.items():
                historical_rank = self._get_historical_rank(
                    db, ranking.industry_group, latest_date, days
                )
                if historical_rank is not None:
                    # Positive change means rank improved (moved up)
                    item[f'rank_change_{period_name}'] = historical_rank - ranking.rank
                else:
                    item[f'rank_change_{period_name}'] = None

            result.append(item)

        return result

    def _get_historical_rank(
        self,
        db: Session,
        industry_group: str,
        current_date: date,
        days_back: int
    ) -> Optional[int]:
        """
        Get historical rank for a group.

        Finds the closest ranking record within the specified days back window.
        """
        target_date = current_date - timedelta(days=days_back)

        # Find closest record to target date (within 7-day window)
        records = db.query(IBDGroupRank).filter(
            and_(
                IBDGroupRank.industry_group == industry_group,
                IBDGroupRank.date >= target_date - timedelta(days=7),
                IBDGroupRank.date <= target_date + timedelta(days=7)
            )
        ).all()

        if not records:
            return None

        # Pick the closest date to target (prefer earlier date if tied)
        def _distance_key(record):
            delta = record.date - target_date
            return (abs(delta.days), delta.days > 0)

        closest = min(records, key=_distance_key)
        return closest.rank

    def get_group_history(
        self,
        db: Session,
        industry_group: str,
        days: int = 180
    ) -> Dict:
        """
        Get historical ranking data for a specific group.

        Args:
            db: Database session
            industry_group: Group name
            days: Number of days of history

        Returns:
            Dict with current info and historical data points
        """
        cutoff_date = datetime.now().date() - timedelta(days=days)

        # Get historical records
        records = db.query(IBDGroupRank).filter(
            and_(
                IBDGroupRank.industry_group == industry_group,
                IBDGroupRank.date >= cutoff_date
            )
        ).order_by(IBDGroupRank.date.desc()).all()

        if not records:
            return {'industry_group': industry_group, 'history': []}

        # Current (most recent) data
        current = records[0]

        # Get rank changes
        period_days = {'1w': 5, '1m': 21, '3m': 63, '6m': 126}
        rank_changes = {}

        for period_name, day_count in period_days.items():
            historical_rank = self._get_historical_rank(
                db, industry_group, current.date, day_count
            )
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

        return {
            'industry_group': industry_group,
            'current_rank': current.rank,
            'current_avg_rs': current.avg_rs_rating,
            'num_stocks': current.num_stocks,
            'top_symbol': current.top_symbol,
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

            # Get scan results for this industry group
            results = db.query(ScanResult).filter(
                and_(
                    ScanResult.scan_id == latest_scan.scan_id,
                    ScanResult.ibd_industry_group == industry_group
                )
            ).order_by(desc(ScanResult.rs_rating)).all()

            stocks = []
            for r in results:
                stocks.append({
                    'symbol': r.symbol,
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
        limit: int = 20
    ) -> Dict:
        """
        Get groups with biggest rank changes over a period.

        Args:
            db: Database session
            period: '1w', '1m', '3m', or '6m'
            limit: Number of top gainers/losers to return

        Returns:
            Dict with 'gainers' and 'losers' lists
        """
        period_days_map = {
            '1w': 5,
            '1m': 21,
            '3m': 63,
            '6m': 126,
        }

        days = period_days_map.get(period, 5)

        # Get current rankings
        current_rankings = self.get_current_rankings(db, limit=197)

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
        active_symbols: set
    ) -> list:
        """
        Get symbols for a group, filtered to only active stocks in stock_universe.

        This ensures group ranking uses the same universe as bulk scans.

        Args:
            db: Database session
            group_name: IBD industry group name
            active_symbols: Set of active symbols from stock_universe

        Returns:
            List of symbols that are both in the group AND active in stock_universe
        """
        group_symbols = IBDIndustryService.get_group_symbols(db, group_name)
        return [s for s in group_symbols if s in active_symbols]

    def _prefetch_all_data(
        self,
        db: Session
    ) -> tuple:
        """
        Pre-fetch all data needed for backfill in one batch.

        This optimization fetches:
        1. SPY benchmark data once
        2. Active symbols from stock_universe (same source as bulk scans)
        3. All prices for all symbols across all groups in a single batch

        Returns:
            Tuple of (spy_prices_df, {symbol: prices_df}, active_symbols_set)
        """
        from .stock_universe_service import stock_universe_service

        logger.info("Pre-fetching all data for optimized backfill...")

        # 1. Get SPY data once
        spy_data = self.benchmark_cache.get_spy_data(period="2y")
        if spy_data is None or spy_data.empty:
            logger.error("Failed to get SPY benchmark data")
            return None, {}, set()

        logger.info(f"Fetched SPY data: {len(spy_data)} days")

        # 2. Get active symbols from stock_universe (same as bulk scans)
        active_symbols_list = stock_universe_service.get_active_symbols(db)
        active_symbols = set(active_symbols_list)
        logger.info(f"Found {len(active_symbols)} active symbols in stock_universe")

        # 3. Collect ALL unique symbols across ALL groups (filtered by active)
        all_groups = IBDIndustryService.get_all_groups(db)
        symbols_to_fetch = set()

        for group in all_groups:
            group_symbols = IBDIndustryService.get_group_symbols(db, group)
            validated = [s for s in group_symbols if s in active_symbols]
            symbols_to_fetch.update(validated)

        logger.info(
            f"Collecting prices for {len(symbols_to_fetch)} unique symbols "
            f"across {len(all_groups)} groups"
        )

        # 4. Batch fetch ALL prices in one call
        all_prices = self.price_cache.get_many(list(symbols_to_fetch), period="2y")

        logger.info(f"Pre-fetched prices for {len(all_prices)} symbols")

        return spy_data, all_prices, active_symbols

    def _delete_rankings_for_range(
        self,
        db: Session,
        start_date: date,
        end_date: date
    ) -> int:
        """
        Delete all existing rankings in the date range.

        This allows recalculation of existing data (no skipping).

        Args:
            db: Database session
            start_date: Start of range
            end_date: End of range

        Returns:
            Number of records deleted
        """
        deleted = db.query(IBDGroupRank).filter(
            and_(
                IBDGroupRank.date >= start_date,
                IBDGroupRank.date <= end_date
            )
        ).delete(synchronize_session=False)

        db.commit()
        logger.info(f"Deleted {deleted} existing rankings for {start_date} to {end_date}")
        return deleted

    def _calculate_group_rs_from_cache(
        self,
        db: Session,
        group_name: str,
        spy_prices: pd.Series,
        all_prices: Dict[str, pd.DataFrame],
        active_symbols: set,
        calculation_date: date
    ) -> Optional[Dict]:
        """
        Calculate average RS rating for a single industry group using pre-fetched prices.

        Only uses symbols that are active in stock_universe (intersection approach).

        Args:
            db: Database session
            group_name: IBD industry group name
            spy_prices: SPY price series (most recent first, filtered to calculation_date)
            all_prices: Dict of all pre-fetched prices {symbol: DataFrame}
            active_symbols: Set of active symbols from stock_universe
            calculation_date: Date for calculation

        Returns:
            Dict with group metrics or None if insufficient data
        """
        # Get validated symbols (intersection of IBD group and active stock_universe)
        symbols = self._get_validated_group_symbols(db, group_name, active_symbols)

        if not symbols:
            logger.debug(f"No validated symbols found for group: {group_name}")
            return None

        # Calculate RS for each symbol using cached prices
        rs_ratings = []
        top_symbol = None
        top_rs = -1
        rs_above_80_count = 0

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

        return {
            'industry_group': group_name,
            'date': calculation_date,
            'avg_rs_rating': round(avg_rs, 2),
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
        spy_data, all_prices, active_symbols = self._prefetch_all_data(db)

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
                        db, group_name, spy_prices, all_prices, active_symbols, calc_date
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
        lookback_days: int = 365
    ) -> List[date]:
        """
        Find missing trading dates in the ranking data.

        Args:
            db: Database session
            lookback_days: How many days back to check for gaps

        Returns:
            List of missing trading dates (weekdays without ranking data)
        """
        from sqlalchemy import func

        today = datetime.now().date()
        start_date = today - timedelta(days=lookback_days)

        # Get all dates that have ranking data
        existing_dates = db.query(
            func.distinct(IBDGroupRank.date)
        ).filter(
            IBDGroupRank.date >= start_date
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
        missing_dates: List[date]
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

        logger.info(f"Filling {len(missing_dates)} missing dates (optimized)")
        start_time = datetime.now()

        # Pre-fetch ALL data upfront
        spy_data, all_prices, active_symbols = self._prefetch_all_data(db)

        if spy_data is None or spy_data.empty:
            logger.error("Cannot proceed without SPY data")
            return {
                'total_dates': len(missing_dates),
                'processed': 0,
                'skipped': 0,
                'errors': len(missing_dates),
                'error': 'Failed to fetch SPY benchmark data',
            }

        all_groups = IBDIndustryService.get_all_groups(db)

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
                        db, group_name, spy_prices, all_prices, active_symbols, calc_date
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


# Singleton instance
ibd_group_rank_service = IBDGroupRankService.get_instance()
