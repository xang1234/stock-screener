"""Focused legacy Group ranking implementation extracted from the public facade."""

import gc
import logging
from datetime import date, datetime, timedelta
from typing import Dict, List

from sqlalchemy.orm import Session

from ..config import settings
from ..infra.db.repositories.market_rs_repo import MarketRsRunRepository
from ..models.industry import IBDGroupRank
from .ibd_industry_service import IBDIndustryService
from .group_ranking_calculation_service import GroupRankingCalculationService
from .legacy_group_rank_contracts import GroupRankPrefetchData
from .legacy_group_rank_data import LegacyGroupRankingEngine
from .market_calendar_service import MarketCalendarService

logger = logging.getLogger(__name__)


class LegacyGroupBackfillService:
    """Legacy history/gap-fill workflow with explicit collaborators."""

    def __init__(
        self,
        *,
        ranking_engine: LegacyGroupRankingEngine,
        market_calendar: MarketCalendarService,
        market_rs_repository: MarketRsRunRepository,
        calculation_service: GroupRankingCalculationService,
    ) -> None:
        self.ranking_engine = ranking_engine
        self.market_calendar = market_calendar
        self.market_rs_repository = market_rs_repository
        self.calculation_service = calculation_service

    def _process_optimized_dates(
        self,
        db: Session,
        *,
        dates: List[date],
        market: str,
        groups: List[str],
        prefetch: GroupRankPrefetchData,
    ) -> tuple[int, int]:
        """Calculate/store optimized legacy dates through one shared loop."""
        processed = 0
        errors = 0
        symbols_by_group = self.ranking_engine.symbols_by_group_for_run(
            db,
            groups,
            prefetch,
            market=market,
        )
        chunk_size = max(1, int(settings.group_rank_gapfill_chunk_size or 30))
        for chunk_start in range(0, len(dates), chunk_size):
            date_chunk = dates[chunk_start:chunk_start + chunk_size]
            rs_by_date = self.ranking_engine.calculate_rs_by_symbol_for_dates(
                prefetch,
                date_chunk,
            )
            for calculation_date in date_chunk:
                try:
                    rankings = self.calculation_service.rank_legacy_metrics(
                        groups=groups,
                        symbols_by_group=symbols_by_group,
                        prefetch=prefetch,
                        rs_by_symbol=rs_by_date.get(calculation_date, {}),
                        calculation_date=calculation_date,
                    )
                    if not rankings:
                        errors += 1
                        logger.warning("No valid Groups for %s", calculation_date)
                        continue
                    self.calculation_service.store_legacy_rankings(
                        db,
                        calculation_date,
                        rankings,
                        market=market,
                    )
                    processed += 1
                except Exception as exc:
                    errors += 1
                    logger.error("Error processing %s: %s", calculation_date, exc)
            rs_by_date.clear()
            gc.collect()
        return processed, errors

    def backfill_rankings_optimized(
        self,
        db: Session,
        start_date: date,
        end_date: date,
        *,
        market: str = "US",
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
        normalized_market = (market or "US").upper()
        logger.info(
            "Starting optimized backfill for market=%s from %s to %s",
            normalized_market, start_date, end_date,
        )
        start_time = datetime.now()

        # 1. Delete existing rankings in range
        deleted = self.ranking_engine.delete_rankings_for_range(
            db,
            start_date,
            end_date,
            market=normalized_market,
        )

        # 2. Pre-fetch ALL data upfront
        prefetch = self.ranking_engine.prefetch_all_data(
            db,
            market=normalized_market,
        )

        if prefetch.benchmark_prices is None or prefetch.benchmark_prices.empty:
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

        all_groups = list(prefetch.symbols_by_group) or IBDIndustryService.get_all_groups(
            db,
            market=normalized_market,
        )

        # 3. Generate trading dates using the target market's calendar.
        dates_to_process = []
        current = end_date
        while current >= start_date:
            if self.market_calendar.is_trading_day(normalized_market, current):
                dates_to_process.append(current)
            current -= timedelta(days=1)

        logger.info(
            f"Processing {len(dates_to_process)} trading days with "
            f"{len(prefetch.prices_by_symbol)} symbols across {len(all_groups)} groups"
        )

        processed, errors = self._process_optimized_dates(
            db,
            dates=dates_to_process,
            market=normalized_market,
            groups=all_groups,
            prefetch=prefetch,
        )

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
        end_date: date,
        *,
        market: str = "US",
        formula_version: str | None = None,
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
        normalized_market = (market or "US").upper()
        resolved_formula = formula_version or self.market_rs_repository.active_formula(
            db,
            market=normalized_market,
        )
        logger.info(
            "Starting backfill for market=%s from %s to %s",
            normalized_market, start_date, end_date,
        )

        # Generate list of dates to process using the target market's calendar.
        current_date = end_date
        dates_to_process = []

        while current_date >= start_date:
            if self.market_calendar.is_trading_day(normalized_market, current_date):
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
                    IBDGroupRank.date == calc_date,
                    IBDGroupRank.market == normalized_market,
                    IBDGroupRank.rs_formula_version == resolved_formula,
                ).first()

                if existing:
                    logger.debug(f"Skipping {calc_date} - already calculated")
                    skipped += 1
                    continue

                # Calculate rankings for this date
                results = self.calculation_service.calculate_and_store(
                    db,
                    calc_date,
                    market=normalized_market,
                    formula_version=resolved_formula,
                )

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
        end_date: date | None = None,
        formula_version: str | None = None,
    ) -> List[date]:
        """Find missing trading dates in the ranking data for one market."""
        from sqlalchemy import func

        normalized_market = (market or "US").upper()
        resolved_formula = formula_version or self.market_rs_repository.active_formula(
            db,
            market=normalized_market,
        )
        window_end = end_date or self.market_calendar.market_now(normalized_market).date()
        start_date = window_end - timedelta(days=lookback_days)

        existing_dates = db.query(
            func.distinct(IBDGroupRank.date)
        ).filter(
            IBDGroupRank.date >= start_date,
            IBDGroupRank.market == normalized_market,
            IBDGroupRank.rs_formula_version == resolved_formula,
        ).all()

        existing_date_set = {d[0] for d in existing_dates}

        # Generate all market trading days in range
        missing_dates = []
        current_date = start_date

        while current_date < window_end:  # Exclude the target day; it is calculated separately.
            if self.market_calendar.is_trading_day(normalized_market, current_date):
                if current_date not in existing_date_set:
                    missing_dates.append(current_date)
            current_date += timedelta(days=1)

        logger.info(f"Found {len(missing_dates)} missing dates in last {lookback_days} days")
        return sorted(missing_dates)

    def fill_gaps(
        self,
        db: Session,
        missing_dates: List[date],
        *,
        market: str = "US",
        formula_version: str | None = None,
    ) -> Dict:
        """
        Fill specific missing dates (used by startup gap-fill).

        Args:
            db: Database session
            missing_dates: List of dates to calculate rankings for

        Returns:
            Statistics about the gap-fill operation
        """
        normalized_market = (market or "US").upper()
        logger.info(
            "Filling %d missing dates for market=%s",
            len(missing_dates), normalized_market,
        )

        stats = {
            'total_dates': len(missing_dates),
            'processed': 0,
            'skipped': 0,
            'errors': 0,
        }

        for calc_date in missing_dates:
            try:
                results = self.calculation_service.calculate_and_store(
                    db,
                    calc_date,
                    market=normalized_market,
                    formula_version=formula_version,
                )

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
        prefetch = self.ranking_engine.prefetch_all_data(
            db,
            market=normalized_market,
        )

        if prefetch.benchmark_prices is None or prefetch.benchmark_prices.empty:
            logger.error("Cannot proceed without SPY data")
            return {
                'total_dates': len(missing_dates),
                'processed': 0,
                'skipped': 0,
                'errors': len(missing_dates),
                'error': 'Failed to fetch SPY benchmark data',
            }

        all_groups = list(prefetch.symbols_by_group) or IBDIndustryService.get_all_groups(
            db,
            market=normalized_market,
        )

        stats = {
            'total_dates': len(missing_dates),
            'processed': 0,
            'skipped': 0,
            'errors': 0,
        }

        stats['processed'], stats['errors'] = self._process_optimized_dates(
            db,
            dates=missing_dates,
            market=normalized_market,
            groups=all_groups,
            prefetch=prefetch,
        )

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
        chunk_size_days: int = 30,
        *,
        market: str = "US",
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
        normalized_market = (market or "US").upper()
        logger.info(
            "Starting chunked backfill for market=%s from %s to %s",
            normalized_market, start_date, end_date,
        )

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
                chunk_result = self.backfill_rankings(
                    db,
                    chunk_start,
                    chunk_end,
                    market=normalized_market,
                )

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
