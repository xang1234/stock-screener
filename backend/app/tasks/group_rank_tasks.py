"""
Celery tasks for IBD Industry Group ranking calculations.

Provides background tasks for:
- Daily group ranking calculation after market close
- Historical backfill of ranking data

All data-fetching tasks use the @serialized_data_fetch decorator
to ensure only one task fetches external data at a time.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import logging
from typing import Optional
from datetime import datetime, date, timedelta
import time

from celery.exceptions import SoftTimeLimitExceeded

from ..celery_app import celery_app
from ..database import SessionLocal
from ..services.ibd_group_rank_service import (
    IncompleteGroupRankingCacheError,
)
from ..wiring.bootstrap import get_group_rank_service
from ..utils.market_hours import is_trading_day, get_eastern_now
from .data_fetch_lock import serialized_data_fetch

logger = logging.getLogger(__name__)
TRANSIENT_TASK_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)
_ALLOW_SAME_DAY_WARMUP_BYPASS: ContextVar[bool] = ContextVar(
    "allow_same_day_group_rank_warmup_bypass",
    default=False,
)


@contextmanager
def allow_same_day_group_rank_warmup_bypass():
    """Allow same-day cache-only rankings without warmup metadata in-process."""
    token = _ALLOW_SAME_DAY_WARMUP_BYPASS.set(True)
    try:
        yield
    finally:
        _ALLOW_SAME_DAY_WARMUP_BYPASS.reset(token)


def _retry_transient_failure(task, task_name: str, exc: Exception) -> None:
    retries = getattr(getattr(task, "request", None), "retries", 0) or 0
    countdown = min(60 * (2 ** retries), 600)
    logger.warning(
        "Transient error in %s: %s. Retrying in %ss (attempt %s/2).",
        task_name,
        exc,
        countdown,
        retries + 1,
    )
    raise task.retry(exc=exc, countdown=countdown, max_retries=2)


def _validate_same_day_cache_only_group_rankings(
    price_cache,
    market: Optional[str] = None,
) -> Optional[str]:
    """Block same-day group rankings when the post-close warmup is incomplete."""
    warmup_meta = price_cache.get_warmup_metadata(market=market) if price_cache else None
    if not warmup_meta:
        return "Missing cache warmup metadata for same-day group ranking run"

    if warmup_meta.get("status") != "completed":
        return (
            f"Cache warmup not complete for same-day group ranking run "
            f"({warmup_meta.get('status')}, {warmup_meta.get('count')}/{warmup_meta.get('total')})"
        )

    completed_at_raw = warmup_meta.get("completed_at")
    if completed_at_raw:
        try:
            completed_at = datetime.fromisoformat(completed_at_raw)
            if datetime.now() - completed_at > timedelta(hours=12):
                return "Cache warmup metadata is stale for same-day group ranking run"
        except ValueError:
            return "Cache warmup metadata timestamp is invalid"

    return None


@celery_app.task(
    bind=True,
    name='app.tasks.group_rank_tasks.calculate_daily_group_rankings',
    soft_time_limit=3600,
    max_retries=2,
)
@serialized_data_fetch('calculate_daily_group_rankings')
def calculate_daily_group_rankings(
    self,
    calculation_date: str | None = None,
    force_cache_only: bool = False,
    market: str | None = None,
):
    """
    Calculate and store daily IBD industry group rankings.

    This task calculates rankings for all IBD industry groups based on
    average RS rating of constituent stocks.

    Args:
        calculation_date: Optional YYYY-MM-DD string (defaults to today)

    Returns:
        Dict with calculation results
    """
    from .market_queues import market_tag, log_extra, normalize_market
    from ..services.runtime_preferences_service import is_market_enabled_now
    _log_extra = log_extra(market)
    logger.info("=" * 60)
    logger.info(
        "TASK: Calculate Daily IBD Group Rankings %s", market_tag(market), extra=_log_extra,
    )
    if market is not None and not is_market_enabled_now(normalize_market(market)):
        logger.info("Skipping group rankings for disabled market %s", market, extra=_log_extra)
        return {
            'status': 'skipped',
            'reason': f'market {normalize_market(market)} is disabled in local runtime preferences',
            'market': normalize_market(market),
            'timestamp': datetime.now().isoformat(),
        }
    # Group-rank computation aggregates across markets; however, same-day
    # warmup completeness validation is market-scoped via warmup metadata.
    if market is not None:
        logger.debug(
            "Group-rank computation aggregates across markets; warmup gate is market-scoped.",
            extra=_log_extra,
        )
    today_et = get_eastern_now().date()

    # Parse date
    if calculation_date:
        try:
            calc_date = datetime.strptime(calculation_date, '%Y-%m-%d').date()
            logger.info(f"Calculating group rankings for: {calc_date}")
        except ValueError as e:
            logger.error(f"Invalid date format: {calculation_date}. Use YYYY-MM-DD")
            return {'error': 'Invalid date format', 'timestamp': datetime.now().isoformat()}
    else:
        calc_date = today_et

        # Skip on non-trading days (weekends, holidays)
        if not is_trading_day(calc_date):
            logger.info(f"Skipping group rankings - {calc_date} is not a trading day")
            return {'skipped': True, 'reason': 'Not a trading day', 'date': calc_date.isoformat()}

        logger.info(f"Calculating group rankings for today: {calc_date}")

    logger.info("=" * 60)

    db = SessionLocal()
    start_time = time.time()

    try:
        # Initialize service
        service = get_group_rank_service()
        same_day_cache_only = force_cache_only or calc_date == today_et

        if same_day_cache_only:
            if force_cache_only or _ALLOW_SAME_DAY_WARMUP_BYPASS.get():
                logger.info(
                    "Bypassing same-day group ranking warmup metadata gate for in-process static export"
                )
            else:
                completeness_error = _validate_same_day_cache_only_group_rankings(
                    service.price_cache,
                    market=market,
                )
                if completeness_error:
                    logger.error("✗ Refusing to publish daily group rankings: %s", completeness_error)
                    logger.info("=" * 60)
                    return {
                        'error': completeness_error,
                        'date': calc_date.strftime('%Y-%m-%d'),
                        'timestamp': datetime.now().isoformat(),
                        'cache_only': True,
                    }

        # Calculate rankings
        logger.info(f"Starting group ranking calculation for {calc_date}...")
        results = service.calculate_group_rankings(
            db,
            calc_date,
            cache_only=same_day_cache_only,
            require_complete_cache=same_day_cache_only,
        )

        # Calculate duration
        duration = time.time() - start_time

        if not results:
            logger.warning(f"No groups ranked for {calc_date}")
            return {
                'date': calc_date.strftime('%Y-%m-%d'),
                'groups_ranked': 0,
                'warning': 'No groups could be ranked',
                'calculation_duration_seconds': round(duration, 2),
                'timestamp': datetime.now().isoformat()
            }

        logger.info(f"Successfully ranked {len(results)} groups in {duration:.2f}s")

        # Log top 5 groups
        logger.info("Top 5 groups:")
        for r in results[:5]:
            logger.info(
                f"  #{r['rank']}: {r['industry_group']} "
                f"(avg RS: {r['avg_rs_rating']:.1f}, {r['num_stocks']} stocks)"
            )

        logger.info("=" * 60)

        try:
            from ..services.ui_snapshot_service import safe_publish_groups_bootstrap

            safe_publish_groups_bootstrap()
        except Exception as snapshot_error:
            logger.warning("Group rankings snapshot publish failed: %s", snapshot_error)

        return {
            'date': calc_date.strftime('%Y-%m-%d'),
            'groups_ranked': len(results),
            'top_group': results[0]['industry_group'] if results else None,
            'top_avg_rs': results[0]['avg_rs_rating'] if results else None,
            'calculation_duration_seconds': round(duration, 2),
            'cache_only': same_day_cache_only,
            'timestamp': datetime.now().isoformat()
        }

    except SoftTimeLimitExceeded:
        db.rollback()
        logger.error("Soft time limit exceeded in calculate_daily_group_rankings", exc_info=True)
        raise
    except IncompleteGroupRankingCacheError as e:
        db.rollback()
        logger.error("✗ Refusing to publish daily group rankings: %s", e)
        logger.info("=" * 60)
        return {
            'error': str(e),
            'date': calc_date.strftime('%Y-%m-%d') if calc_date else None,
            'timestamp': datetime.now().isoformat(),
            'cache_only': True,
            'prefetch_stats': e.stats,
        }
    except TRANSIENT_TASK_EXCEPTIONS as e:
        db.rollback()
        _retry_transient_failure(self, "calculate_daily_group_rankings", e)
    except Exception as e:
        db.rollback()
        logger.error(f"Error in calculate_daily_group_rankings task: {e}", exc_info=True)
        logger.info("=" * 60)
        return {
            'error': str(e),
            'date': calc_date.strftime('%Y-%m-%d') if calc_date else None,
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()


@celery_app.task(bind=True, name='app.tasks.group_rank_tasks.backfill_group_rankings')
@serialized_data_fetch('backfill_group_rankings')
def backfill_group_rankings(self, start_date: str, end_date: str):
    """
    Backfill historical group rankings for a date range (optimized version).

    This optimized backfill:
    1. Uses same universe as bulk scans (intersection of IBD groups and stock_universe)
    2. Deletes existing rankings and recalculates (no skipping)
    3. Pre-fetches all data once for efficiency

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        Dict with backfill statistics
    """
    logger.info("=" * 60)
    logger.info("TASK: Backfill IBD Group Rankings (Optimized)")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info("=" * 60)

    try:
        # Parse dates
        start = datetime.strptime(start_date, '%Y-%m-%d').date()
        end = datetime.strptime(end_date, '%Y-%m-%d').date()

        if start > end:
            logger.error("Start date must be before end date")
            return {
                'error': 'Invalid date range: start_date > end_date',
                'timestamp': datetime.now().isoformat()
            }

    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        return {
            'error': f'Invalid date format. Use YYYY-MM-DD: {e}',
            'timestamp': datetime.now().isoformat()
        }

    db = SessionLocal()
    start_time = time.time()

    try:
        # Initialize service
        service = get_group_rank_service()

        # Use optimized backfill (deletes existing, pre-fetches all data, uses validated universe)
        result = service.backfill_rankings_optimized(db, start, end)

        # Calculate total duration
        total_duration = time.time() - start_time

        logger.info("=" * 60)
        logger.info("Backfill Complete!")
        logger.info(f"Total days: {result['total_dates']}")
        logger.info(f"Deleted existing: {result.get('deleted', 0)}")
        logger.info(f"Processed: {result['processed']}")
        logger.info(f"Skipped: {result['skipped']}")
        logger.info(f"Errors: {result['errors']}")
        logger.info(f"Total duration: {total_duration:.2f}s")
        logger.info("=" * 60)

        try:
            from ..services.ui_snapshot_service import safe_publish_groups_bootstrap

            safe_publish_groups_bootstrap()
        except Exception as snapshot_error:
            logger.warning("Group rankings snapshot publish failed after backfill: %s", snapshot_error)

        return {
            'start_date': start_date,
            'end_date': end_date,
            'total_dates': result['total_dates'],
            'deleted': result.get('deleted', 0),
            'processed': result['processed'],
            'skipped': result['skipped'],
            'errors': result['errors'],
            'total_duration_seconds': round(total_duration, 2),
            'avg_duration_per_day': round(
                total_duration / max(result['processed'], 1), 2
            ),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Error in backfill_group_rankings task: {e}", exc_info=True)
        logger.info("=" * 60)
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()


@celery_app.task(bind=True, name='app.tasks.group_rank_tasks.gapfill_group_rankings')
@serialized_data_fetch('gapfill_group_rankings')
def gapfill_group_rankings(self, max_days: int = 365):
    """
    Detect and fill gaps in group ranking data (optimized version).

    This optimized gap-fill:
    1. Uses same universe as bulk scans (intersection of IBD groups and stock_universe)
    2. Pre-fetches all data once for efficiency
    3. Processes all missing dates with cached data

    Serialization with other data-fetching tasks is handled by the
    @serialized_data_fetch decorator and the data_fetch queue.

    Args:
        max_days: Maximum days to look back for gaps

    Returns:
        Dict with gap-fill statistics
    """
    logger.info("=" * 60)
    logger.info("TASK: Gap-Fill IBD Group Rankings (Optimized)")
    logger.info(f"Looking back {max_days} days")
    logger.info("=" * 60)

    db = SessionLocal()
    start_time = time.time()

    try:
        service = get_group_rank_service()

        # Find missing dates
        missing_dates = service.find_missing_dates(db, lookback_days=max_days)

        if not missing_dates:
            logger.info("No gaps found - data is complete")
            return {
                'status': 'complete',
                'gaps_found': 0,
                'message': 'No gaps to fill',
                'timestamp': datetime.now().isoformat()
            }

        logger.info(f"Found {len(missing_dates)} gaps to fill")
        logger.info(f"Date range: {missing_dates[0]} to {missing_dates[-1]}")

        # Fill the gaps using optimized method
        result = service.fill_gaps_optimized(db, missing_dates)

        duration = time.time() - start_time

        logger.info("=" * 60)
        logger.info("Gap-Fill Complete!")
        logger.info(f"Gaps found: {len(missing_dates)}")
        logger.info(f"Processed: {result['processed']}")
        logger.info(f"Errors: {result['errors']}")
        logger.info(f"Duration: {duration:.2f}s")
        logger.info("=" * 60)

        try:
            from ..services.ui_snapshot_service import safe_publish_groups_bootstrap

            safe_publish_groups_bootstrap()
        except Exception as snapshot_error:
            logger.warning("Group rankings snapshot publish failed after gapfill: %s", snapshot_error)

        return {
            'status': 'complete',
            'gaps_found': len(missing_dates),
            'processed': result['processed'],
            'errors': result['errors'],
            'total_duration_seconds': round(duration, 2),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Error in gapfill_group_rankings task: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()


@celery_app.task(bind=True, name='app.tasks.group_rank_tasks.backfill_group_rankings_1year')
@serialized_data_fetch('backfill_group_rankings_1year')
def backfill_group_rankings_1year(self):
    """
    One-time task to backfill 1 year of group rankings (optimized version).

    This optimized backfill:
    1. Uses same universe as bulk scans (intersection of IBD groups and stock_universe)
    2. Deletes existing rankings and recalculates (no skipping)
    3. Pre-fetches all data once for efficiency

    Returns:
        Dict with backfill statistics
    """
    logger.info("=" * 60)
    logger.info("TASK: 1-Year Backfill IBD Group Rankings (Optimized)")
    logger.info("=" * 60)

    db = SessionLocal()
    start_time = time.time()

    try:
        service = get_group_rank_service()

        # Calculate date range
        end_date = get_eastern_now().date()
        start_date = end_date - timedelta(days=365)

        # Use optimized backfill (deletes existing, pre-fetches all data, uses validated universe)
        result = service.backfill_rankings_optimized(db, start_date, end_date)

        duration = time.time() - start_time

        logger.info("=" * 60)
        logger.info("1-Year Backfill Complete!")
        logger.info(f"Total days: {result['total_dates']}")
        logger.info(f"Deleted existing: {result.get('deleted', 0)}")
        logger.info(f"Processed: {result['processed']}")
        logger.info(f"Skipped: {result['skipped']}")
        logger.info(f"Errors: {result['errors']}")
        logger.info(f"Duration: {duration:.2f}s")
        logger.info("=" * 60)

        try:
            from ..services.ui_snapshot_service import safe_publish_groups_bootstrap

            safe_publish_groups_bootstrap()
        except Exception as snapshot_error:
            logger.warning("Group rankings snapshot publish failed after 1-year backfill: %s", snapshot_error)

        result['total_duration_seconds'] = round(duration, 2)
        result['timestamp'] = datetime.now().isoformat()

        return result

    except Exception as e:
        db.rollback()
        logger.error(f"Error in backfill_group_rankings_1year: {e}", exc_info=True)
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()
