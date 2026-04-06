"""
Celery tasks for market breadth calculations.

Provides background tasks for:
- Daily breadth calculation after market close
- Historical backfill of breadth data
- On-demand breadth calculation

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
from ..services.breadth_calculator_service import BreadthCalculatorService
from ..models.market_breadth import MarketBreadth
from ..config import settings
from .data_fetch_lock import serialized_data_fetch

logger = logging.getLogger(__name__)
TRANSIENT_TASK_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)
CACHE_MISS_TOLERANCE_RATIO = 0.05  # Allow up to 5% cache misses in cache-only breadth runs
_ALLOW_SAME_DAY_BREADTH_WARMUP_BYPASS: ContextVar[bool] = ContextVar(
    "allow_same_day_breadth_warmup_bypass",
    default=False,
)


@contextmanager
def allow_same_day_breadth_warmup_bypass():
    """Allow same-day cache-only breadth runs without warmup metadata in-process."""
    token = _ALLOW_SAME_DAY_BREADTH_WARMUP_BYPASS.set(True)
    try:
        yield
    finally:
        _ALLOW_SAME_DAY_BREADTH_WARMUP_BYPASS.reset(token)


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


def _validate_same_day_cache_only_breadth(price_cache, metrics: dict) -> Optional[str]:
    """Block publishing daily breadth when the same-day warmup/cache state is incomplete."""
    warmup_meta = price_cache.get_warmup_metadata() if price_cache else None
    if not warmup_meta:
        return "Missing cache warmup metadata for same-day breadth run"

    if warmup_meta.get("status") != "completed":
        return (
            f"Cache warmup not complete for same-day breadth run "
            f"({warmup_meta.get('status')}, {warmup_meta.get('count')}/{warmup_meta.get('total')})"
        )

    completed_at_raw = warmup_meta.get("completed_at")
    if completed_at_raw:
        try:
            completed_at = datetime.fromisoformat(completed_at_raw)
            if datetime.now() - completed_at > timedelta(hours=12):
                return "Cache warmup metadata is stale for same-day breadth run"
        except ValueError:
            return "Cache warmup metadata timestamp is invalid"

    return _validate_same_day_cache_only_breadth_metrics(metrics)


def _validate_same_day_cache_only_breadth_metrics(metrics: dict) -> Optional[str]:
    """Validate cache completeness without requiring Redis warmup metadata."""
    cache_misses = int(metrics.get("cache_miss_stocks", 0) or 0)
    errors = int(metrics.get("error_stocks", 0) or 0)
    total_scanned = int(metrics.get("total_stocks_scanned", 0) or 0)
    if errors > 0:
        return f"Cache-only breadth run has errors (errors={errors})"
    if total_scanned > 0 and cache_misses / total_scanned > CACHE_MISS_TOLERANCE_RATIO:
        return (
            f"Cache-only breadth run exceeds miss tolerance "
            f"(cache_misses={cache_misses}, total={total_scanned}, "
            f"ratio={cache_misses / total_scanned:.1%}, limit={CACHE_MISS_TOLERANCE_RATIO:.0%})"
        )
    if cache_misses > 0:
        logger.warning(
            "Cache-only breadth run has %d cache misses out of %d stocks (%.1f%%) -- within tolerance",
            cache_misses, total_scanned, cache_misses / total_scanned * 100,
        )
    return None


def _generate_trading_dates(start: date, end: date) -> tuple[list[date], int]:
    """Return trading dates in chronological order and the skipped non-trading-day count."""
    from ..utils.market_hours import is_trading_day

    current_date = start
    trading_dates: list[date] = []
    skipped_non_trading_days = 0

    while current_date <= end:
        if is_trading_day(current_date):
            trading_dates.append(current_date)
        else:
            skipped_non_trading_days += 1
        current_date += timedelta(days=1)

    return trading_dates, skipped_non_trading_days


@celery_app.task(bind=True, name='app.tasks.breadth_tasks.calculate_daily_breadth')
@serialized_data_fetch('calculate_daily_breadth')
def calculate_daily_breadth(self, calculation_date: str = None, force_cache_only: bool = False):
    """
    Calculate and store daily market breadth indicators.

    This task calculates 13 breadth indicators for all active stocks
    in the universe and stores them in the database.

    Args:
        calculation_date: Optional YYYY-MM-DD string for backfill
                         (defaults to today)

    Returns:
        Dict with calculation results and metadata:
        {
            'date': str,
            'indicators': {all 13 breadth metrics},
            'total_stocks_scanned': int,
            'calculation_duration_seconds': float,
            'timestamp': str
        }
    """
    logger.info("=" * 60)
    logger.info("TASK: Calculate Daily Market Breadth")

    # Parse date
    from ..utils.market_hours import is_trading_day, get_eastern_now

    today_et = get_eastern_now().date()
    if calculation_date:
        try:
            calc_date = datetime.strptime(calculation_date, '%Y-%m-%d').date()
            logger.info(f"Calculating breadth for: {calc_date}")
        except ValueError as e:
            logger.error(f"Invalid date format: {calculation_date}. Use YYYY-MM-DD")
            return {'error': 'Invalid date format', 'timestamp': datetime.now().isoformat()}
    else:
        calc_date = today_et

        if not is_trading_day(calc_date):
            logger.warning(f"Today ({calc_date}) is not a trading day. Skipping.")
            return {
                'skipped': True,
                'reason': f'{calc_date} is not a trading day',
                'date': calc_date.strftime('%Y-%m-%d'),
                'timestamp': datetime.now().isoformat()
            }

        logger.info(f"Calculating breadth for today: {calc_date}")

    logger.info("=" * 60)

    db = SessionLocal()
    start_time = time.time()

    try:
        # Initialize breadth calculator service
        calculator = BreadthCalculatorService(db)
        cache_only = force_cache_only or calc_date == today_et

        # Calculate breadth indicators
        logger.info(f"Starting breadth calculation for {calc_date}...")
        metrics = calculator.calculate_daily_breadth(
            calculation_date=calc_date,
            cache_only=cache_only,
        )

        # Calculate duration
        duration = time.time() - start_time
        metrics['calculation_duration_seconds'] = round(duration, 2)

        if cache_only:
            if force_cache_only or _ALLOW_SAME_DAY_BREADTH_WARMUP_BYPASS.get():
                logger.info(
                    "Bypassing same-day breadth warmup metadata gate for in-process static export"
                )
                completeness_error = _validate_same_day_cache_only_breadth_metrics(metrics)
            else:
                completeness_error = _validate_same_day_cache_only_breadth(
                    calculator.price_cache,
                    metrics,
                )
            if completeness_error:
                logger.error("✗ Refusing to publish daily breadth: %s", completeness_error)
                logger.info("=" * 60)
                return {
                    'error': completeness_error,
                    'date': calc_date.strftime('%Y-%m-%d'),
                    'timestamp': datetime.now().isoformat(),
                    'cache_only': True,
                    'metrics': {
                        'total_stocks_scanned': metrics['total_stocks_scanned'],
                        'skipped_stocks': metrics.get('skipped_stocks', 0),
                        'cache_miss_stocks': metrics.get('cache_miss_stocks', 0),
                        'error_stocks': metrics.get('error_stocks', 0),
                    },
                }

        logger.info(f"✓ Breadth calculation completed in {duration:.2f}s")
        logger.info(f"  Stocks scanned: {metrics['total_stocks_scanned']}")
        logger.info(f"  Stocks skipped: {metrics.get('skipped_stocks', 0)}")
        logger.info(f"  Up 4%+: {metrics['stocks_up_4pct']}")
        logger.info(f"  Down 4%+: {metrics['stocks_down_4pct']}")
        logger.info(f"  5-day ratio: {metrics['ratio_5day']}")
        logger.info(f"  10-day ratio: {metrics['ratio_10day']}")

        # Check if record already exists for this date
        existing_record = db.query(MarketBreadth).filter(
            MarketBreadth.date == calc_date
        ).first()

        if existing_record:
            # Update existing record
            existing_record.stocks_up_4pct = metrics['stocks_up_4pct']
            existing_record.stocks_down_4pct = metrics['stocks_down_4pct']
            existing_record.ratio_5day = metrics['ratio_5day']
            existing_record.ratio_10day = metrics['ratio_10day']
            existing_record.stocks_up_25pct_quarter = metrics['stocks_up_25pct_quarter']
            existing_record.stocks_down_25pct_quarter = metrics['stocks_down_25pct_quarter']
            existing_record.stocks_up_25pct_month = metrics['stocks_up_25pct_month']
            existing_record.stocks_down_25pct_month = metrics['stocks_down_25pct_month']
            existing_record.stocks_up_50pct_month = metrics['stocks_up_50pct_month']
            existing_record.stocks_down_50pct_month = metrics['stocks_down_50pct_month']
            existing_record.stocks_up_13pct_34days = metrics['stocks_up_13pct_34days']
            existing_record.stocks_down_13pct_34days = metrics['stocks_down_13pct_34days']
            existing_record.total_stocks_scanned = metrics['total_stocks_scanned']
            existing_record.calculation_duration_seconds = duration
            logger.info(f"Updating existing breadth record for {calc_date}")
        else:
            # Create new MarketBreadth record
            breadth_record = MarketBreadth(
                date=calc_date,
                stocks_up_4pct=metrics['stocks_up_4pct'],
                stocks_down_4pct=metrics['stocks_down_4pct'],
                ratio_5day=metrics['ratio_5day'],
                ratio_10day=metrics['ratio_10day'],
                stocks_up_25pct_quarter=metrics['stocks_up_25pct_quarter'],
                stocks_down_25pct_quarter=metrics['stocks_down_25pct_quarter'],
                stocks_up_25pct_month=metrics['stocks_up_25pct_month'],
                stocks_down_25pct_month=metrics['stocks_down_25pct_month'],
                stocks_up_50pct_month=metrics['stocks_up_50pct_month'],
                stocks_down_50pct_month=metrics['stocks_down_50pct_month'],
                stocks_up_13pct_34days=metrics['stocks_up_13pct_34days'],
                stocks_down_13pct_34days=metrics['stocks_down_13pct_34days'],
                total_stocks_scanned=metrics['total_stocks_scanned'],
                calculation_duration_seconds=duration
            )
            db.add(breadth_record)
            logger.info(f"Creating new breadth record for {calc_date}")

        db.commit()
        try:
            from ..services.ui_snapshot_service import safe_publish_breadth_bootstrap

            safe_publish_breadth_bootstrap()
        except Exception as snapshot_error:
            logger.warning("Breadth snapshot publish failed: %s", snapshot_error)

        logger.info(f"✓ Breadth data saved to database for {calc_date}")
        logger.info("=" * 60)

        return {
            'date': calc_date.strftime('%Y-%m-%d'),
            'indicators': {
                'stocks_up_4pct': metrics['stocks_up_4pct'],
                'stocks_down_4pct': metrics['stocks_down_4pct'],
                'ratio_5day': metrics['ratio_5day'],
                'ratio_10day': metrics['ratio_10day'],
                'stocks_up_25pct_quarter': metrics['stocks_up_25pct_quarter'],
                'stocks_down_25pct_quarter': metrics['stocks_down_25pct_quarter'],
                'stocks_up_25pct_month': metrics['stocks_up_25pct_month'],
                'stocks_down_25pct_month': metrics['stocks_down_25pct_month'],
                'stocks_up_50pct_month': metrics['stocks_up_50pct_month'],
                'stocks_down_50pct_month': metrics['stocks_down_50pct_month'],
                'stocks_up_13pct_34days': metrics['stocks_up_13pct_34days'],
                'stocks_down_13pct_34days': metrics['stocks_down_13pct_34days'],
            },
            'total_stocks_scanned': metrics['total_stocks_scanned'],
            'calculation_duration_seconds': duration,
            'cache_only': cache_only,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        db.rollback()
        logger.error(f"✗ Error in calculate_daily_breadth task: {e}", exc_info=True)
        logger.info("=" * 60)
        return {
            'error': str(e),
            'date': calc_date.strftime('%Y-%m-%d') if calc_date else None,
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()


@celery_app.task(bind=True, name='app.tasks.breadth_tasks.backfill_breadth_data')
@serialized_data_fetch('backfill_breadth_data')
def backfill_breadth_data(self, start_date: str, end_date: str):
    """
    Backfill market breadth data for historical date range.

    Processes each date in the range and calculates breadth indicators.
    Useful for populating historical data or filling gaps.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        Dict with backfill statistics:
        {
            'start_date': str,
            'end_date': str,
            'total_days_processed': int,
            'successful': int,
            'failed': int,
            'skipped_weekends': int,
            'total_duration_seconds': float,
            'timestamp': str
        }
    """
    logger.info("=" * 60)
    logger.info("TASK: Backfill Market Breadth Data")
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

    trading_dates, skipped_non_trading_days = _generate_trading_dates(start, end)

    logger.info(f"Generated {len(trading_dates)} trading days to process")
    logger.info(f"Skipped {skipped_non_trading_days} non-trading days")

    db = SessionLocal()
    start_time = time.time()

    try:
        calculator = BreadthCalculatorService(db)
        result = calculator.backfill_range(start, end, trading_dates=trading_dates)
        total_duration = time.time() - start_time

    except Exception as e:
        db.rollback()
        logger.error(f"Error during breadth backfill: {e}", exc_info=True)
        return {
            'error': str(e),
            'start_date': start_date,
            'end_date': end_date,
            'timestamp': datetime.now().isoformat(),
        }
    finally:
        db.close()

    logger.info("=" * 60)
    logger.info("Backfill Complete!")
    logger.info(f"Total days processed: {result['total_dates']}")
    logger.info(f"Successful: {result['processed']}")
    logger.info(f"Failed: {result['errors']}")
    logger.info(f"Skipped non-trading days: {skipped_non_trading_days}")
    logger.info(f"Total duration: {total_duration:.2f}s")
    logger.info(f"Average per day: {total_duration / max(result['total_dates'], 1):.2f}s")
    logger.info("=" * 60)

    try:
        from ..services.ui_snapshot_service import safe_publish_breadth_bootstrap

        safe_publish_breadth_bootstrap()
    except Exception as snapshot_error:
        logger.warning("Breadth snapshot publish failed after backfill: %s", snapshot_error)

    return {
        'start_date': start_date,
        'end_date': end_date,
        'total_days_processed': result['total_dates'],
        'successful': result['processed'],
        'failed': result['errors'],
        'skipped_weekends': skipped_non_trading_days,
        'skipped_non_trading_days': skipped_non_trading_days,
        'failed_dates': result['error_dates'][:10],  # Only return first 10 failures
        'total_duration_seconds': round(total_duration, 2),
        'avg_duration_per_day': round(total_duration / max(result['total_dates'], 1), 2),
        'timestamp': datetime.now().isoformat()
    }


@celery_app.task(
    bind=True,
    name='app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill',
    soft_time_limit=3600,
    max_retries=2,
)
@serialized_data_fetch('calculate_daily_breadth_with_gapfill')
def calculate_daily_breadth_with_gapfill(self, max_gap_days: int = None):
    """
    Calculate daily breadth with automatic gap detection and filling.

    This wrapper task:
    1. Detects missing trading days in the lookback window
    2. Fills gaps (oldest to newest) to ensure ratio accuracy
    3. Calculates today's breadth

    Args:
        max_gap_days: Maximum days to look back for gaps (default from settings)

    Returns:
        Dict with gap-fill stats and today's breadth calculation:
        {
            'gap_fill': {stats about gap filling},
            'today': {today's breadth calculation result},
            'timestamp': str
        }
    """
    logger.info("=" * 60)
    logger.info("TASK: Calculate Daily Market Breadth (with Gap-Fill)")
    logger.info("=" * 60)

    # Use config value if not specified
    if max_gap_days is None:
        max_gap_days = settings.breadth_gapfill_max_days

    db = SessionLocal()
    start_time = time.time()

    result = {
        'gap_fill': None,
        'today': None,
        'timestamp': datetime.now().isoformat()
    }

    try:
        calculator = BreadthCalculatorService(db)

        # Step 1: Check if gap-fill is enabled
        if settings.breadth_gapfill_enabled:
            logger.info(f"Checking for gaps in last {max_gap_days} days...")

            # Find missing dates
            missing_dates = calculator.find_missing_dates(lookback_days=max_gap_days)

            if missing_dates:
                logger.info(f"Found {len(missing_dates)} missing breadth dates")
                logger.info(f"Date range: {missing_dates[0]} to {missing_dates[-1]}")

                # Fill gaps
                gap_stats = calculator.fill_gaps(missing_dates)
                result['gap_fill'] = gap_stats

                logger.info(
                    f"✓ Gap-fill complete: {gap_stats['processed']} dates filled, "
                    f"{gap_stats['errors']} errors"
                )
            else:
                logger.info("No missing dates found - breadth data is complete")
                result['gap_fill'] = {
                    'total_dates': 0,
                    'processed': 0,
                    'errors': 0,
                    'message': 'No gaps detected'
                }
        else:
            logger.info("Gap-fill disabled in settings, skipping gap detection")
            result['gap_fill'] = {'message': 'Gap-fill disabled'}

        # Step 2: Calculate today's breadth (only if trading day)
        from ..utils.market_hours import is_trading_day, get_last_trading_day, get_eastern_now

        today = get_eastern_now().date()

        if is_trading_day(today):
            logger.info(f"Calculating breadth for today ({today})...")
            today_result = calculate_daily_breadth()
            result['today'] = today_result
        else:
            last_trading = get_last_trading_day(today)
            logger.info(f"Today ({today}) is not a trading day. Skipping.")
            result['today'] = {
                'skipped': True,
                'reason': f'{today} is not a trading day (weekend or holiday)',
                'last_trading_day': last_trading.strftime('%Y-%m-%d'),
                'date': today.strftime('%Y-%m-%d'),
                'timestamp': datetime.now().isoformat()
            }

        # Calculate total duration
        total_duration = time.time() - start_time

        logger.info("=" * 60)
        logger.info(f"✓ Breadth calculation complete in {total_duration:.2f}s")
        if result['gap_fill'] and result['gap_fill'].get('processed', 0) > 0:
            logger.info(f"  Gap-filled dates: {result['gap_fill']['processed']}")
        logger.info(f"  Today's result: {result['today'].get('date', 'N/A')}")
        logger.info("=" * 60)

        result['total_duration_seconds'] = round(total_duration, 2)

        return result

    except SoftTimeLimitExceeded:
        db.rollback()
        logger.error("Soft time limit exceeded in calculate_daily_breadth_with_gapfill", exc_info=True)
        raise
    except TRANSIENT_TASK_EXCEPTIONS as e:
        db.rollback()
        _retry_transient_failure(self, "calculate_daily_breadth_with_gapfill", e)
    except Exception as e:
        logger.error(f"✗ Error in calculate_daily_breadth_with_gapfill: {e}", exc_info=True)
        logger.info("=" * 60)
        return {
            'error': str(e),
            'gap_fill': result.get('gap_fill'),
            'today': result.get('today'),
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()
