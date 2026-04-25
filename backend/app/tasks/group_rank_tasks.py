"""
Celery tasks for IBD Industry Group ranking calculations.

Provides background tasks for:
- Daily group ranking calculation after market close
- Historical backfill of ranking data

Group ranking tasks mutate market-derived data and therefore use the
market workload lease to avoid same-market overlap with scans.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import logging
from typing import Optional
from datetime import datetime, date, timedelta
import time

from celery.exceptions import Retry, SoftTimeLimitExceeded

from ..celery_app import celery_app
from ..config import settings
from ..database import SessionLocal
from ..services.market_activity_service import (
    mark_market_activity_completed,
    mark_market_activity_failed,
    mark_market_activity_started,
)
from ..services.ibd_group_rank_service import (
    IncompleteGroupRankingCacheError,
    MissingIBDIndustryMappingsError,
)
from ..services.market_taxonomy_service import TaxonomyLoadError
from ..wiring.bootstrap import get_group_rank_service, get_market_calendar_service
from .workload_coordination import serialized_market_workload

logger = logging.getLogger(__name__)
TRANSIENT_TASK_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)
TAXONOMY_UNAVAILABLE_EXCEPTIONS = (
    MissingIBDIndustryMappingsError,
    TaxonomyLoadError,
)


class GroupRankReasonCode:
    INVALID_DATE = "invalid_date"
    WARMUP_INCOMPLETE = "warmup_incomplete"
    MISSING_IBD_MAPPINGS = "missing_ibd_mappings"
    NO_GROUPS_RANKED = "no_groups_ranked"
    UNKNOWN = "unknown"


_ALLOW_SAME_DAY_WARMUP_BYPASS: ContextVar[bool] = ContextVar(
    "allow_same_day_group_rank_warmup_bypass",
    default=False,
)
_PROPAGATE_IN_PROCESS_TRANSIENT_ERRORS: ContextVar[bool] = ContextVar(
    "propagate_in_process_group_rank_transient_errors",
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


def _mark_market_activity_failed_safely(db, **kwargs) -> None:
    try:
        mark_market_activity_failed(db, **kwargs)
    except Exception:
        logger.warning(
            "Failed to publish market activity failure for group ranking task",
            extra={
                "market": kwargs.get("market"),
                "stage_key": kwargs.get("stage_key"),
                "task_id": kwargs.get("task_id"),
            },
            exc_info=True,
        )


def _group_rank_result_error(result) -> str | None:
    if not isinstance(result, dict):
        return None
    error = result.get("error")
    if error:
        return str(error)
    return None


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


def _should_repair_current_us_metadata(
    *,
    calc_date: date,
    today_et: date,
    activity_lifecycle: str,
) -> bool:
    """Only repair live US surfaces for same-day or bootstrap ranking runs."""
    return activity_lifecycle == "bootstrap" or calc_date == today_et


@celery_app.task(
    bind=True,
    name='app.tasks.group_rank_tasks.calculate_daily_group_rankings',
    soft_time_limit=3600,
    max_retries=2,
)
@serialized_market_workload('calculate_daily_group_rankings')
def calculate_daily_group_rankings(
    self,
    calculation_date: str | None = None,
    force_cache_only: bool = False,
    market: str | None = None,
    activity_lifecycle: str | None = None,
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
    effective_market = normalize_market(market) if market is not None else "US"
    activity_lifecycle = activity_lifecycle or "daily_refresh"
    logger.info("=" * 60)
    logger.info(
        "TASK: Calculate Daily Industry Group Rankings %s", market_tag(market), extra=_log_extra,
    )
    if market is not None and not is_market_enabled_now(normalize_market(market)):
        logger.info("Skipping group rankings for disabled market %s", market, extra=_log_extra)
        return {
            'status': 'skipped',
            'reason': f'market {effective_market} is disabled in local runtime preferences',
            'market': effective_market,
            'timestamp': datetime.now().isoformat(),
        }
    calendar_service = get_market_calendar_service()
    today_local = calendar_service.market_now(effective_market).date()

    # Parse date
    if calculation_date:
        try:
            calc_date = datetime.strptime(calculation_date, '%Y-%m-%d').date()
            logger.info(f"Calculating group rankings for: {calc_date}")
        except ValueError as e:
            logger.error(f"Invalid date format: {calculation_date}. Use YYYY-MM-DD")
            return {
                'error': 'Invalid date format',
                'reason_code': GroupRankReasonCode.INVALID_DATE,
                'timestamp': datetime.now().isoformat(),
            }
    else:
        calc_date = today_local

        # Skip on non-trading days in this market (weekends, local holidays)
        if not calendar_service.is_trading_day(effective_market, calc_date):
            logger.info(
                "Skipping group rankings - %s is not a trading day for %s",
                calc_date, effective_market,
            )
            return {'skipped': True, 'reason': 'Not a trading day', 'date': calc_date.isoformat()}

        logger.info(f"Calculating group rankings for today ({effective_market}): {calc_date}")

    logger.info("=" * 60)

    db = SessionLocal()
    start_time = time.time()

    try:
        mark_market_activity_started(
            db,
            market=effective_market,
            stage_key="groups",
            lifecycle=activity_lifecycle,
            task_name=getattr(self, "name", "calculate_daily_group_rankings"),
            task_id=getattr(getattr(self, "request", None), "id", None),
            message="Calculating group rankings",
        )
        # Initialize service
        service = get_group_rank_service()
        same_day_cache_only = force_cache_only or calc_date == today_local

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
                    _mark_market_activity_failed_safely(
                        db,
                        market=effective_market,
                        stage_key="groups",
                        lifecycle=activity_lifecycle,
                        task_name=getattr(self, "name", "calculate_daily_group_rankings"),
                        task_id=getattr(getattr(self, "request", None), "id", None),
                        message=completeness_error,
                    )
                    return {
                        'error': completeness_error,
                        'reason_code': GroupRankReasonCode.WARMUP_INCOMPLETE,
                        'date': calc_date.strftime('%Y-%m-%d'),
                        'timestamp': datetime.now().isoformat(),
                        'cache_only': True,
                    }

        # Calculate rankings
        logger.info(f"Starting group ranking calculation for {calc_date}...")
        results = service.calculate_group_rankings(
            db,
            calc_date,
            market=effective_market,
            cache_only=same_day_cache_only,
            require_complete_cache=same_day_cache_only,
        )

        # Calculate duration
        duration = time.time() - start_time

        if not results:
            no_groups_message = (
                "No groups could be ranked (insufficient price data or all groups below 3-stock threshold)"
            )
            logger.warning("No groups ranked for %s", calc_date)
            _mark_market_activity_failed_safely(
                db,
                market=effective_market,
                stage_key="groups",
                lifecycle=activity_lifecycle,
                task_name=getattr(self, "name", "calculate_daily_group_rankings"),
                task_id=getattr(getattr(self, "request", None), "id", None),
                message=no_groups_message,
            )
            return {
                'date': calc_date.strftime('%Y-%m-%d'),
                'groups_ranked': 0,
                'warning': 'No groups could be ranked',
                'error': no_groups_message,
                'reason_code': GroupRankReasonCode.NO_GROUPS_RANKED,
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

        repair_stats = None
        # US-only: the repair helper touches US-scoped feature-store metadata
        # and has no equivalent on other markets. Non-US runs skip it.
        if effective_market == "US" and _should_repair_current_us_metadata(
            calc_date=calc_date,
            today_et=today_local,
            activity_lifecycle=activity_lifecycle,
        ):
            from ..interfaces.tasks.feature_store_tasks import _repair_current_us_group_metadata

            repair_stats = _repair_current_us_group_metadata(ranking_date=calc_date)

        try:
            from ..services.ui_snapshot_service import safe_publish_groups_bootstrap

            safe_publish_groups_bootstrap()
        except Exception as snapshot_error:
            logger.warning("Group rankings snapshot publish failed: %s", snapshot_error)
        mark_market_activity_completed(
            db,
            market=effective_market,
            stage_key="groups",
            lifecycle=activity_lifecycle,
            task_name=getattr(self, "name", "calculate_daily_group_rankings"),
            task_id=getattr(getattr(self, "request", None), "id", None),
            current=len(results),
            total=len(results),
            message="Group rankings completed",
        )

        return {
            'date': calc_date.strftime('%Y-%m-%d'),
            'groups_ranked': len(results),
            'top_group': results[0]['industry_group'] if results else None,
            'top_avg_rs': results[0]['avg_rs_rating'] if results else None,
            'calculation_duration_seconds': round(duration, 2),
            'cache_only': same_day_cache_only,
            'metadata_repair': repair_stats,
            'timestamp': datetime.now().isoformat()
        }

    except SoftTimeLimitExceeded:
        db.rollback()
        logger.error("Soft time limit exceeded in calculate_daily_group_rankings", exc_info=True)
        _mark_market_activity_failed_safely(
            db,
            market=effective_market,
            stage_key="groups",
            lifecycle=activity_lifecycle,
            task_name=getattr(self, "name", "calculate_daily_group_rankings"),
            task_id=getattr(getattr(self, "request", None), "id", None),
            message="Soft time limit exceeded",
        )
        raise
    except IncompleteGroupRankingCacheError as e:
        db.rollback()
        logger.error("✗ Refusing to publish daily group rankings: %s", e)
        logger.info("=" * 60)
        _mark_market_activity_failed_safely(
            db,
            market=effective_market,
            stage_key="groups",
            lifecycle=activity_lifecycle,
            task_name=getattr(self, "name", "calculate_daily_group_rankings"),
            task_id=getattr(getattr(self, "request", None), "id", None),
            message=str(e),
        )
        return {
            'error': str(e),
            'reason_code': GroupRankReasonCode.WARMUP_INCOMPLETE,
            'date': calc_date.strftime('%Y-%m-%d') if calc_date else None,
            'timestamp': datetime.now().isoformat(),
            'cache_only': True,
            'prefetch_stats': e.stats,
        }
    except MissingIBDIndustryMappingsError as e:
        db.rollback()
        logger.error("✗ Refusing to publish daily group rankings: %s", e)
        logger.info("=" * 60)
        _mark_market_activity_failed_safely(
            db,
            market=effective_market,
            stage_key="groups",
            lifecycle=activity_lifecycle,
            task_name=getattr(self, "name", "calculate_daily_group_rankings"),
            task_id=getattr(getattr(self, "request", None), "id", None),
            message=str(e),
        )
        return {
            'error': str(e),
            'reason_code': GroupRankReasonCode.MISSING_IBD_MAPPINGS,
            'date': calc_date.strftime('%Y-%m-%d') if calc_date else None,
            'timestamp': datetime.now().isoformat(),
            'cache_only': same_day_cache_only,
        }
    except TRANSIENT_TASK_EXCEPTIONS as e:
        db.rollback()
        if _PROPAGATE_IN_PROCESS_TRANSIENT_ERRORS.get():
            raise
        _mark_market_activity_failed_safely(
            db,
            market=effective_market,
            stage_key="groups",
            lifecycle=activity_lifecycle,
            task_name=getattr(self, "name", "calculate_daily_group_rankings"),
            task_id=getattr(getattr(self, "request", None), "id", None),
            message=str(e),
        )
        _retry_transient_failure(self, "calculate_daily_group_rankings", e)
    except Exception as e:
        db.rollback()
        logger.error(f"Error in calculate_daily_group_rankings task: {e}", exc_info=True)
        logger.info("=" * 60)
        _mark_market_activity_failed_safely(
            db,
            market=effective_market,
            stage_key="groups",
            lifecycle=activity_lifecycle,
            task_name=getattr(self, "name", "calculate_daily_group_rankings"),
            task_id=getattr(getattr(self, "request", None), "id", None),
            message=str(e),
        )
        return {
            'error': str(e),
            'reason_code': GroupRankReasonCode.UNKNOWN,
            'date': calc_date.strftime('%Y-%m-%d') if calc_date else None,
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()


def _calculate_daily_group_rankings_in_process(
    *,
    market: str | None = None,
    activity_lifecycle: str | None = None,
):
    """Run the daily ranking task body without re-acquiring the market workload lease.

    The orchestrator already holds the per-market workload lease, but the daily
    task's @serialized_market_workload decorator would otherwise try to re-acquire
    it from this nested call. ``disable_serialized_market_workload()`` flips a
    ContextVar that the decorator honors as a bypass — without it, the inner
    call's task_id resolves to "unknown" (Celery's per-class request_stack is
    empty for direct invocation), the reentrancy check is skipped, the SET NX
    fails, and Retry propagates back into the orchestrator.

    Transient exceptions are also propagated directly here so the outer
    orchestrator owns retry scheduling. Calling the Celery task body via
    ``task.run()`` does not provide a worker request context for the requested
    market, so allowing the inner task to call ``self.retry()`` can schedule an
    orphan retry with default arguments.
    """
    from .workload_coordination import disable_serialized_market_workload

    task = calculate_daily_group_rankings
    transient_token = _PROPAGATE_IN_PROCESS_TRANSIENT_ERRORS.set(True)
    try:
        if str(getattr(task, "__module__", "")).startswith("unittest.mock"):
            return task(market=market, activity_lifecycle=activity_lifecycle)
        with disable_serialized_market_workload():
            if hasattr(task, "request") and callable(getattr(task, "run", None)):
                return task.run(market=market, activity_lifecycle=activity_lifecycle)
            return task(market=market, activity_lifecycle=activity_lifecycle)
    finally:
        _PROPAGATE_IN_PROCESS_TRANSIENT_ERRORS.reset(transient_token)


@celery_app.task(
    bind=True,
    name='app.tasks.group_rank_tasks.calculate_daily_group_rankings_with_gapfill',
    soft_time_limit=3600,
    max_retries=2,
)
@serialized_market_workload('calculate_daily_group_rankings_with_gapfill')
def calculate_daily_group_rankings_with_gapfill(
    self,
    max_gap_days: int | None = None,
    market: str | None = None,
    activity_lifecycle: str | None = None,
):
    """
    Calculate daily group rankings with automatic gap detection and filling.

    This wrapper task:
    1. Detects missing trading days in the lookback window
    2. Fills gaps (oldest to newest)
    3. Calculates today's ranking only if today is a trading day for the market

    Mirrors the pattern of ``calculate_daily_breadth_with_gapfill`` so a fresh
    bootstrap on a non-trading day still produces ranking data for recent
    trading days, instead of short-circuiting on the trading-day guard.

    Args:
        max_gap_days: Maximum days to look back for gaps (default from settings)
        market: Market code (default "US")
        activity_lifecycle: Lifecycle tag for market activity tracking
    """
    from .market_queues import market_tag, log_extra, normalize_market
    from ..services.ibd_industry_service import IBDIndustryService
    from ..services.runtime_preferences_service import is_market_enabled_now
    _log_extra = log_extra(market)
    effective_market = normalize_market(market) if market is not None else "US"
    activity_lifecycle = activity_lifecycle or "daily_refresh"
    logger.info("=" * 60)
    logger.info(
        "TASK: Calculate Daily Group Rankings (with Gap-Fill) %s", market_tag(market),
        extra=_log_extra,
    )
    logger.info("=" * 60)
    if market is not None and not is_market_enabled_now(normalize_market(market)):
        logger.info(
            "Skipping group rankings (with gapfill) for disabled market %s", market,
            extra=_log_extra,
        )
        return {
            'status': 'skipped',
            'reason': f'market {effective_market} is disabled in local runtime preferences',
            'market': effective_market,
            'timestamp': datetime.now().isoformat(),
        }

    if max_gap_days is None:
        max_gap_days = settings.group_rank_gapfill_max_days

    db = SessionLocal()
    start_time = time.time()

    result = {
        'gap_fill': None,
        'today': None,
        'market': effective_market,
        'timestamp': datetime.now().isoformat(),
    }

    try:
        mark_market_activity_started(
            db,
            market=effective_market,
            stage_key="groups",
            lifecycle=activity_lifecycle,
            task_name=getattr(self, "name", "calculate_daily_group_rankings_with_gapfill"),
            task_id=getattr(getattr(self, "request", None), "id", None),
            message="Calculating group rankings (with gap-fill)",
        )

        # Defensive skip: non-US markets can be enabled before their CSV is
        # loaded, but missing US group mappings are a bootstrap failure.
        try:
            all_groups = IBDIndustryService.get_all_groups(db, market=effective_market)
        except TAXONOMY_UNAVAILABLE_EXCEPTIONS as e:
            if effective_market == "US":
                raise
            logger.warning(
                "Taxonomy unavailable for market=%s; skipping rankings: %s",
                effective_market,
                e,
                exc_info=True,
            )
            all_groups = []
        if not all_groups:
            if effective_market == "US":
                raise MissingIBDIndustryMappingsError()
            logger.info(
                "No industry groups available for market=%s; skipping rankings.",
                effective_market,
            )
            mark_market_activity_completed(
                db,
                market=effective_market,
                stage_key="groups",
                lifecycle=activity_lifecycle,
                task_name=getattr(self, "name", "calculate_daily_group_rankings_with_gapfill"),
                task_id=getattr(getattr(self, "request", None), "id", None),
                message="No taxonomy for market; skipped",
            )
            result['status'] = 'skipped'
            result['reason'] = 'no_taxonomy_for_market'
            result['total_duration_seconds'] = round(time.time() - start_time, 2)
            return result

        service = get_group_rank_service()

        if settings.group_rank_gapfill_enabled:
            logger.info(
                "Checking for group-ranking gaps in last %s days (market=%s)...",
                max_gap_days, effective_market,
            )
            missing_dates = service.find_missing_dates(
                db,
                lookback_days=max_gap_days,
                market=effective_market,
            )
            if missing_dates:
                logger.info(
                    "Found %d missing ranking dates (range %s to %s)",
                    len(missing_dates), missing_dates[0], missing_dates[-1],
                )
                gap_stats = service.fill_gaps_optimized(
                    db,
                    missing_dates,
                    market=effective_market,
                )
                result['gap_fill'] = gap_stats
                logger.info(
                    "✓ Gap-fill complete: %s processed, %s errors",
                    gap_stats.get('processed', 0), gap_stats.get('errors', 0),
                )
            else:
                logger.info("No missing ranking dates found - data is complete")
                result['gap_fill'] = {
                    'total_dates': 0,
                    'processed': 0,
                    'errors': 0,
                    'message': 'No gaps detected',
                }
        else:
            logger.info("Group ranking gap-fill disabled in settings, skipping detection")
            result['gap_fill'] = {'message': 'Gap-fill disabled'}

        calendar_service = get_market_calendar_service()
        today_local = calendar_service.market_now(effective_market).date()

        if calendar_service.is_trading_day(effective_market, today_local):
            logger.info(
                "Calculating group rankings for today (%s, %s)...",
                effective_market, today_local,
            )
            today_result = _calculate_daily_group_rankings_in_process(
                market=market,
                activity_lifecycle=activity_lifecycle,
            )
            result['today'] = today_result
            today_error = _group_rank_result_error(today_result)
            if today_error:
                raise RuntimeError(f"Daily group ranking failed: {today_error}")
        else:
            last_trading = calendar_service.last_completed_trading_day(effective_market)
            logger.info(
                "Today (%s) is not a trading day for %s. Skipping same-day calc.",
                today_local, effective_market,
            )
            result['today'] = {
                'skipped': True,
                'reason': f'{today_local} is not a trading day for {effective_market}',
                'last_trading_day': last_trading.strftime('%Y-%m-%d'),
                'date': today_local.strftime('%Y-%m-%d'),
                'timestamp': datetime.now().isoformat(),
            }

        total_duration = time.time() - start_time
        result['total_duration_seconds'] = round(total_duration, 2)

        logger.info("=" * 60)
        logger.info(f"✓ Group ranking orchestration complete in {total_duration:.2f}s")
        if result['gap_fill'] and result['gap_fill'].get('processed', 0) > 0:
            logger.info(f"  Gap-filled dates: {result['gap_fill']['processed']}")
        logger.info(f"  Today's result: {result['today'].get('date', 'N/A')}")
        logger.info("=" * 60)

        mark_market_activity_completed(
            db,
            market=effective_market,
            stage_key="groups",
            lifecycle=activity_lifecycle,
            task_name=getattr(self, "name", "calculate_daily_group_rankings_with_gapfill"),
            task_id=getattr(getattr(self, "request", None), "id", None),
            message="Group ranking orchestration completed",
        )
        return result

    except SoftTimeLimitExceeded:
        db.rollback()
        logger.error(
            "Soft time limit exceeded in calculate_daily_group_rankings_with_gapfill",
            exc_info=True,
        )
        _mark_market_activity_failed_safely(
            db,
            market=effective_market,
            stage_key="groups",
            lifecycle=activity_lifecycle,
            task_name=getattr(self, "name", "calculate_daily_group_rankings_with_gapfill"),
            task_id=getattr(getattr(self, "request", None), "id", None),
            message="Soft time limit exceeded",
        )
        raise
    except TRANSIENT_TASK_EXCEPTIONS as e:
        db.rollback()
        _mark_market_activity_failed_safely(
            db,
            market=effective_market,
            stage_key="groups",
            lifecycle=activity_lifecycle,
            task_name=getattr(self, "name", "calculate_daily_group_rankings_with_gapfill"),
            task_id=getattr(getattr(self, "request", None), "id", None),
            message=str(e),
        )
        _retry_transient_failure(self, "calculate_daily_group_rankings_with_gapfill", e)
    except Retry as e:
        db.rollback()
        logger.warning(
            "Retry requested from calculate_daily_group_rankings_with_gapfill: %s",
            e,
        )
        raise
    except Exception as e:
        db.rollback()
        logger.error(
            "✗ Error in calculate_daily_group_rankings_with_gapfill: %s", e,
            exc_info=True,
        )
        logger.info("=" * 60)
        _mark_market_activity_failed_safely(
            db,
            market=effective_market,
            stage_key="groups",
            lifecycle=activity_lifecycle,
            task_name=getattr(self, "name", "calculate_daily_group_rankings_with_gapfill"),
            task_id=getattr(getattr(self, "request", None), "id", None),
            message=str(e),
        )
        return {
            'error': str(e),
            'gap_fill': result.get('gap_fill'),
            'today': result.get('today'),
            'market': effective_market,
            'timestamp': datetime.now().isoformat(),
        }
    finally:
        db.close()


@celery_app.task(bind=True, name='app.tasks.group_rank_tasks.backfill_group_rankings')
@serialized_market_workload('backfill_group_rankings')
def backfill_group_rankings(self, start_date: str, end_date: str, market: str = "US"):
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
        result = service.backfill_rankings_optimized(db, start, end, market=market)

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
@serialized_market_workload('gapfill_group_rankings')
def gapfill_group_rankings(self, max_days: int = 365, market: str = "US"):
    """
    Detect and fill gaps in group ranking data (optimized version).

    This optimized gap-fill:
    1. Uses same universe as bulk scans (intersection of IBD groups and stock_universe)
    2. Pre-fetches all data once for efficiency
    3. Processes all missing dates with cached data

    Serialization with other same-market write workloads is handled by the
    market workload lease and the market-jobs queue family.

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
        missing_dates = service.find_missing_dates(
            db,
            lookback_days=max_days,
            market=market,
        )

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
        result = service.fill_gaps_optimized(db, missing_dates, market=market)

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
@serialized_market_workload('backfill_group_rankings_1year')
def backfill_group_rankings_1year(self, market: str = "US"):
    """
    One-time task to backfill 1 year of group rankings (optimized version).

    This optimized backfill:
    1. Uses same universe as bulk scans (intersection of IBD groups and stock_universe)
    2. Deletes existing rankings and recalculates (no skipping)
    3. Pre-fetches all data once for efficiency

    Returns:
        Dict with backfill statistics
    """
    from .market_queues import normalize_market

    effective_market = normalize_market(market)

    logger.info("=" * 60)
    logger.info(
        "TASK: 1-Year Backfill IBD Group Rankings (Optimized) (%s)",
        effective_market,
    )
    logger.info("=" * 60)

    db = SessionLocal()
    start_time = time.time()

    try:
        service = get_group_rank_service()

        # Calculate date range in the target market's local calendar.
        calendar_service = get_market_calendar_service()
        end_date = calendar_service.market_now(effective_market).date()
        start_date = end_date - timedelta(days=365)

        # Use optimized backfill (deletes existing, pre-fetches all data, uses validated universe)
        result = service.backfill_rankings_optimized(
            db,
            start_date,
            end_date,
            market=effective_market,
        )

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
