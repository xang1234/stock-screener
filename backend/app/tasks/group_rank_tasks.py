"""Scheduled IBD group-ranking tasks serialized by market workload."""
from contextlib import contextmanager
from contextvars import ContextVar
import logging
from datetime import datetime, date
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
from ..services.daily_group_rank_runner import (
    DailyGroupRankDependencies,
    DailyGroupRankRequest,
    GroupRankWarmupIncomplete,
    NoGroupRankingsCalculated,
    run_daily_group_rankings,
)
from ..services.derived_data_execution_policy import (
    resolve_derived_data_execution_policy,
)
from ..services.ibd_group_rank_service import (
    IncompleteGroupRankingCacheError,
    MissingIBDIndustryMappingsError,
)
from ..services.market_taxonomy_service import TaxonomyLoadError
from ..wiring.bootstrap import get_group_rank_service, get_market_calendar_service
from .date_resolution import resolve_task_target_date
from .group_rank_memory import release_group_rank_gapfill_memory as _release_group_rank_gapfill_memory
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


def _repair_current_us_group_metadata(calculation_date: date):
    from ..interfaces.tasks.feature_store_tasks import (
        _repair_current_us_group_metadata as repair,
    )

    return repair(ranking_date=calculation_date)


def _daily_group_rank_dependencies(
    service,
) -> DailyGroupRankDependencies:
    from ..services.group_rankings_cache import bump_group_rankings_epoch
    from ..services.ui_snapshot_service import safe_publish_groups_bootstrap

    return DailyGroupRankDependencies(
        service=service,
        bump_epoch=bump_group_rankings_epoch,
        publish_snapshot=safe_publish_groups_bootstrap,
        repair_current_us_metadata=_repair_current_us_group_metadata,
    )


def _run_daily_group_rankings_response(
    db,
    request: DailyGroupRankRequest,
    dependencies: DailyGroupRankDependencies,
) -> dict:
    policy = request.policy
    try:
        outcome = run_daily_group_rankings(
            db,
            request,
            dependencies,
        )
    except GroupRankWarmupIncomplete as exc:
        result = {
            "error": str(exc),
            "reason_code": GroupRankReasonCode.WARMUP_INCOMPLETE,
            "date": request.calculation_date.isoformat(),
            "timestamp": datetime.now().isoformat(),
            "cache_only": True,
        }
        return policy.annotate_response(result)
    except IncompleteGroupRankingCacheError as exc:
        result = {
            "error": str(exc),
            "reason_code": GroupRankReasonCode.WARMUP_INCOMPLETE,
            "date": request.calculation_date.isoformat(),
            "timestamp": datetime.now().isoformat(),
            "cache_only": True,
            "prefetch_stats": exc.stats.to_dict(),
        }
        return policy.annotate_response(result)
    except MissingIBDIndustryMappingsError as exc:
        result = {
            "error": str(exc),
            "reason_code": GroupRankReasonCode.MISSING_IBD_MAPPINGS,
            "date": request.calculation_date.isoformat(),
            "timestamp": datetime.now().isoformat(),
        }
        return policy.annotate_response(
            result,
            include_cache_only=True,
        )
    except NoGroupRankingsCalculated as exc:
        result = {
            "date": request.calculation_date.isoformat(),
            "groups_ranked": 0,
            "warning": "No groups could be ranked",
            "error": str(exc),
            "reason_code": GroupRankReasonCode.NO_GROUPS_RANKED,
            "calculation_duration_seconds": round(
                exc.duration_seconds,
                2,
            ),
            "timestamp": datetime.now().isoformat(),
        }
        policy.annotate_response(result)
        if policy.response_cache_policy is not None:
            result["prefetch_stats"] = exc.prefetch_stats.to_dict()
        return result
    return outcome.to_task_result(policy)


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
    refresh_guarded_cache_only: bool = False,
    market: str | None = None,
    activity_lifecycle: str | None = None,
    execution_policy: str | None = None,
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
    policy = None

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
        service = get_group_rank_service()
        policy = resolve_derived_data_execution_policy(
            execution_policy=execution_policy,
            force_cache_only=force_cache_only,
            refresh_guarded_cache_only=refresh_guarded_cache_only,
            target_date=calc_date,
            current_date=today_local,
            allow_same_day_warmup_bypass=(
                _ALLOW_SAME_DAY_WARMUP_BYPASS.get()
            ),
        )
        request = DailyGroupRankRequest(
            calculation_date=calc_date,
            current_date=today_local,
            market=effective_market,
            activity_lifecycle=activity_lifecycle,
            policy=policy,
        )
        task_result = _run_daily_group_rankings_response(
            db,
            request,
            _daily_group_rank_dependencies(service),
        )
        if task_result.get("error"):
            db.rollback()
            _mark_market_activity_failed_safely(
                db,
                market=effective_market,
                stage_key="groups",
                lifecycle=activity_lifecycle,
                task_name=getattr(self, "name", "calculate_daily_group_rankings"),
                task_id=getattr(getattr(self, "request", None), "id", None),
                message=str(task_result["error"]),
            )
            return task_result

        groups_ranked = int(task_result["groups_ranked"])
        mark_market_activity_completed(
            db,
            market=effective_market,
            stage_key="groups",
            lifecycle=activity_lifecycle,
            task_name=getattr(self, "name", "calculate_daily_group_rankings"),
            task_id=getattr(getattr(self, "request", None), "id", None),
            current=groups_ranked,
            total=groups_ranked,
            message="Group rankings completed",
        )
        return task_result

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
    except TRANSIENT_TASK_EXCEPTIONS as e:
        db.rollback()
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
        error_result = {
            'error': str(e),
            'reason_code': GroupRankReasonCode.UNKNOWN,
            'date': calc_date.strftime('%Y-%m-%d') if calc_date else None,
            'timestamp': datetime.now().isoformat()
        }
        if policy is not None:
            policy.annotate_response(error_result)
        return error_result

    finally:
        db.close()


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
    calculation_date: str | None = None,
    refresh_guarded_cache_only: bool = False,
    execution_policy: str | None = None,
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
    policy = None

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

        calendar_service = get_market_calendar_service()
        resolved_date = resolve_task_target_date(
            calculation_date,
            market=effective_market,
            calendar_service=calendar_service,
        )
        target_date = resolved_date.target_date
        current_date = calendar_service.market_now(effective_market).date()
        policy = resolve_derived_data_execution_policy(
            execution_policy=execution_policy,
            refresh_guarded_cache_only=refresh_guarded_cache_only,
            target_date=target_date,
            current_date=current_date,
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
        gap_policy = policy.for_gap_fill()
        policy.annotate_response(result)

        if settings.group_rank_gapfill_enabled:
            logger.info(
                "Checking for group-ranking gaps in last %s days (market=%s)...",
                max_gap_days, effective_market,
            )
            missing_dates = service.find_missing_dates(
                db,
                lookback_days=max_gap_days,
                market=effective_market,
                end_date=target_date,
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
                    policy=gap_policy,
                )
                result['gap_fill'] = gap_stats
                logger.info(
                    "✓ Gap-fill complete: %s processed, %s errors",
                    gap_stats.get('processed', 0), gap_stats.get('errors', 0),
                )
                _release_group_rank_gapfill_memory()
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

        if calendar_service.is_trading_day(effective_market, target_date):
            logger.info(
                "Calculating group rankings for %s (%s)...",
                effective_market, target_date,
            )
            today_result = _run_daily_group_rankings_response(
                db,
                DailyGroupRankRequest(
                    calculation_date=target_date,
                    current_date=current_date,
                    market=effective_market,
                    activity_lifecycle=activity_lifecycle,
                    policy=policy,
                ),
                _daily_group_rank_dependencies(service),
            )
            result['today'] = today_result
            today_error = None
            if isinstance(today_result, dict) and today_result.get("error"):
                today_error = str(today_result["error"])
            if today_error:
                raise RuntimeError(f"Daily group ranking failed: {today_error}")
        else:
            last_trading = calendar_service.last_completed_trading_day(effective_market)
            logger.info(
                "Today (%s) is not a trading day for %s. Skipping same-day calc.",
                target_date, effective_market,
            )
            result['today'] = {
                'skipped': True,
                'reason': f'{target_date} is not a trading day for {effective_market}',
                'last_trading_day': last_trading.strftime('%Y-%m-%d'),
                'date': target_date.strftime('%Y-%m-%d'),
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
        error_result = {
            'error': str(e),
            'gap_fill': result.get('gap_fill'),
            'today': result.get('today'),
            'market': effective_market,
            'timestamp': datetime.now().isoformat(),
        }
        if policy is not None:
            policy.annotate_response(error_result)
        return error_result
    finally:
        db.close()


from .group_rank_backfill_tasks import (
    backfill_group_rankings,
    backfill_group_rankings_1year,
    gapfill_group_rankings,
)
