"""Manual and administrative IBD group-ranking backfill tasks."""

import logging
import time
from datetime import datetime, timedelta

from ..celery_app import celery_app
from ..database import SessionLocal
from ..wiring.bootstrap import (
    get_group_rank_service,
    get_market_calendar_service,
)
from .workload_coordination import serialized_market_workload


logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="app.tasks.group_rank_tasks.backfill_group_rankings",
)
@serialized_market_workload("backfill_group_rankings")
def backfill_group_rankings(
    self,
    start_date: str,
    end_date: str,
    market: str = "US",
):
    """Backfill historical group rankings for a date range."""
    logger.info("=" * 60)
    logger.info("TASK: Backfill IBD Group Rankings (Optimized)")
    logger.info("Date range: %s to %s", start_date, end_date)
    logger.info("=" * 60)

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        if start > end:
            logger.error("Start date must be before end date")
            return {
                "error": "Invalid date range: start_date > end_date",
                "timestamp": datetime.now().isoformat(),
            }
    except ValueError as exc:
        logger.error("Invalid date format: %s", exc)
        return {
            "error": f"Invalid date format. Use YYYY-MM-DD: {exc}",
            "timestamp": datetime.now().isoformat(),
        }

    db = SessionLocal()
    start_time = time.time()
    try:
        service = get_group_rank_service()
        result = service.backfill_rankings_optimized(
            db,
            start,
            end,
            market=market,
        )
        duration = time.time() - start_time

        from ..services.group_rankings_cache import (
            bump_group_rankings_epoch,
        )

        bump_group_rankings_epoch(market)
        try:
            from ..services.ui_snapshot_service import (
                safe_publish_groups_bootstrap,
            )

            safe_publish_groups_bootstrap()
        except Exception as snapshot_error:
            logger.warning(
                "Group rankings snapshot publish failed after backfill: %s",
                snapshot_error,
            )

        return {
            "start_date": start_date,
            "end_date": end_date,
            "total_dates": result["total_dates"],
            "deleted": result.get("deleted", 0),
            "processed": result["processed"],
            "skipped": result["skipped"],
            "errors": result["errors"],
            "total_duration_seconds": round(duration, 2),
            "avg_duration_per_day": round(
                duration / max(result["processed"], 1),
                2,
            ),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        db.rollback()
        logger.error(
            "Error in backfill_group_rankings task: %s",
            exc,
            exc_info=True,
        )
        return {
            "error": str(exc),
            "timestamp": datetime.now().isoformat(),
        }
    finally:
        db.close()


@celery_app.task(
    bind=True,
    name="app.tasks.group_rank_tasks.gapfill_group_rankings",
)
@serialized_market_workload("gapfill_group_rankings")
def gapfill_group_rankings(
    self,
    max_days: int = 365,
    market: str = "US",
):
    """Detect and fill gaps in group ranking data."""
    logger.info("=" * 60)
    logger.info("TASK: Gap-Fill IBD Group Rankings (Optimized)")
    logger.info("Looking back %s days", max_days)
    logger.info("=" * 60)

    db = SessionLocal()
    start_time = time.time()
    try:
        service = get_group_rank_service()
        missing_dates = service.find_missing_dates(
            db,
            lookback_days=max_days,
            market=market,
        )
        if not missing_dates:
            logger.info("No gaps found - data is complete")
            return {
                "status": "complete",
                "gaps_found": 0,
                "message": "No gaps to fill",
                "timestamp": datetime.now().isoformat(),
            }

        result = service.fill_gaps_optimized(
            db,
            missing_dates,
            market=market,
        )
        duration = time.time() - start_time

        from ..services.group_rankings_cache import (
            bump_group_rankings_epoch,
        )

        bump_group_rankings_epoch(market)
        try:
            from ..services.ui_snapshot_service import (
                safe_publish_groups_bootstrap,
            )

            safe_publish_groups_bootstrap()
        except Exception as snapshot_error:
            logger.warning(
                "Group rankings snapshot publish failed after gapfill: %s",
                snapshot_error,
            )

        return {
            "status": "complete",
            "gaps_found": len(missing_dates),
            "processed": result["processed"],
            "errors": result["errors"],
            "total_duration_seconds": round(duration, 2),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        db.rollback()
        logger.error(
            "Error in gapfill_group_rankings task: %s",
            exc,
            exc_info=True,
        )
        return {
            "status": "error",
            "error": str(exc),
            "timestamp": datetime.now().isoformat(),
        }
    finally:
        db.close()


@celery_app.task(
    bind=True,
    name="app.tasks.group_rank_tasks.backfill_group_rankings_1year",
)
@serialized_market_workload("backfill_group_rankings_1year")
def backfill_group_rankings_1year(
    self,
    market: str = "US",
):
    """Backfill one year of group rankings for a market."""
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
        calendar_service = get_market_calendar_service()
        end_date = calendar_service.market_now(effective_market).date()
        start_date = end_date - timedelta(days=365)
        result = service.backfill_rankings_optimized(
            db,
            start_date,
            end_date,
            market=effective_market,
        )
        duration = time.time() - start_time

        try:
            from ..services.ui_snapshot_service import (
                safe_publish_groups_bootstrap,
            )

            safe_publish_groups_bootstrap()
        except Exception as snapshot_error:
            logger.warning(
                "Group rankings snapshot publish failed after "
                "1-year backfill: %s",
                snapshot_error,
            )

        result["total_duration_seconds"] = round(duration, 2)
        result["timestamp"] = datetime.now().isoformat()
        return result
    except Exception as exc:
        db.rollback()
        logger.error(
            "Error in backfill_group_rankings_1year: %s",
            exc,
            exc_info=True,
        )
        return {
            "error": str(exc),
            "timestamp": datetime.now().isoformat(),
        }
    finally:
        db.close()
