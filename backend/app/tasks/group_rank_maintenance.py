"""Maintenance workflows behind the public Group-ranking Celery tasks."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def _publish_groups_snapshot(*, context: str) -> None:
    try:
        from ..services.ui_snapshot_service import safe_publish_groups_bootstrap

        safe_publish_groups_bootstrap()
    except Exception as snapshot_error:
        logger.warning(
            "Group rankings snapshot publish failed after %s: %s",
            context,
            snapshot_error,
        )


def run_group_rank_backfill(
    *,
    start_date: str,
    end_date: str,
    market: str,
    session_factory,
    group_service_provider,
    calendar_service_provider,
    coordinate_group_dates,
    resolve_active_formula,
) -> dict:
    """Backfill a bounded date range through the formula-aware coordinator."""
    logger.info("TASK: Backfill IBD Group Rankings (Optimized)")
    logger.info("Date range: %s to %s", start_date, end_date)

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        if start > end:
            return {
                "error": "Invalid date range: start_date > end_date",
                "timestamp": datetime.now().isoformat(),
            }
    except ValueError as error:
        return {
            "error": f"Invalid date format. Use YYYY-MM-DD: {error}",
            "timestamp": datetime.now().isoformat(),
        }

    db = session_factory()
    started_at = time.time()
    try:
        service = group_service_provider()
        effective_market = str(market or "US").upper()
        resolved_formula = resolve_active_formula(
            db,
            market=effective_market,
            group_service=service,
        )
        calendar = calendar_service_provider()
        dates = []
        cursor = start
        while cursor <= end:
            if calendar.is_trading_day(effective_market, cursor):
                dates.append(cursor)
            cursor += timedelta(days=1)

        result = {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "deleted": 0,
            **coordinate_group_dates(
                db,
                dates=dates,
                market=effective_market,
                formula_version=resolved_formula,
            ),
        }
        duration = time.time() - started_at

        from ..services.group_rankings_cache import bump_group_rankings_epoch

        bump_group_rankings_epoch(market)
        _publish_groups_snapshot(context="backfill")
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
    except Exception as error:
        db.rollback()
        logger.error("Error in backfill_group_rankings task: %s", error, exc_info=True)
        return {"error": str(error), "timestamp": datetime.now().isoformat()}
    finally:
        db.close()


def run_group_rank_gapfill(
    *,
    max_days: int,
    market: str,
    session_factory,
    group_service_provider,
    coordinate_group_dates,
    resolve_active_formula,
) -> dict:
    """Find and fill missing Group ranking dates."""
    db = session_factory()
    started_at = time.time()
    try:
        service = group_service_provider()
        effective_market = str(market or "US").upper()
        resolved_formula = resolve_active_formula(
            db,
            market=effective_market,
            group_service=service,
        )
        missing_dates = service.find_missing_dates(
            db,
            lookback_days=max_days,
            market=market,
            formula_version=resolved_formula,
        )
        if not missing_dates:
            return {
                "status": "complete",
                "gaps_found": 0,
                "message": "No gaps to fill",
                "timestamp": datetime.now().isoformat(),
            }

        result = coordinate_group_dates(
            db,
            dates=missing_dates,
            market=effective_market,
            formula_version=resolved_formula,
        )
        duration = time.time() - started_at

        from ..services.group_rankings_cache import bump_group_rankings_epoch

        bump_group_rankings_epoch(market)
        _publish_groups_snapshot(context="gapfill")
        return {
            "status": "complete",
            "gaps_found": len(missing_dates),
            "processed": result["processed"],
            "errors": result["errors"],
            "total_duration_seconds": round(duration, 2),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as error:
        db.rollback()
        logger.error("Error in gapfill_group_rankings task: %s", error, exc_info=True)
        return {
            "status": "error",
            "error": str(error),
            "timestamp": datetime.now().isoformat(),
        }
    finally:
        db.close()


def run_one_year_group_rank_backfill(
    *,
    market: str,
    session_factory,
    group_service_provider,
    calendar_service_provider,
    coordinate_group_dates,
    resolve_active_formula,
    normalize_market,
) -> dict:
    """Backfill one calendar year for a normalized market."""
    effective_market = normalize_market(market)
    db = session_factory()
    started_at = time.time()
    try:
        service = group_service_provider()
        resolved_formula = resolve_active_formula(
            db,
            market=effective_market,
            group_service=service,
        )
        calendar = calendar_service_provider()
        end_date = calendar.market_now(effective_market).date()
        start_date = end_date - timedelta(days=365)
        dates = []
        cursor = start_date
        while cursor <= end_date:
            if calendar.is_trading_day(effective_market, cursor):
                dates.append(cursor)
            cursor += timedelta(days=1)

        result = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "deleted": 0,
            **coordinate_group_dates(
                db,
                dates=dates,
                market=effective_market,
                formula_version=resolved_formula,
            ),
        }
        result["total_duration_seconds"] = round(time.time() - started_at, 2)
        result["timestamp"] = datetime.now().isoformat()
        _publish_groups_snapshot(context="1-year backfill")
        return result
    except Exception as error:
        db.rollback()
        logger.error("Error in backfill_group_rankings_1year: %s", error, exc_info=True)
        return {"error": str(error), "timestamp": datetime.now().isoformat()}
    finally:
        db.close()
