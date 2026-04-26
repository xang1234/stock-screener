"""Per-market daily refresh pipeline orchestration."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from celery import chain

from app.celery_app import celery_app
from app.services.market_calendar_service import MarketCalendarService
from app.tasks.market_queues import (
    data_fetch_queue_for_market,
    market_jobs_queue_for_market,
    normalize_market,
)

logger = logging.getLogger(__name__)

_MIN_DAILY_PRICE_REFRESH_SUCCESS_RATE = 0.90


def _normalize_pipeline_market(market: str | None) -> str:
    normalized = normalize_market(market)
    if normalized == "SHARED":
        raise ValueError("Daily market pipelines require an explicit market")
    return normalized


def _daily_pipeline_universe_name(market: str) -> str:
    return f"market:{market.upper()}"


def _result_failed(result: Any) -> bool:
    if not isinstance(result, dict):
        return True
    if result.get("error") or result.get("reason") == "not_trading_day":
        return True
    if result.get("skipped") is True:
        return True
    return str(result.get("status", "")).lower() in {
        "already_running",
        "failed",
        "partial",
        "skipped",
    }


def _partial_price_refresh_meets_minimum(result: dict) -> bool:
    try:
        refreshed = float(result.get("refreshed", 0))
        total = float(result.get("total", 0))
    except (TypeError, ValueError):
        return False
    return total > 0 and refreshed / total >= _MIN_DAILY_PRICE_REFRESH_SUCCESS_RATE


def _price_refresh_failed(result: Any) -> bool:
    if not isinstance(result, dict):
        return True
    status = str(result.get("status", "")).lower()
    if status == "partial":
        if result.get("error") or result.get("reason") == "not_trading_day":
            return True
        if result.get("skipped") is True:
            return True
        return not _partial_price_refresh_meets_minimum(result)
    return _result_failed(result)


@celery_app.task(
    name="app.tasks.daily_market_pipeline_tasks.guard_price_refresh",
    queue="celery",
)
def guard_price_refresh(result: dict | None = None, *, market: str) -> dict:
    if _price_refresh_failed(result):
        raise RuntimeError(f"Daily price refresh failed for {market}: {result}")
    return {"status": "ok", "market": market, "stage": "prices"}


@celery_app.task(
    name="app.tasks.daily_market_pipeline_tasks.guard_breadth_result",
    queue="celery",
)
def guard_breadth_result(result: dict | None = None, *, market: str) -> dict:
    if _result_failed(result):
        raise RuntimeError(f"Daily breadth calculation failed for {market}: {result}")
    return {"status": "ok", "market": market, "stage": "breadth"}


@celery_app.task(
    name="app.tasks.daily_market_pipeline_tasks.guard_group_result",
    queue="celery",
)
def guard_group_result(result: dict | None = None, *, market: str) -> dict:
    if _result_failed(result):
        raise RuntimeError(f"Daily group ranking failed for {market}: {result}")
    return {"status": "ok", "market": market, "stage": "groups"}


@celery_app.task(
    name="app.tasks.daily_market_pipeline_tasks.guard_snapshot_result",
    queue="celery",
)
def guard_snapshot_result(result: dict | None = None, *, market: str) -> dict:
    if isinstance(result, dict) and result.get("auto_scan_id"):
        return {
            "status": "ok",
            "market": market,
            "stage": "scan",
            "auto_scan_id": result["auto_scan_id"],
        }
    raise RuntimeError(f"Daily market scan did not publish for {market}: {result}")


def _build_daily_market_pipeline_signatures(market: str, trading_date: date) -> list:
    from app.interfaces.tasks.feature_store_tasks import build_daily_snapshot
    from app.tasks.breadth_tasks import calculate_daily_breadth_with_gapfill
    from app.tasks.cache_tasks import smart_refresh_cache
    from app.tasks.group_rank_tasks import calculate_daily_group_rankings_with_gapfill

    market_code = _normalize_pipeline_market(market)
    as_of_date = trading_date.isoformat()
    return [
        smart_refresh_cache.si(mode="full", market=market_code).set(
            queue=data_fetch_queue_for_market(market_code)
        ),
        guard_price_refresh.s(market=market_code).set(
            queue=market_jobs_queue_for_market(market_code)
        ),
        calculate_daily_breadth_with_gapfill.si(market=market_code).set(
            queue=market_jobs_queue_for_market(market_code)
        ),
        guard_breadth_result.s(market=market_code).set(
            queue=market_jobs_queue_for_market(market_code)
        ),
        calculate_daily_group_rankings_with_gapfill.si(market=market_code).set(
            queue=market_jobs_queue_for_market(market_code)
        ),
        guard_group_result.s(market=market_code).set(
            queue=market_jobs_queue_for_market(market_code)
        ),
        build_daily_snapshot.si(
            market=market_code,
            as_of_date_str=as_of_date,
            universe_name=_daily_pipeline_universe_name(market_code),
            publish_pointer_key=f"latest_published_market:{market_code}",
        ).set(queue=market_jobs_queue_for_market(market_code)),
        guard_snapshot_result.s(market=market_code).set(
            queue=market_jobs_queue_for_market(market_code)
        ),
    ]


def _market_pipeline_active(market: str) -> dict | None:
    try:
        from app.wiring.bootstrap import get_workload_coordination

        return get_workload_coordination().get_market_workload_holder(market)
    except Exception:
        logger.debug("Could not inspect market workload holder", exc_info=True)
        return None


@celery_app.task(
    bind=True,
    name="app.tasks.daily_market_pipeline_tasks.queue_daily_market_pipeline",
    queue="celery",
)
def queue_daily_market_pipeline(self, market: str) -> dict:
    from app.services.runtime_preferences_service import is_market_enabled_now

    market_code = _normalize_pipeline_market(market)
    if not is_market_enabled_now(market_code):
        return {
            "status": "skipped",
            "reason": f"market {market_code} is disabled in local runtime preferences",
            "market": market_code,
            "timestamp": datetime.now().isoformat(),
        }

    holder = _market_pipeline_active(market_code)
    if holder:
        return {
            "status": "skipped",
            "reason": "market_pipeline_already_active",
            "market": market_code,
            "active_task_name": holder.get("task_name"),
            "active_task_id": holder.get("task_id"),
            "timestamp": datetime.now().isoformat(),
        }

    calendar = MarketCalendarService()
    today = calendar.market_now(market_code).date()
    if not calendar.is_trading_day(market_code, today):
        return {
            "status": "skipped",
            "reason": "not_trading_day",
            "market": market_code,
            "as_of_date": today.isoformat(),
            "timestamp": datetime.now().isoformat(),
        }

    trading_day = calendar.last_completed_trading_day(market_code)
    task = chain(*_build_daily_market_pipeline_signatures(market_code, trading_day)).apply_async()
    logger.info(
        "Queued daily market pipeline",
        extra={"market": market_code, "task_id": task.id, "as_of_date": trading_day.isoformat()},
    )
    return {
        "status": "queued",
        "market": market_code,
        "as_of_date": trading_day.isoformat(),
        "task_id": task.id,
        "timestamp": datetime.now().isoformat(),
    }
