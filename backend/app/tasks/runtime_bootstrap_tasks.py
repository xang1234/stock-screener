"""Local-default bootstrap orchestration tasks."""

from __future__ import annotations

import logging
from typing import Iterable

from celery import chain, chord, group

from ..database import SessionLocal
from ..domain.bootstrap.plan import (
    BootstrapQueueKind,
    MarketBootstrapPlan,
    build_bootstrap_plan,
)
from ..services.market_activity_service import (
    mark_current_market_activity_failed,
    mark_market_activity_failed,
)
from ..tasks.market_queues import (
    data_fetch_queue_for_market,
    market_jobs_queue_for_market,
)
from ..celery_app import celery_app

logger = logging.getLogger(__name__)


def _bootstrap_universe_name(market: str) -> str:
    return f"market:{market.upper()}"


def _queue_for_stage(stage) -> str:
    if stage.queue_kind == BootstrapQueueKind.DATA_FETCH:
        return data_fetch_queue_for_market(stage.kwargs["market"])
    if stage.queue_kind == BootstrapQueueKind.MARKET_JOBS:
        return market_jobs_queue_for_market(stage.kwargs["market"])
    raise ValueError(f"Unsupported bootstrap queue kind: {stage.queue_kind}")


def _build_market_bootstrap_signatures(market_plan: MarketBootstrapPlan) -> list:
    from app.interfaces.tasks.feature_store_tasks import build_daily_snapshot
    from app.tasks.breadth_tasks import calculate_daily_breadth_with_gapfill
    from app.tasks.cache_tasks import smart_refresh_cache
    from app.tasks.fundamentals_tasks import refresh_all_fundamentals
    from app.tasks.group_rank_tasks import calculate_daily_group_rankings_with_gapfill
    from app.tasks.industry_tasks import load_tracked_ibd_industry_groups
    from app.tasks.universe_tasks import (
        refresh_official_market_universe,
        refresh_stock_universe,
    )

    task_by_name = {
        "refresh_stock_universe": refresh_stock_universe,
        "refresh_official_market_universe": refresh_official_market_universe,
        "load_tracked_ibd_industry_groups": load_tracked_ibd_industry_groups,
        "smart_refresh_cache": smart_refresh_cache,
        "refresh_all_fundamentals": refresh_all_fundamentals,
        "calculate_daily_breadth_with_gapfill": calculate_daily_breadth_with_gapfill,
        "calculate_daily_group_rankings_with_gapfill": calculate_daily_group_rankings_with_gapfill,
        "build_daily_snapshot": build_daily_snapshot,
    }
    return [
        task_by_name[stage.task_name]
        .si(**stage.kwargs)
        .set(queue=_queue_for_stage(stage))
        for stage in market_plan.stages
    ]


def queue_local_runtime_bootstrap(*, primary_market: str, enabled_markets: Iterable[str]) -> str:
    plan = build_bootstrap_plan(
        primary_market=primary_market,
        enabled_markets=enabled_markets,
    )
    primary = plan.primary_market
    enabled = list(plan.enabled_markets)

    market_workflows = []
    for market_plan in plan.market_plans:
        market_workflows.append(
            chain(*_build_market_bootstrap_signatures(market_plan))
        )
    completion = complete_local_runtime_bootstrap.si(
        primary_market=primary,
        enabled_markets=enabled,
    ).set(queue="celery")
    workflow = chord(group(market_workflows), completion)
    errback = fail_local_runtime_bootstrap.s(
        primary_market=primary,
        enabled_markets=enabled,
    ).set(queue="celery")
    try:
        task = workflow.apply_async(link_error=errback)
    except TypeError:
        task = workflow.apply_async()
    logger.info(
        "Queued local runtime bootstrap",
        extra={
            "primary_market": primary,
            "enabled_markets": enabled,
            "task_id": task.id,
        },
    )
    return task.id


@celery_app.task(
    name="app.tasks.runtime_bootstrap_tasks.complete_local_runtime_bootstrap",
    queue="celery",
)
def complete_local_runtime_bootstrap(primary_market: str, enabled_markets: list[str]) -> dict:
    from ..services.bootstrap_readiness_service import BootstrapReadinessService
    from ..services.runtime_preferences_service import (
        get_runtime_preferences,
        set_bootstrap_state,
    )

    db = SessionLocal()
    try:
        prefs = get_runtime_preferences(db)
        readiness = BootstrapReadinessService().evaluate(
            db,
            enabled_markets=enabled_markets,
            bootstrap_started_at=prefs.bootstrap_started_at,
        )
        missing_markets = readiness.missing_markets
        if missing_markets:
            set_bootstrap_state(db, "failed")
            core_missing_markets = []
            scan_missing_markets = []
            for market in missing_markets:
                market_result = readiness.market_results[market]
                if not market_result.core_ready:
                    core_missing_markets.append(market)
                    stage_key = "core"
                    message = "Bootstrap core data incomplete"
                else:
                    scan_missing_markets.append(market)
                    stage_key = "scan"
                    message = "Bootstrap scan did not publish"
                mark_market_activity_failed(
                    db,
                    market=market,
                    stage_key=stage_key,
                    lifecycle="bootstrap",
                    task_name="runtime_bootstrap",
                    task_id=None,
                    message=message,
                )
            failure_reasons = []
            if core_missing_markets:
                failure_reasons.append(
                    "missing core market data for: " + ", ".join(core_missing_markets)
                )
            if scan_missing_markets:
                failure_reasons.append(
                    "missing published auto scans for: " + ", ".join(scan_missing_markets)
                )
            return {
                "status": "failed",
                "primary_market": primary_market,
                "enabled_markets": enabled_markets,
                "reason": "; ".join(failure_reasons),
            }
        set_bootstrap_state(db, "ready")
    finally:
        db.close()

    logger.info(
        "Completed local runtime bootstrap",
        extra={
            "primary_market": primary_market,
            "enabled_markets": enabled_markets,
        },
    )
    return {
        "primary_market": primary_market,
        "enabled_markets": enabled_markets,
    }


@celery_app.task(
    name="app.tasks.runtime_bootstrap_tasks.fail_local_runtime_bootstrap",
    queue="celery",
)
def fail_local_runtime_bootstrap(*args, primary_market: str, enabled_markets: list[str], **kwargs) -> dict:
    from ..services.runtime_preferences_service import set_bootstrap_state

    db = SessionLocal()
    try:
        set_bootstrap_state(db, "failed")
        for market in enabled_markets:
            mark_current_market_activity_failed(
                db,
                market=str(market).upper(),
                lifecycle="bootstrap",
                message="Bootstrap failed",
            )
    finally:
        db.close()

    logger.warning(
        "Marked local runtime bootstrap failed",
        extra={
            "primary_market": primary_market,
            "enabled_markets": enabled_markets,
            "callback_args": args,
            "callback_kwargs": kwargs,
        },
    )
    return {
        "status": "failed",
        "primary_market": primary_market,
        "enabled_markets": enabled_markets,
    }
