"""Local-default bootstrap orchestration tasks."""

from __future__ import annotations

import logging
from typing import Iterable

from celery import chain

from ..domain.scanning.defaults import get_default_scan_profile
from ..database import SessionLocal
from ..services.market_activity_service import mark_market_activity_queued
from ..tasks.market_queues import SHARED_DATA_FETCH_QUEUE
from ..celery_app import celery_app

logger = logging.getLogger(__name__)


def _bootstrap_universe_name(market: str) -> str:
    return f"market:{market.upper()}"


def _build_market_bootstrap_signatures(
    market: str,
    *,
    include_initial_scan: bool = False,
) -> list:
    from app.interfaces.tasks.feature_store_tasks import build_daily_snapshot
    from app.tasks.breadth_tasks import calculate_daily_breadth_with_gapfill
    from app.tasks.cache_tasks import smart_refresh_cache
    from app.tasks.fundamentals_tasks import refresh_all_fundamentals
    from app.tasks.group_rank_tasks import calculate_daily_group_rankings
    from app.tasks.universe_tasks import (
        refresh_official_market_universe,
        refresh_stock_universe,
    )

    signatures = [
        (
            refresh_stock_universe.si(
                market=market,
                activity_lifecycle="bootstrap",
            )
            if market == "US"
            else refresh_official_market_universe.si(
                market=market,
                activity_lifecycle="bootstrap",
            )
        ).set(queue=SHARED_DATA_FETCH_QUEUE),
        smart_refresh_cache.si(
            mode="full",
            market=market,
            activity_lifecycle="bootstrap",
        ).set(queue=SHARED_DATA_FETCH_QUEUE),
        refresh_all_fundamentals.si(
            market=market,
            activity_lifecycle="bootstrap",
        ).set(queue=SHARED_DATA_FETCH_QUEUE),
        calculate_daily_breadth_with_gapfill.si(
            market=market,
            activity_lifecycle="bootstrap",
        ).set(queue=SHARED_DATA_FETCH_QUEUE),
        calculate_daily_group_rankings.si(
            market=market,
            activity_lifecycle="bootstrap",
        ).set(queue=SHARED_DATA_FETCH_QUEUE),
    ]
    if market == "US":
        signatures.append(
            build_daily_snapshot.si(
                market=market,
                universe_name=_bootstrap_universe_name(market),
                publish_pointer_key=f"latest_published_market:{market}",
                activity_lifecycle="bootstrap",
            ).set(queue=SHARED_DATA_FETCH_QUEUE)
        )
    elif include_initial_scan:
        signatures.append(
            queue_market_bootstrap_scan.si(market=market).set(queue="celery")
        )
    return signatures


def queue_local_runtime_bootstrap(*, primary_market: str, enabled_markets: Iterable[str]) -> str:
    enabled = [str(market).upper() for market in enabled_markets]
    task = chain(
        *_build_market_bootstrap_signatures(primary_market, include_initial_scan=True),
        complete_local_runtime_bootstrap.si(
            primary_market=primary_market,
            enabled_markets=enabled,
        ).set(queue="celery"),
    ).apply_async()
    logger.info(
        "Queued local runtime bootstrap",
        extra={
            "primary_market": primary_market,
            "enabled_markets": enabled,
            "task_id": task.id,
        },
    )
    return task.id


@celery_app.task(
    name="app.tasks.runtime_bootstrap_tasks.queue_market_bootstrap_scan",
    queue="celery",
)
def queue_market_bootstrap_scan(market: str) -> dict:
    from ..infra.db.uow import SqlUnitOfWork
    from ..schemas.universe import Market, UniverseDefinition, UniverseType
    from ..use_cases.scanning.create_scan import ActiveScanConflictError, CreateScanCommand
    from ..wiring.bootstrap import get_create_scan_use_case

    market_code = str(market).upper()
    defaults = get_default_scan_profile()
    universe_def = UniverseDefinition(type=UniverseType.MARKET, market=Market(market_code))
    use_case = get_create_scan_use_case()
    uow = SqlUnitOfWork(SessionLocal)

    try:
        result = use_case.execute(
            uow,
            CreateScanCommand(
                universe_def=universe_def,
                universe_label=universe_def.label(),
                universe_key=universe_def.key(),
                universe_type=universe_def.type.value,
                universe_market=universe_def.market.value if universe_def.market else None,
                screeners=defaults["screeners"],
                composite_method=defaults["composite_method"],
                criteria=defaults["criteria"],
            ),
        )
    except ActiveScanConflictError as exc:
        logger.info(
            "Skipped bootstrap market scan for %s because another scan is active",
            market_code,
            extra={"active_scan_id": exc.active_scan.scan_id},
        )
        return {
            "status": "skipped",
            "reason": "scan_already_active",
            "market": market_code,
            "active_scan_id": exc.active_scan.scan_id,
        }

    logger.info(
        "Queued bootstrap market scan for %s",
        market_code,
        extra={"scan_id": result.scan_id, "status": result.status},
    )
    return {
        "status": result.status,
        "market": market_code,
        "scan_id": result.scan_id,
        "total_stocks": result.total_stocks,
    }


@celery_app.task(
    name="app.tasks.runtime_bootstrap_tasks.complete_local_runtime_bootstrap",
    queue="celery",
)
def complete_local_runtime_bootstrap(primary_market: str, enabled_markets: list[str]) -> dict:
    from ..services.runtime_preferences_service import set_bootstrap_state

    db = SessionLocal()
    try:
        set_bootstrap_state(db, "ready")
    finally:
        db.close()

    secondary_markets = [market for market in enabled_markets if market != primary_market]
    queued_secondary = []
    for market in secondary_markets:
        task = chain(*_build_market_bootstrap_signatures(market)).apply_async()
        mark_market_activity_queued(
            db,
            market=market,
            stage_key="universe",
            lifecycle="bootstrap",
            task_name="runtime_bootstrap",
            task_id=task.id,
            message=f"Queued bootstrap for {market}",
        )
        queued_secondary.append({"market": market, "task_id": task.id})

    logger.info(
        "Completed primary local runtime bootstrap",
        extra={
            "primary_market": primary_market,
            "secondary_markets": secondary_markets,
        },
    )
    return {
        "primary_market": primary_market,
        "secondary_markets": secondary_markets,
        "queued_secondary": queued_secondary,
    }
