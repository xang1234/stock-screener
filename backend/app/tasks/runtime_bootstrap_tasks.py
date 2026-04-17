"""Local-default bootstrap orchestration tasks."""

from __future__ import annotations

import logging
from typing import Iterable

from celery import chain

from ..database import SessionLocal
from ..tasks.market_queues import SHARED_DATA_FETCH_QUEUE
from ..celery_app import celery_app

logger = logging.getLogger(__name__)


def _bootstrap_universe_name(market: str) -> str:
    return f"market:{market.lower()}"


def _build_market_bootstrap_signatures(market: str) -> list:
    from app.interfaces.tasks.feature_store_tasks import build_daily_snapshot
    from app.tasks.breadth_tasks import calculate_daily_breadth_with_gapfill
    from app.tasks.cache_tasks import smart_refresh_cache
    from app.tasks.fundamentals_tasks import refresh_all_fundamentals
    from app.tasks.group_rank_tasks import calculate_daily_group_rankings
    from app.tasks.universe_tasks import (
        refresh_official_market_universe,
        refresh_stock_universe,
    )

    universe_task = (
        refresh_stock_universe.si(market=market)
        if market == "US"
        else refresh_official_market_universe.si(market=market)
    )
    return [
        universe_task.set(queue=SHARED_DATA_FETCH_QUEUE),
        smart_refresh_cache.si(mode="full", market=market).set(queue=SHARED_DATA_FETCH_QUEUE),
        refresh_all_fundamentals.si(market=market).set(queue=SHARED_DATA_FETCH_QUEUE),
        calculate_daily_breadth_with_gapfill.si(market=market).set(queue=SHARED_DATA_FETCH_QUEUE),
        calculate_daily_group_rankings.si(market=market).set(queue=SHARED_DATA_FETCH_QUEUE),
        build_daily_snapshot.si(
            market=market,
            universe_name=_bootstrap_universe_name(market),
            publish_pointer_key=f"latest_published_market:{market}",
        ).set(queue=SHARED_DATA_FETCH_QUEUE),
    ]


def queue_local_runtime_bootstrap(*, primary_market: str, enabled_markets: Iterable[str]) -> str:
    enabled = [str(market).upper() for market in enabled_markets]
    task = chain(
        *_build_market_bootstrap_signatures(primary_market),
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


@celery_app.task(name="app.tasks.runtime_bootstrap_tasks.complete_local_runtime_bootstrap")
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
