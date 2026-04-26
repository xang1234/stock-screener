"""Local-default bootstrap orchestration tasks."""

from __future__ import annotations

import logging
from typing import Iterable

from celery import chain

from ..database import SessionLocal
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


def _build_market_bootstrap_signatures(market: str) -> list:
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

    market_code = str(market).upper()
    signatures = [
        (
            refresh_stock_universe.si(
                market=market_code,
                activity_lifecycle="bootstrap",
            )
            if market_code == "US"
            else refresh_official_market_universe.si(
                market=market_code,
                activity_lifecycle="bootstrap",
            )
        ).set(queue=data_fetch_queue_for_market(market_code)),
    ]
    if market_code == "US":
        signatures.append(
            load_tracked_ibd_industry_groups.si(
                market=market_code,
                activity_lifecycle="bootstrap",
            ).set(queue=market_jobs_queue_for_market(market_code))
        )
    signatures.extend([
        smart_refresh_cache.si(
            mode="full",
            market=market_code,
            activity_lifecycle="bootstrap",
        ).set(queue=data_fetch_queue_for_market(market_code)),
        refresh_all_fundamentals.si(
            market=market_code,
            activity_lifecycle="bootstrap",
        ).set(queue=data_fetch_queue_for_market(market_code)),
        calculate_daily_breadth_with_gapfill.si(
            market=market_code,
            activity_lifecycle="bootstrap",
        ).set(queue=market_jobs_queue_for_market(market_code)),
        calculate_daily_group_rankings_with_gapfill.si(
            market=market_code,
            activity_lifecycle="bootstrap",
        ).set(queue=market_jobs_queue_for_market(market_code)),
        build_daily_snapshot.si(
            market=market_code,
            universe_name=_bootstrap_universe_name(market_code),
            publish_pointer_key=f"latest_published_market:{market_code}",
            activity_lifecycle="bootstrap",
        ).set(queue=market_jobs_queue_for_market(market_code)),
    ])
    return signatures


def queue_local_runtime_bootstrap(*, primary_market: str, enabled_markets: Iterable[str]) -> str:
    primary = str(primary_market).upper()
    enabled = []
    for market in [primary, *[str(item).upper() for item in enabled_markets]]:
        if market not in enabled:
            enabled.append(market)

    signatures = []
    for market in enabled:
        signatures.extend(_build_market_bootstrap_signatures(market))
    signatures.append(
        complete_local_runtime_bootstrap.si(
            primary_market=primary,
            enabled_markets=enabled,
        ).set(queue="celery")
    )
    workflow = chain(*signatures)
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
    from ..services.runtime_preferences_service import (
        _has_completed_auto_scan,
        set_bootstrap_state,
    )

    db = SessionLocal()
    try:
        missing_markets = [
            str(market).upper()
            for market in enabled_markets
            if not _has_completed_auto_scan(db, str(market).upper())
        ]
        if missing_markets:
            set_bootstrap_state(db, "failed")
            for market in missing_markets:
                mark_market_activity_failed(
                    db,
                    market=market,
                    stage_key="scan",
                    lifecycle="bootstrap",
                    task_name="runtime_bootstrap",
                    task_id=None,
                    message="Bootstrap scan did not publish",
                )
            raise RuntimeError(
                "Bootstrap completed without published auto scans for: "
                + ", ".join(missing_markets)
            )
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
