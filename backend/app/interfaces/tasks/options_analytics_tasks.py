"""Celery boundary for the bounded US Options Analytics refresh."""

from __future__ import annotations

import logging

from app.celery_app import celery_app
from app.config import settings
from app.database import SessionLocal
from app.domain.markets.catalog import get_market_catalog
from app.infra.db.models.feature_store import FeatureRunPointer
from app.services.market_activity_service import (
    mark_market_activity_completed,
    mark_market_activity_failed,
    mark_market_activity_started,
)
from app.tasks.data_fetch_lock import serialized_data_fetch_task
from app.use_cases.options_analytics import RefreshOptionsAnalyticsCommand
from app.wiring.bootstrap import get_refresh_options_analytics_use_case

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    return bool(settings.options_analytics_enabled)


def _mark_activity_safely(function, db, **values) -> None:
    try:
        function(db, **values)
    except Exception:
        logger.warning("Could not publish Options Analytics activity", exc_info=True)
        try:
            db.rollback()
        except Exception:
            logger.warning(
                "Could not roll back failed Options Analytics activity",
                exc_info=True,
            )


@serialized_data_fetch_task(
    celery_app,
    "daily-us-options-analytics",
    enabled=_enabled,
    disabled_reason="options_analytics_disabled",
    name="app.interfaces.tasks.options_analytics_tasks.refresh_options_analytics",
)
def refresh_options_analytics(
    self,
    source_run_id: int | None = None,
    *,
    market: str = "US",
    force: bool = False,
) -> dict:
    market_code = market.strip().upper()
    if market_code != "US":
        return {"status": "skipped", "reason_codes": ["market_unsupported"]}
    if not get_market_catalog().get(market_code).capabilities.options_analytics:
        return {
            "status": "skipped",
            "reason_codes": ["market_capability_unavailable"],
        }

    db = SessionLocal()
    task_id = getattr(getattr(self, "request", None), "id", None)
    activity = {
        "market": "US",
        "stage_key": "options",
        "lifecycle": "daily_refresh",
        "task_name": getattr(self, "name", "daily-us-options-analytics"),
        "task_id": task_id,
    }
    try:
        if source_run_id is None:
            pointer = db.get(FeatureRunPointer, "latest_published_market:US")
            if pointer is None:
                return {
                    "status": "skipped",
                    "reason_codes": ["source_run_unavailable"],
                }
            source_run_id = pointer.run_id
        _mark_activity_safely(
            mark_market_activity_started,
            db,
            **activity,
            current=0,
            message="Refreshing Options Command Center",
        )
        use_case = get_refresh_options_analytics_use_case(db)
        result = use_case.execute(
            RefreshOptionsAnalyticsCommand(
                source_run_id=source_run_id,
                market="US",
                enabled=True,
                force=force,
            )
        )
        expected = int(result.get("expected_count") or 0)
        completed = int(result.get("completed_count") or 0)
        message = (
            f"Options Analytics: completed={completed}/{expected}, "
            f"core_valid={int(result.get('core_valid_current_count') or 0)}, "
            f"failed={int(result.get('failed_count') or 0)}, "
            f"retried={int(result.get('retried_count') or 0)}, "
            f"coverage={float(result.get('coverage') or 0):.1%}"
        )
        marker = (
            mark_market_activity_completed
            if result.get("status") == "published"
            else mark_market_activity_failed
        )
        _mark_activity_safely(
            marker,
            db,
            **activity,
            current=completed,
            total=expected,
            message=message,
        )
        return result
    except Exception as exc:
        _mark_activity_safely(
            mark_market_activity_failed,
            db,
            **activity,
            message=str(exc),
        )
        raise
    finally:
        db.close()
