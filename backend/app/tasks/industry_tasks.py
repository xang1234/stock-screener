"""Runtime tasks for tracked market reference datasets."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path

from ..celery_app import celery_app
from ..config import settings
from ..database import SessionLocal
from ..services.ibd_industry_service import IBDIndustryService
from .market_queues import log_extra, normalize_market
from .workload_coordination import serialized_market_workload

logger = logging.getLogger(__name__)


def _tracked_ibd_csv_path() -> Path:
    return IBDIndustryService.resolve_tracked_csv_path(settings.ibd_industry_csv_path)


@celery_app.task(
    bind=True,
    name="app.tasks.industry_tasks.load_tracked_ibd_industry_groups",
)
@serialized_market_workload("load_tracked_ibd_industry_groups")
def load_tracked_ibd_industry_groups(
    self,
    *,
    market: str = "US",
    activity_lifecycle: str | None = None,
) -> dict:
    """Load the tracked US IBD industry-group CSV into runtime tables."""
    del activity_lifecycle  # reserved for parity with bootstrap task signatures

    effective_market = normalize_market(market)
    _log_extra = log_extra(market)
    if effective_market != "US":
        logger.info("Skipping tracked IBD industry load for unsupported market %s", market, extra=_log_extra)
        return {
            "status": "skipped",
            "reason": "ibd_industry_mappings_are_us_only",
            "market": effective_market,
            "timestamp": datetime.now().isoformat(),
        }

    db = SessionLocal()
    try:
        csv_path = _tracked_ibd_csv_path()
        loaded = IBDIndustryService.load_from_csv(db, csv_path=csv_path)
        logger.info(
            "Loaded tracked IBD industry mappings",
            extra={"market": "us", "loaded_rows": loaded, "csv_path": str(csv_path)},
        )
        return {
            "status": "loaded",
            "market": effective_market,
            "loaded_rows": loaded,
            "csv_path": str(csv_path),
            "timestamp": datetime.now().isoformat(),
        }
    finally:
        db.close()
