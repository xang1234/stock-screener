"""Celery task wrapper for the daily feature snapshot use case.

Thin shim — all business logic lives in
:class:`~app.use_cases.feature_store.build_daily_snapshot.BuildDailyFeatureSnapshotUseCase`.
"""

from __future__ import annotations

import logging
from datetime import date

from app.celery_app import celery_app
from app.domain.scanning.signature import (
    build_scan_signature_payload,
    hash_scan_signature,
    hash_universe_symbols,
)
from app.tasks.data_fetch_lock import serialized_data_fetch

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="app.interfaces.tasks.feature_store_tasks.build_daily_snapshot",
    autoretry_for=(ConnectionError, TimeoutError, OSError),
    retry_backoff=60,
    retry_backoff_max=600,
    max_retries=3,
)
@serialized_data_fetch("build_daily_snapshot")
def build_daily_snapshot(
    self,
    as_of_date_str: str | None = None,
    screener_names: list[str] | None = None,
    universe_name: str | None = None,
    criteria: dict | None = None,
    composite_method: str | None = None,
    skip_if_published: bool = True,
) -> dict:
    """Build a full feature snapshot for a trading day.

    Parameters
    ----------
    as_of_date_str:
        ISO date string (YYYY-MM-DD). Defaults to today.
    screener_names:
        Screener list, e.g. ``["minervini", "canslim"]``.
    universe_name:
        Universe filter name. Defaults to the shared scan default profile.
    skip_if_published:
        If True (default), skip when a PUBLISHED run already exists for the
        date. Pass False to force a rebuild.
    """
    from app.database import SessionLocal
    from app.domain.scanning.ports import NeverCancelledToken
    from app.infra.db.uow import SqlUnitOfWork
    from app.infra.tasks.progress_sink import CeleryProgressSink
    from app.services.universe_resolver import normalize_universe_definition
    from app.use_cases.feature_store.build_daily_snapshot import (
        BuildDailySnapshotCommand,
        _is_us_trading_day,
    )
    from app.wiring.bootstrap import get_build_daily_snapshot_use_case

    as_of = date.fromisoformat(as_of_date_str) if as_of_date_str else date.today()
    from app.domain.scanning.defaults import get_default_scan_profile

    defaults = get_default_scan_profile()
    screeners = defaults["screeners"] if screener_names is None else screener_names
    universe_name = defaults["universe"] if universe_name is None else universe_name
    criteria = defaults["criteria"] if criteria is None else criteria
    composite_method = (
        defaults["composite_method"]
        if composite_method is None
        else composite_method
    )
    correlation_id = self.request.id

    logger.info(
        "┌─── build_daily_snapshot ───────────────────┐\n"
        "│  date=%s  correlation_id=%s\n"
        "│  screeners=%s  universe=%s  composite=%s\n"
        "└────────────────────────────────────────────┘",
        as_of, correlation_id, screeners, universe_name, composite_method,
    )

    # ── Trading-day guard (fast exit — lock released immediately) ─
    if not _is_us_trading_day(as_of):
        logger.info("Skipping build_daily_snapshot: %s is not a US trading day", as_of)
        return {"status": "skipped", "reason": "not_trading_day", "as_of_date": str(as_of)}

    # ── Skip-if-published guard (cheap DB check) ─────────────
    if skip_if_published:
        uow_check = SqlUnitOfWork(SessionLocal)
        with uow_check:
            universe_def = normalize_universe_definition(universe_name)
            symbols = uow_check.universe.resolve_symbols(universe_def)
            signature_payload = build_scan_signature_payload(
                universe_type=getattr(universe_def, "type", "all"),
                screeners=screeners,
                composite_method=composite_method,
                criteria=criteria,
            )
            input_hash = hash_scan_signature(signature_payload)
            universe_hash = hash_universe_symbols(symbols)
            matching_run = uow_check.feature_runs.find_latest_published_exact(
                input_hash=input_hash,
                universe_hash=universe_hash,
                as_of_date=as_of,
            )
            if matching_run is not None:
                logger.info(
                    "Skipping build_daily_snapshot: run %d already published for %s",
                    matching_run.id, as_of,
                )
                return {
                    "status": "skipped",
                    "reason": "already_published",
                    "existing_run_id": matching_run.id,
                    "as_of_date": str(as_of),
                }

    # ── Execute use case ─────────────────────────────────────
    use_case = get_build_daily_snapshot_use_case()
    uow = SqlUnitOfWork(SessionLocal)
    cmd = BuildDailySnapshotCommand(
        as_of_date=as_of,
        screener_names=screeners,
        universe_def=normalize_universe_definition(universe_name),
        criteria=criteria,
        composite_method=composite_method,
        correlation_id=correlation_id,
    )

    result = use_case.execute(
        uow=uow,
        cmd=cmd,
        progress=CeleryProgressSink(self),
        cancel=NeverCancelledToken(),
    )

    logger.info(
        "build_daily_snapshot completed: run_id=%d status=%s "
        "total=%d processed=%d failed=%d dq_passed=%s",
        result.run_id, result.status,
        result.total_symbols, result.processed_symbols,
        result.failed_symbols, result.dq_passed,
    )

    return {
        "run_id": result.run_id,
        "status": result.status,
        "as_of_date": str(as_of),
        "total_symbols": result.total_symbols,
        "processed_symbols": result.processed_symbols,
        "failed_symbols": result.failed_symbols,
        "dq_passed": result.dq_passed,
    }
