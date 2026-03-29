"""Celery task wrapper for the daily feature snapshot use case.

Thin shim — all business logic lives in
:class:`~app.use_cases.feature_store.build_daily_snapshot.BuildDailyFeatureSnapshotUseCase`.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from uuid import uuid4

from celery.exceptions import SoftTimeLimitExceeded

from app.celery_app import celery_app
from app.domain.scanning.signature import (
    build_scan_signature_payload,
    hash_scan_signature,
    hash_universe_symbols,
)
from app.tasks.data_fetch_lock import serialized_data_fetch

logger = logging.getLogger(__name__)


def _create_auto_scan_for_published_run(
    *,
    feature_run_id: int,
    universe_name: str,
    screeners: list[str],
    criteria: dict | None,
    composite_method: str,
) -> str:
    """Create a completed scan row bound to a published feature run."""
    from app.database import SessionLocal
    from app.infra.db.uow import SqlUnitOfWork
    from app.models.scan_result import SCAN_TRIGGER_SOURCE_AUTO
    from app.services.ui_snapshot_service import safe_publish_scan_bootstrap
    from app.services.scan_execution import cleanup_old_scans
    from app.services.universe_resolver import normalize_universe_definition

    auto_idempotency_key = f"auto-feature-run:{feature_run_id}"
    universe_key: str | None = None

    with SqlUnitOfWork(SessionLocal) as uow:
        existing = uow.scans.get_by_idempotency_key(auto_idempotency_key)
        if existing is not None:
            scan_id = existing.scan_id
        else:
            universe_def = normalize_universe_definition(universe_name)
            feature_run = uow.feature_runs.get_run(feature_run_id)
            symbols = uow.universe.resolve_symbols(universe_def)
            ran_at = feature_run.published_at or datetime.now(timezone.utc)
            passed_stocks = (
                feature_run.stats.passed_symbols
                if feature_run.stats and feature_run.stats.passed_symbols is not None
                else 0
            )
            scan = uow.scans.create(
                scan_id=str(uuid4()),
                criteria=criteria or {},
                universe=universe_def.label(),
                universe_key=universe_def.key(),
                universe_type=universe_def.type.value,
                universe_exchange=(
                    universe_def.exchange.value if universe_def.exchange else None
                ),
                universe_index=(
                    universe_def.index.value if universe_def.index else None
                ),
                universe_symbols=universe_def.symbols,
                screener_types=screeners,
                composite_method=composite_method,
                total_stocks=len(symbols),
                passed_stocks=passed_stocks,
                status="completed",
                task_id=None,
                idempotency_key=auto_idempotency_key,
                feature_run_id=feature_run_id,
                trigger_source=SCAN_TRIGGER_SOURCE_AUTO,
                started_at=ran_at,
                completed_at=ran_at,
            )
            scan_id = scan.scan_id
            universe_key = scan.universe_key
            uow.commit()

    if universe_key:
        db = SessionLocal()
        try:
            cleanup_old_scans(db, universe_key)
        finally:
            db.close()

    safe_publish_scan_bootstrap(scan_id)
    safe_publish_scan_bootstrap()
    return scan_id


@celery_app.task(
    bind=True,
    name="app.interfaces.tasks.feature_store_tasks.build_daily_snapshot",
    autoretry_for=(ConnectionError, TimeoutError, OSError),
    retry_backoff=60,
    retry_backoff_max=600,
    max_retries=3,
    soft_time_limit=3600,
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
                auto_scan_id = _create_auto_scan_for_published_run(
                    feature_run_id=matching_run.id,
                    universe_name=universe_name,
                    screeners=screeners,
                    criteria=criteria,
                    composite_method=composite_method,
                )
                logger.info(
                    "Skipping build_daily_snapshot: run %d already published for %s "
                    "(auto scan %s ensured)",
                    matching_run.id, as_of, auto_scan_id,
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

    try:
        result = use_case.execute(
            uow=uow,
            cmd=cmd,
            progress=CeleryProgressSink(self),
            cancel=NeverCancelledToken(),
        )
    except SoftTimeLimitExceeded:
        logger.error(
            "Soft time limit exceeded in build_daily_snapshot for %s (correlation_id=%s)",
            as_of,
            correlation_id,
            exc_info=True,
        )
        raise

    logger.info(
        "build_daily_snapshot completed: run_id=%d status=%s "
        "total=%d processed=%d failed=%d dq_passed=%s",
        result.run_id, result.status,
        result.total_symbols, result.processed_symbols,
        result.failed_symbols, result.dq_passed,
    )

    if result.status == "published":
        auto_scan_id = _create_auto_scan_for_published_run(
            feature_run_id=result.run_id,
            universe_name=universe_name,
            screeners=screeners,
            criteria=criteria,
            composite_method=composite_method,
        )
        logger.info(
            "Created auto scan %s for published feature run %d",
            auto_scan_id,
            result.run_id,
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
