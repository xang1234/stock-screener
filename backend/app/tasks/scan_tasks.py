"""
Celery tasks for bulk stock scanning.

Handles background processing of large-scale stock scans.

User scan execution runs on the dedicated `user_scans` queue and is
intentionally decoupled from the global `data_fetch` lock used by
maintenance jobs.
"""
import logging
from typing import List, Dict, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from ..celery_app import celery_app
from ..database import SessionLocal
from ..models.scan_result import SCAN_TRIGGER_SOURCE_AUTO, Scan, ScanResult
from ..config import settings
from ..services.scan_execution import compute_industry_peer_metrics
from ..utils.parallelism import resolve_scan_compute_processes
from .workload_coordination import serialized_market_workload

logger = logging.getLogger(__name__)

# Only listed sources may fall back to live provider fetches; every other
# source runs cache-only. Today only manual scans reach this runner, so every
# scan it runs is cache-only.
_SCAN_SOURCES_ALLOWED_TO_FETCH = frozenset({SCAN_TRIGGER_SOURCE_AUTO})


def cleanup_old_scans(db: Session, universe_key: str, keep_count: int = 3) -> None:
    from ..services.scan_execution import cleanup_old_scans as _cleanup_old_scans

    _cleanup_old_scans(db, universe_key, keep_count=keep_count)


def _log_setup_engine_distribution(db: Session, scan_id: str) -> None:
    """Log setup_engine score distribution for observability."""
    try:
        from sqlalchemy import select
        from app.infra.db.portability import json_number, json_text

        score_expr = json_number(
            ScanResult.details,
            ("setup_engine", "setup_score"),
            bind_or_session=db,
        )
        ready_expr = json_text(
            ScanResult.details,
            ("setup_engine", "setup_ready"),
            bind_or_session=db,
        )
        scores_rows = db.execute(
            select(
                score_expr,
                ready_expr,
            ).where(
                ScanResult.scan_id == scan_id,
                score_expr.isnot(None),
            )
        ).all()

        if not scores_rows:
            return

        scores = [float(row[0]) for row in scores_rows if row[0] is not None]
        if not scores:
            return

        ready_count = sum(1 for row in scores_rows if row[1] in (True, 1, "true", "1"))
        scores_sorted = sorted(scores)
        n = len(scores_sorted)
        median = (
            scores_sorted[n // 2]
            if n % 2 == 1
            else (scores_sorted[n // 2 - 1] + scores_sorted[n // 2]) / 2.0
        )

        logger.info(
            "SE score distribution [%s]: n=%d min=%.1f max=%.1f mean=%.1f median=%.1f ready=%d",
            scan_id, n, min(scores), max(scores),
            sum(scores) / n, median, ready_count,
        )
    except Exception as e:
        logger.debug("SE distribution telemetry skipped: %s", e)


def _run_post_scan_pipeline(scan_id: str, *, warm_chart_cache: bool = True) -> None:
    """Post-scan: peer metrics, retention cleanup, snapshot publish, optional chart warmup."""
    db = SessionLocal()
    try:
        compute_industry_peer_metrics(db, scan_id)
        _log_setup_engine_distribution(db, scan_id)
        scan = db.query(Scan).filter(Scan.scan_id == scan_id).first()
        if scan and scan.universe_key:
            cleanup_old_scans(db, scan.universe_key)
        try:
            from ..services.ui_snapshot_service import safe_publish_scan_bootstrap

            # Publishes the scan's variant and its market's latest variant.
            safe_publish_scan_bootstrap(scan_id)
        except Exception as e:
            logger.warning("Scan snapshot publish failed: %s", e)
        if warm_chart_cache:
            try:
                from .cache_tasks import prewarm_chart_cache_for_scan

                prewarm_chart_cache_for_scan.delay(scan_id, top_n=50)
            except Exception as e:
                logger.warning("Chart cache warming failed: %s", e)
    except Exception as e:
        logger.error("Post-scan pipeline error for %s: %s", scan_id, e, exc_info=True)
    finally:
        db.close()


def _mark_scan_failed(scan_id: str) -> None:
    """Best-effort update of the scan row to failed status for wrapper-level errors.

    The use case has its own rollback+failed-status handler for exceptions that
    occur during execute(); this helper covers the narrow window where the
    wrapper raises BEFORE execute() is entered (e.g., DB lookup error while
    resolving trigger_source). Failures here are swallowed — the outer task
    still re-raises so Celery sees the error.
    """
    try:
        db = SessionLocal()
        try:
            scan_row = db.query(Scan).filter(Scan.scan_id == scan_id).first()
            if scan_row is not None and scan_row.status in {"queued", "running"}:
                scan_row.status = "failed"
                db.commit()
        finally:
            db.close()
    except Exception:
        logger.warning("Could not mark scan %s failed from wrapper", scan_id, exc_info=True)


def _run_bulk_scan_via_use_case(task_instance, scan_id, symbol_list, criteria):
    """Thin wrapper that delegates to RunBulkScanUseCase (new path).

    All business logic lives in the use case; this function only
    wires infrastructure adapters (progress, cancellation, UoW).
    """
    # Lazy imports to avoid circular dep:
    # bootstrap -> dispatcher -> scan_tasks -> bootstrap
    from ..wiring.bootstrap import get_run_bulk_scan_use_case
    from ..infra.db.uow import SqlUnitOfWork
    from ..infra.tasks.progress_sink import CeleryProgressSink
    from ..infra.tasks.cancellation import DbCancellationToken
    from ..use_cases.scanning.run_bulk_scan import RunBulkScanCommand

    progress = CeleryProgressSink(task_instance)
    cancel = DbCancellationToken(SessionLocal, scan_id)

    try:
        # This lookup has to live inside the try block so a transient DB error
        # still runs the cancel-token cleanup and lets the use case (via its
        # own exception handler) mark the scan failed.
        db = SessionLocal()
        try:
            scan_row = db.query(Scan).filter(Scan.scan_id == scan_id).first()
            trigger_source = (getattr(scan_row, "trigger_source", None) or "manual").lower()
        finally:
            db.close()
        cache_only = trigger_source not in _SCAN_SOURCES_ALLOWED_TO_FETCH

        uow = SqlUnitOfWork(SessionLocal)
        use_case = get_run_bulk_scan_use_case()
        cmd = RunBulkScanCommand(
            scan_id=scan_id,
            symbols=symbol_list,
            criteria=criteria or {},
            chunk_size=settings.scan_usecase_chunk_size,
            correlation_id=task_instance.request.id,
            cache_only=cache_only,
            parallel_workers=resolve_scan_compute_processes(
                settings.scan_compute_processes
            ),
        )
        result = use_case.execute(uow, cmd, progress, cancel)
    except Exception:
        logger.error("Fatal error in use case scan %s", scan_id, exc_info=True)
        _mark_scan_failed(scan_id)
        raise
    finally:
        cancel.close()

    if result.status == "completed":
        try:
            finalize_scan_artifacts.delay(scan_id)
        except Exception:
            logger.warning("Falling back to inline post-scan finalization for %s", scan_id, exc_info=True)
            _run_post_scan_pipeline(scan_id)
    elif result.status == "cancelled":
        # The cancel request published while this worker could still be writing
        # a chunk, and an explicit scan variant never goes stale; republish now
        # that no more results will be written.
        queue_scan_bootstrap_publish(scan_id)

    return {
        "scan_id": result.scan_id,
        "completed": result.total_scanned,
        "passed": result.passed,
        "failed": result.failed,
        "status": result.status,
        "scan_path": "use_case",
    }


@celery_app.task(bind=True, name='app.tasks.scan_tasks.run_bulk_scan')
@serialized_market_workload("run_bulk_scan")
def run_bulk_scan(
    self,
    scan_id: str,
    symbol_list: List[str],
    criteria: dict = None,
    market: str | None = None,
):
    """Scan multiple stocks in background via RunBulkScanUseCase."""
    return _run_bulk_scan_via_use_case(self, scan_id, symbol_list, criteria)


@celery_app.task(name='app.tasks.scan_tasks.finalize_scan_artifacts', queue='celery')
def finalize_scan_artifacts(scan_id: str):
    """Finalize scan artifacts on the general queue after the scan worker exits."""
    _run_post_scan_pipeline(scan_id, warm_chart_cache=True)
    return {"scan_id": scan_id, "status": "queued_post_scan_finalization"}


# ignore_result: otherwise apply_async subscribes to the result backend first,
# which retries for minutes when Redis is down and would hang the request.
@celery_app.task(
    name='app.tasks.scan_tasks.publish_scan_bootstrap_snapshots',
    queue='celery',
    ignore_result=True,
)
def publish_scan_bootstrap_snapshots(scan_id: str, prune_history: bool = False):
    """Rebuild a scan's bootstrap snapshot and its market's "latest" one on the general queue.

    Request handlers queue this instead of building snapshots inline. "latest"
    is resolved when the task runs, and the publish re-checks it under the
    variant's advisory lock, so a slow task never moves "latest" back to an older scan.

    ``prune_history`` applies scan retention first, for a scan that completed
    instantly and so never ran the post-scan pipeline that prunes. Retention
    keeps the newest auto scan (#682), and deletes cancelled scans, so the
    cancel path never asks for it.
    """
    from ..services.ui_snapshot_service import safe_publish_scan_bootstrap

    if prune_history:
        from ..services.scan_execution import prune_history_of_scan

        db = SessionLocal()
        try:
            prune_history_of_scan(db, scan_id)
        except Exception:  # noqa: BLE001 - retention must never block the publish
            logger.warning("Scan retention failed for %s", scan_id, exc_info=True)
        finally:
            db.close()
    safe_publish_scan_bootstrap(scan_id)
    return {"scan_id": scan_id, "status": "published_scan_bootstrap"}


@celery_app.task(
    name='app.tasks.scan_tasks.publish_scan_latest_bootstrap',
    queue='celery',
    ignore_result=True,
)
def publish_scan_latest_bootstrap(market: str | None = None):
    """Publish a market's "latest" scan bootstrap that a reader found missing.

    Scan publishes refresh that variant, so a market whose snapshot was reset,
    or never published, would otherwise stay without one until its next scan.
    """
    from ..services.ui_snapshot_service import safe_publish_scan_latest

    safe_publish_scan_latest(market)
    return {"market": market, "status": "published_scan_latest_bootstrap"}


_BOOTSTRAP_ENQUEUE_TRANSPORT_OPTIONS = {
    "socket_connect_timeout": 1.0,
    "socket_timeout": 1.0,
    "max_retries": 0,  # one connect attempt, no kombu reconnect loop
}


def queue_scan_bootstrap_publish(scan_id: str, *, prune_history: bool = False) -> None:
    """Queue the bootstrap rebuild for a request handler instead of building it inline.

    A failed enqueue only delays the snapshot: the scan page treats an absent or
    stale bootstrap as missing and reads the regular scan endpoints, so the
    request still succeeds.

    The send uses its own short-timeout broker connection so an outage fails
    within about a second: with no socket timeouts, connecting to an
    unreachable host waits for the OS TCP timeout (~75 s measured), and kombu
    also retries the connect. The app-wide transport options, which workers
    use, are left alone.
    """
    try:
        with celery_app.connection_for_write(
            transport_options=_BOOTSTRAP_ENQUEUE_TRANSPORT_OPTIONS
        ) as connection:
            publish_scan_bootstrap_snapshots.apply_async(
                args=[scan_id],
                **({"kwargs": {"prune_history": True}} if prune_history else {}),
                retry=False,
                connection=connection,
            )
    except Exception:
        logger.warning("Could not queue scan bootstrap publish for %s", scan_id, exc_info=True)


# One queued publish per missing latest variant at a time: every page load
# that finds it missing asks, and the build is the same for all of them.
_LATEST_PUBLISH_CLAIM_SECONDS = 60


def queue_scan_latest_bootstrap_publish(market: str | None) -> None:
    """Queue ``publish_scan_latest_bootstrap``; fails fast like the scan publish.

    A short Redis claim per variant coalesces concurrent requests. Without
    Redis every request enqueues, which only repeats an idempotent build.
    """
    from ..services.redis_pool import get_redis_client

    claim_key = f"scan_bootstrap:latest_publish_queued:{market or 'all'}"
    redis = None
    try:
        redis = get_redis_client()
        if redis is not None and not redis.set(
            claim_key, "1", nx=True, ex=_LATEST_PUBLISH_CLAIM_SECONDS
        ):
            return
    except Exception:
        logger.warning("Could not claim the latest scan bootstrap publish for %s", market, exc_info=True)
        redis = None
    try:
        with celery_app.connection_for_write(
            transport_options=_BOOTSTRAP_ENQUEUE_TRANSPORT_OPTIONS
        ) as connection:
            publish_scan_latest_bootstrap.apply_async(
                args=[market], retry=False, connection=connection
            )
    except Exception:
        logger.warning(
            "Could not queue the latest scan bootstrap publish for %s", market, exc_info=True
        )
        if redis is not None:
            try:
                redis.delete(claim_key)  # let the next request try again
            except Exception:
                logger.warning("Could not release %s", claim_key, exc_info=True)


@celery_app.task(name='app.tasks.scan_tasks.test_celery')
def test_celery():
    """
    Simple test task to verify Celery is working.

    Returns:
        Success message
    """
    logger.info("Test Celery task executed successfully")
    return {'status': 'success', 'message': 'Celery is working!'}
