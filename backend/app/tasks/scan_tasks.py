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
from ..database import SessionLocal, is_corruption_error, safe_rollback
from ..models.scan_result import Scan, ScanResult
from ..config import settings

logger = logging.getLogger(__name__)


def cleanup_old_scans(db: Session, universe_key: str, keep_count: int = 3) -> None:
    """
    Delete old scans, keeping only the most recent `keep_count` per universe_key.

    Called after successful scan completion to maintain retention policy.
    Uses universe_key (canonical identifier) instead of the legacy universe string,
    so different exchange/custom scans get separate retention buckets.

    Handles:
    1. Cancelled scans - deleted immediately (no value)
    2. Stale running/queued scans - deleted if older than 1 hour (orphaned)
    3. Completed scans - keep only the most recent `keep_count`

    Args:
        db: Database session
        universe_key: Canonical universe key (e.g., "all", "exchange:NYSE", "custom:<hash>")
        keep_count: Number of recent scans to keep (default: 3)
    """
    from datetime import timedelta

    try:
        total_deleted_scans = 0
        total_deleted_results = 0

        # 1. Delete all cancelled scans for this universe (they have no value)
        cancelled_scans = db.query(Scan).filter(
            Scan.universe_key == universe_key,
            Scan.status == "cancelled"
        ).all()

        if cancelled_scans:
            logger.info(f"Cleaning up {len(cancelled_scans)} cancelled scans for universe_key '{universe_key}'")
            for scan in cancelled_scans:
                deleted_results = db.query(ScanResult).filter(
                    ScanResult.scan_id == scan.scan_id
                ).delete()
                db.delete(scan)
                total_deleted_results += deleted_results
                total_deleted_scans += 1
                logger.debug(f"Deleted cancelled scan {scan.scan_id} ({deleted_results} results)")

        # 2. Delete stale running/queued scans (older than 1 hour - they will never complete)
        stale_cutoff = datetime.utcnow() - timedelta(hours=1)
        stale_scans = db.query(Scan).filter(
            Scan.universe_key == universe_key,
            Scan.status.in_(["running", "queued"]),
            Scan.started_at < stale_cutoff
        ).all()

        if stale_scans:
            logger.info(f"Cleaning up {len(stale_scans)} stale running/queued scans for universe_key '{universe_key}'")
            for scan in stale_scans:
                deleted_results = db.query(ScanResult).filter(
                    ScanResult.scan_id == scan.scan_id
                ).delete()
                db.delete(scan)
                total_deleted_results += deleted_results
                total_deleted_scans += 1
                logger.debug(f"Deleted stale scan {scan.scan_id} (status={scan.status}, {deleted_results} results)")

        # 3. Keep only the last `keep_count` completed scans (existing logic)
        completed_scans = db.query(Scan).filter(
            Scan.universe_key == universe_key,
            Scan.status == "completed"
        ).order_by(Scan.completed_at.desc()).all()

        scans_to_delete = completed_scans[keep_count:]

        if scans_to_delete:
            logger.info(f"Cleaning up {len(scans_to_delete)} old completed scans for universe_key '{universe_key}'")
            for scan in scans_to_delete:
                deleted_results = db.query(ScanResult).filter(
                    ScanResult.scan_id == scan.scan_id
                ).delete()
                db.delete(scan)
                total_deleted_results += deleted_results
                total_deleted_scans += 1
                logger.debug(f"Deleted old scan {scan.scan_id} ({deleted_results} results)")

        if total_deleted_scans > 0:
            db.commit()
            logger.info(
                f"Cleanup complete for '{universe_key}': deleted {total_deleted_scans} scans, "
                f"{total_deleted_results} results. Kept {keep_count} most recent completed scans."
            )

    except Exception as e:
        if is_corruption_error(e):
            logger.critical("DATABASE CORRUPTION in cleanup_old_scans: %s — run scripts/check_db_integrity.py --repair", e)
        else:
            logger.error("Error cleaning up old scans: %s", e, exc_info=True)
        safe_rollback(db)


def compute_industry_peer_metrics(db: Session, scan_id: str):
    """
    Compute aggregate metrics for each industry group in scan.
    Called after scan completion.

    Args:
        db: Database session
        scan_id: Scan ID
    """
    from ..models.industry import IBDGroupPeerCache
    from sqlalchemy import func

    try:
        logger.info(f"Computing industry peer metrics for scan {scan_id}...")

        # Query all industry groups in this scan
        groups = db.query(
            ScanResult.ibd_industry_group,
            func.count(ScanResult.symbol).label('total_stocks'),
            func.avg(ScanResult.rs_rating_1m).label('avg_rs_1m'),
            func.avg(ScanResult.rs_rating_3m).label('avg_rs_3m'),
            func.avg(ScanResult.rs_rating_12m).label('avg_rs_12m'),
            func.avg(ScanResult.minervini_score).label('avg_minervini_score'),
            func.avg(ScanResult.composite_score).label('avg_composite_score'),
        ).filter(
            ScanResult.scan_id == scan_id,
            ScanResult.ibd_industry_group.isnot(None)
        ).group_by(
            ScanResult.ibd_industry_group
        ).all()

        # Save cache records
        for group in groups:
            # Find top performer in this group
            top = db.query(ScanResult).filter(
                ScanResult.scan_id == scan_id,
                ScanResult.ibd_industry_group == group.ibd_industry_group
            ).order_by(
                ScanResult.composite_score.desc()
            ).first()

            cache = IBDGroupPeerCache(
                scan_id=scan_id,
                industry_group=group.ibd_industry_group,
                total_stocks=group.total_stocks,
                avg_rs_1m=group.avg_rs_1m,
                avg_rs_3m=group.avg_rs_3m,
                avg_rs_12m=group.avg_rs_12m,
                avg_minervini_score=group.avg_minervini_score,
                avg_composite_score=group.avg_composite_score,
                top_symbol=top.symbol if top else None,
                top_score=top.composite_score if top else None
            )
            db.add(cache)

        db.commit()
        logger.info(f"Computed peer metrics for {len(groups)} industry groups in scan {scan_id}")

    except Exception as e:
        if is_corruption_error(e):
            logger.critical("DATABASE CORRUPTION in compute_industry_peer_metrics: %s — run scripts/check_db_integrity.py --repair", e)
        else:
            logger.error("Error computing peer metrics: %s", e, exc_info=True)
        safe_rollback(db)


def _log_setup_engine_distribution(db: Session, scan_id: str) -> None:
    """Log setup_engine score distribution for observability."""
    try:
        from sqlalchemy import func, select
        scores_rows = db.execute(
            select(
                func.json_extract(ScanResult.details, '$.setup_engine.setup_score'),
                func.json_extract(ScanResult.details, '$.setup_engine.setup_ready'),
            ).where(
                ScanResult.scan_id == scan_id,
                func.json_extract(ScanResult.details, '$.setup_engine.setup_score').isnot(None),
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


def _run_post_scan_pipeline(scan_id: str) -> None:
    """Post-scan: peer metrics, retention cleanup, chart cache warming, SE telemetry."""
    db = SessionLocal()
    try:
        compute_industry_peer_metrics(db, scan_id)
        _log_setup_engine_distribution(db, scan_id)
        scan = db.query(Scan).filter(Scan.scan_id == scan_id).first()
        if scan and scan.universe_key:
            cleanup_old_scans(db, scan.universe_key)
        try:
            from .cache_tasks import prewarm_chart_cache_for_scan
            prewarm_chart_cache_for_scan.delay(scan_id, top_n=50)
        except Exception as e:
            logger.warning("Chart cache warming failed: %s", e)
    except Exception as e:
        logger.error("Post-scan pipeline error for %s: %s", scan_id, e, exc_info=True)
    finally:
        db.close()


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
        uow = SqlUnitOfWork(SessionLocal)
        use_case = get_run_bulk_scan_use_case()
        cmd = RunBulkScanCommand(
            scan_id=scan_id,
            symbols=symbol_list,
            criteria=criteria or {},
            chunk_size=settings.scan_usecase_chunk_size,
            correlation_id=task_instance.request.id,
        )
        result = use_case.execute(uow, cmd, progress, cancel)
    except Exception:
        logger.error("Fatal error in use case scan %s", scan_id, exc_info=True)
        raise
    finally:
        cancel.close()

    if result.status == "completed":
        _run_post_scan_pipeline(scan_id)

    return {
        "scan_id": result.scan_id,
        "completed": result.total_scanned,
        "passed": result.passed,
        "failed": result.failed,
        "status": result.status,
        "scan_path": "use_case",
    }


@celery_app.task(bind=True, name='app.tasks.scan_tasks.run_bulk_scan')
def run_bulk_scan(self, scan_id: str, symbol_list: List[str], criteria: dict = None):
    """Scan multiple stocks in background via RunBulkScanUseCase."""
    return _run_bulk_scan_via_use_case(self, scan_id, symbol_list, criteria)


@celery_app.task(name='app.tasks.scan_tasks.test_celery')
def test_celery():
    """
    Simple test task to verify Celery is working.

    Returns:
        Success message
    """
    logger.info("Test Celery task executed successfully")
    return {'status': 'success', 'message': 'Celery is working!'}
