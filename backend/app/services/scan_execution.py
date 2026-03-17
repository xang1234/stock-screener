"""Shared scan execution helpers that do not import Celery at module import time."""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal, is_corruption_error, safe_rollback
from app.models.scan_result import Scan, ScanResult

logger = logging.getLogger(__name__)


def cleanup_old_scans(db: Session, universe_key: str, keep_count: int = 3) -> None:
    """Delete stale and old scans while keeping recent completed history."""
    from datetime import timedelta

    try:
        total_deleted_scans = 0
        total_deleted_results = 0

        cancelled_scans = db.query(Scan).filter(
            Scan.universe_key == universe_key,
            Scan.status == "cancelled",
        ).all()

        if cancelled_scans:
            logger.info(
                "Cleaning up %d cancelled scans for universe_key '%s'",
                len(cancelled_scans),
                universe_key,
            )
            for scan in cancelled_scans:
                deleted_results = db.query(ScanResult).filter(
                    ScanResult.scan_id == scan.scan_id
                ).delete()
                db.delete(scan)
                total_deleted_results += deleted_results
                total_deleted_scans += 1

        stale_cutoff = datetime.utcnow() - timedelta(hours=1)
        stale_scans = db.query(Scan).filter(
            Scan.universe_key == universe_key,
            Scan.status.in_(["running", "queued"]),
            Scan.started_at < stale_cutoff,
        ).all()

        if stale_scans:
            logger.info(
                "Cleaning up %d stale running/queued scans for universe_key '%s'",
                len(stale_scans),
                universe_key,
            )
            for scan in stale_scans:
                deleted_results = db.query(ScanResult).filter(
                    ScanResult.scan_id == scan.scan_id
                ).delete()
                db.delete(scan)
                total_deleted_results += deleted_results
                total_deleted_scans += 1

        completed_scans = db.query(Scan).filter(
            Scan.universe_key == universe_key,
            Scan.status == "completed",
        ).order_by(Scan.completed_at.desc()).all()

        scans_to_delete = completed_scans[keep_count:]
        if scans_to_delete:
            logger.info(
                "Cleaning up %d old completed scans for universe_key '%s'",
                len(scans_to_delete),
                universe_key,
            )
            for scan in scans_to_delete:
                deleted_results = db.query(ScanResult).filter(
                    ScanResult.scan_id == scan.scan_id
                ).delete()
                db.delete(scan)
                total_deleted_results += deleted_results
                total_deleted_scans += 1

        if total_deleted_scans > 0:
            db.commit()
            logger.info(
                "Cleanup complete for '%s': deleted %d scans, %d results.",
                universe_key,
                total_deleted_scans,
                total_deleted_results,
            )

    except Exception as exc:  # noqa: BLE001
        if is_corruption_error(exc):
            logger.critical(
                "DATABASE CORRUPTION in cleanup_old_scans: %s — run scripts/check_db_integrity.py --repair",
                exc,
            )
        else:
            logger.error("Error cleaning up old scans: %s", exc, exc_info=True)
        safe_rollback(db)


def compute_industry_peer_metrics(db: Session, scan_id: str) -> None:
    """Compute aggregate metrics for each industry group in a scan."""
    from sqlalchemy import func

    from app.models.industry import IBDGroupPeerCache

    try:
        groups = db.query(
            ScanResult.ibd_industry_group,
            func.count(ScanResult.symbol).label("total_stocks"),
            func.avg(ScanResult.rs_rating_1m).label("avg_rs_1m"),
            func.avg(ScanResult.rs_rating_3m).label("avg_rs_3m"),
            func.avg(ScanResult.rs_rating_12m).label("avg_rs_12m"),
            func.avg(ScanResult.minervini_score).label("avg_minervini_score"),
            func.avg(ScanResult.composite_score).label("avg_composite_score"),
        ).filter(
            ScanResult.scan_id == scan_id,
            ScanResult.ibd_industry_group.isnot(None),
        ).group_by(
            ScanResult.ibd_industry_group
        ).all()

        for group in groups:
            top = db.query(ScanResult).filter(
                ScanResult.scan_id == scan_id,
                ScanResult.ibd_industry_group == group.ibd_industry_group,
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
                top_score=top.composite_score if top else None,
            )
            db.add(cache)

        db.commit()
    except Exception as exc:  # noqa: BLE001
        if is_corruption_error(exc):
            logger.critical(
                "DATABASE CORRUPTION in compute_industry_peer_metrics: %s — run scripts/check_db_integrity.py --repair",
                exc,
            )
        else:
            logger.error("Error computing peer metrics: %s", exc, exc_info=True)
        safe_rollback(db)


def _log_setup_engine_distribution(db: Session, scan_id: str) -> None:
    """Log setup-engine score distribution for observability."""
    try:
        from sqlalchemy import func, select

        scores_rows = db.execute(
            select(
                func.json_extract(ScanResult.details, "$.setup_engine.setup_score"),
                func.json_extract(ScanResult.details, "$.setup_engine.setup_ready"),
            ).where(
                ScanResult.scan_id == scan_id,
                func.json_extract(ScanResult.details, "$.setup_engine.setup_score").isnot(None),
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
            scan_id,
            n,
            min(scores),
            max(scores),
            sum(scores) / n,
            median,
            ready_count,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("SE distribution telemetry skipped: %s", exc)


def run_post_scan_pipeline(scan_id: str) -> None:
    """Post-scan pipeline used by both Celery and local desktop runs."""
    db = SessionLocal()
    try:
        compute_industry_peer_metrics(db, scan_id)
        _log_setup_engine_distribution(db, scan_id)
        scan = db.query(Scan).filter(Scan.scan_id == scan_id).first()
        if scan and scan.universe_key:
            cleanup_old_scans(db, scan.universe_key)
        try:
            from app.tasks.cache_tasks import prewarm_chart_cache_for_scan

            prewarm_chart_cache_for_scan.delay(scan_id, top_n=50)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chart cache warming failed: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.error("Post-scan pipeline error for %s: %s", scan_id, exc, exc_info=True)
    finally:
        db.close()


def run_bulk_scan_via_use_case(task_instance, scan_id: str, symbol_list: list[str], criteria: dict | None):
    """Run a scan via the use-case path without importing Celery task modules."""
    from app.infra.db.uow import SqlUnitOfWork
    from app.infra.tasks.cancellation import DbCancellationToken
    from app.infra.tasks.progress_sink import CeleryProgressSink
    from app.use_cases.scanning.run_bulk_scan import RunBulkScanCommand
    from app.wiring.bootstrap import get_run_bulk_scan_use_case

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
        run_post_scan_pipeline(scan_id)

    return {
        "scan_id": result.scan_id,
        "completed": result.total_scanned,
        "passed": result.passed,
        "failed": result.failed,
        "status": result.status,
        "scan_path": "use_case",
    }
