"""Background job backends for scan execution and desktop bootstrap."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import logging
import threading
import uuid
from typing import Any, Callable

from app.config import settings

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobSnapshot:
    """In-memory representation of an async job."""

    job_id: str
    job_type: str
    status: str
    message: str | None = None
    current: int | None = None
    total: int | None = None
    percent: float | None = None
    passed: int | None = None
    failed: int | None = None
    throughput: float | None = None
    eta_seconds: int | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobBackend:
    """Read-only interface for querying async job status."""

    def get_status(self, job_id: str) -> JobSnapshot | None:  # pragma: no cover - interface
        raise NotImplementedError


class CeleryJobBackend(JobBackend):
    """Job backend that reads status from Celery AsyncResult."""

    def get_status(self, job_id: str) -> JobSnapshot | None:
        from celery.result import AsyncResult

        result = AsyncResult(job_id)
        state = result.state
        status = {
            "PENDING": "queued",
            "STARTED": "running",
            "PROGRESS": "running",
            "SUCCESS": "completed",
            "FAILURE": "failed",
            "REVOKED": "failed",
        }.get(state, state.lower())

        snapshot = JobSnapshot(
            job_id=job_id,
            job_type="celery",
            status=status,
        )

        info = result.info if isinstance(result.info, dict) else {}
        if state == "PROGRESS":
            snapshot.current = info.get("current")
            snapshot.total = info.get("total")
            snapshot.percent = info.get("percent")
            snapshot.passed = info.get("passed")
            snapshot.failed = info.get("failed")
            snapshot.throughput = info.get("throughput")
            snapshot.eta_seconds = info.get("eta_seconds")
        elif state == "SUCCESS":
            task_result = result.result if isinstance(result.result, dict) else {}
            result_status = task_result.get("status")
            if result_status in {"completed", "cancelled", "failed"}:
                snapshot.status = result_status
            snapshot.result = task_result or None
            snapshot.current = task_result.get("completed")
            snapshot.total = task_result.get("completed")
            snapshot.passed = task_result.get("passed")
            snapshot.failed = task_result.get("failed")
            if snapshot.current is not None and snapshot.total:
                snapshot.percent = 100.0
        elif state == "FAILURE":
            snapshot.error = str(result.result) if result.result else "Task failed"

        return snapshot


class _LocalTaskRequest:
    def __init__(self, job_id: str) -> None:
        self.id = job_id


class _LocalCeleryTaskShim:
    """Compatibility shim for CeleryProgressSink in desktop mode."""

    def __init__(self, backend: "LocalJobBackend", job_id: str) -> None:
        self.request = _LocalTaskRequest(job_id)
        self._backend = backend
        self._job_id = job_id

    def update_state(self, state: str, meta: dict[str, Any] | None = None) -> None:
        self._backend.update_from_celery(self._job_id, state=state, meta=meta or {})


class LocalJobBackend(JobBackend):
    """Single-worker in-process job runner for desktop mode."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="stockscanner-local-jobs",
        )
        self._lock = threading.RLock()
        self._jobs: dict[str, JobSnapshot] = {}

    def get_status(self, job_id: str) -> JobSnapshot | None:
        with self._lock:
            snapshot = self._jobs.get(job_id)
            if snapshot is None:
                return None
            return JobSnapshot(**snapshot.to_dict())

    def submit_scan(self, scan_id: str, symbols: list[str], criteria: dict) -> str:
        job_id = self._create_job(
            job_type="scan",
            message=f"Scan queued for {len(symbols)} symbols",
            total=len(symbols),
        )
        self._executor.submit(self._run_scan, job_id, scan_id, symbols, criteria)
        return job_id

    def submit_job(
        self,
        job_type: str,
        runner: Callable[[str], dict[str, Any] | None],
        *,
        message: str,
        total: int | None = None,
    ) -> str:
        job_id = self._create_job(job_type=job_type, message=message, total=total)
        self._executor.submit(self._run_generic, job_id, runner)
        return job_id

    def update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        message: str | None = None,
        current: int | None = None,
        total: int | None = None,
        percent: float | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            snapshot = self._jobs[job_id]
            if status is not None:
                snapshot.status = status
                if status == "running" and snapshot.started_at is None:
                    snapshot.started_at = _utc_now_iso()
                if status in {"completed", "failed", "cancelled"}:
                    snapshot.completed_at = _utc_now_iso()
            if message is not None:
                snapshot.message = message
            if current is not None:
                snapshot.current = current
            if total is not None:
                snapshot.total = total
            if percent is not None:
                snapshot.percent = percent
            if result is not None:
                snapshot.result = result
            if error is not None:
                snapshot.error = error

    def update_from_celery(self, job_id: str, *, state: str, meta: dict[str, Any]) -> None:
        if state != "PROGRESS":
            return
        with self._lock:
            snapshot = self._jobs[job_id]
            snapshot.status = "running"
            snapshot.started_at = snapshot.started_at or _utc_now_iso()
            snapshot.current = meta.get("current")
            snapshot.total = meta.get("total")
            snapshot.percent = meta.get("percent")
            snapshot.passed = meta.get("passed")
            snapshot.failed = meta.get("failed")
            snapshot.throughput = meta.get("throughput")
            snapshot.eta_seconds = meta.get("eta_seconds")
            snapshot.message = meta.get("message") or snapshot.message

    def _create_job(self, *, job_type: str, message: str, total: int | None = None) -> str:
        job_id = str(uuid.uuid4())
        with self._lock:
            self._jobs[job_id] = JobSnapshot(
                job_id=job_id,
                job_type=job_type,
                status="queued",
                message=message,
                total=total,
            )
        return job_id

    def _run_scan(self, job_id: str, scan_id: str, symbols: list[str], criteria: dict) -> None:
        self.update(job_id, status="running", message="Preparing scan job")
        shim = _LocalCeleryTaskShim(self, job_id)
        try:
            from app.services.scan_execution import run_bulk_scan_via_use_case

            result = run_bulk_scan_via_use_case(shim, scan_id, symbols, criteria)
            status = result.get("status", "completed")
            completed = result.get("completed", len(symbols))
            percent = (completed / len(symbols) * 100) if symbols else 100.0
            with self._lock:
                snapshot = self._jobs[job_id]
                snapshot.status = status
                snapshot.message = f"Scan {status}"
                snapshot.current = completed
                snapshot.total = len(symbols)
                snapshot.percent = percent
                snapshot.passed = result.get("passed")
                snapshot.failed = result.get("failed")
                snapshot.result = result
                snapshot.completed_at = _utc_now_iso()
        except Exception as exc:
            logger.error("Local scan job %s failed", job_id, exc_info=True)
            self.update(
                job_id,
                status="failed",
                message="Scan failed",
                error=str(exc),
            )

    def _run_generic(self, job_id: str, runner: Callable[[str], dict[str, Any] | None]) -> None:
        self.update(job_id, status="running")
        try:
            result = runner(job_id) or {}
            status = result.get("status", "completed")
            if status == "error":
                status = "failed"
            self.update(
                job_id,
                status=status,
                message=result.get("message"),
                current=result.get("current"),
                total=result.get("total"),
                percent=result.get("percent"),
                result=result,
                error=result.get("error"),
            )
        except Exception as exc:
            logger.error("Local job %s failed", job_id, exc_info=True)
            self.update(
                job_id,
                status="failed",
                message="Job failed",
                error=str(exc),
            )


def create_job_backend() -> JobBackend:
    """Factory used by the wiring bootstrap."""
    if settings.desktop_mode:
        return LocalJobBackend()
    return CeleryJobBackend()
