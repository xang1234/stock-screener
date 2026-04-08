"""Background job backend for scan execution via Celery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


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


