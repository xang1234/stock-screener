"""Celery implementation of the TaskDispatcher port."""

from __future__ import annotations

from app.domain.scanning.ports import TaskDispatcher
from app.tasks.scan_tasks import run_bulk_scan
from app.services.job_backend import LocalJobBackend


class CeleryTaskDispatcher(TaskDispatcher):
    """Dispatch scan tasks via Celery."""

    def dispatch_scan(
        self, scan_id: str, symbols: list[str], criteria: dict
    ) -> str:
        task = run_bulk_scan.delay(scan_id, symbols, criteria)
        return task.id


class LocalTaskDispatcher(TaskDispatcher):
    """Dispatch scan tasks through the in-process desktop job backend."""

    def __init__(self, job_backend: LocalJobBackend) -> None:
        self._job_backend = job_backend

    def dispatch_scan(
        self, scan_id: str, symbols: list[str], criteria: dict
    ) -> str:
        return self._job_backend.submit_scan(scan_id, symbols, criteria)
