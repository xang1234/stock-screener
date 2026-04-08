"""Task dispatcher adapter for Celery."""

from __future__ import annotations

from app.domain.scanning.ports import TaskDispatcher


class CeleryTaskDispatcher(TaskDispatcher):
    """Dispatch scan tasks via Celery."""

    def dispatch_scan(
        self, scan_id: str, symbols: list[str], criteria: dict
    ) -> str:
        from app.tasks.scan_tasks import run_bulk_scan

        task = run_bulk_scan.delay(scan_id, symbols, criteria)
        return task.id
