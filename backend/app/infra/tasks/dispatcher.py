"""Task dispatcher adapter for Celery."""

from __future__ import annotations

from app.domain.scanning.ports import TaskDispatcher
from app.tasks.market_queues import SHARED_USER_SCANS_QUEUE


class CeleryTaskDispatcher(TaskDispatcher):
    """Dispatch scan tasks via Celery."""

    def dispatch_scan(
        self, scan_id: str, symbols: list[str], criteria: dict
    ) -> str:
        from app.tasks.scan_tasks import run_bulk_scan

        task = run_bulk_scan.apply_async(
            args=[scan_id, symbols, criteria],
            queue=SHARED_USER_SCANS_QUEUE,
        )
        return task.id
