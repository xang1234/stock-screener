from __future__ import annotations

import time

from app.services.job_backend import LocalJobBackend


def test_local_job_backend_runs_scan_job(monkeypatch):
    import app.tasks.scan_tasks as scan_tasks

    def fake_run(task_instance, scan_id, symbols, criteria):
        task_instance.update_state(
            state="PROGRESS",
            meta={"current": 2, "total": len(symbols), "percent": 66.7},
        )
        return {
            "status": "completed",
            "completed": len(symbols),
            "passed": 2,
            "failed": 1,
        }

    monkeypatch.setattr(scan_tasks, "_run_bulk_scan_via_use_case", fake_run)

    backend = LocalJobBackend()
    job_id = backend.submit_scan("scan-1", ["AAPL", "MSFT", "NVDA"], {})

    deadline = time.time() + 5
    snapshot = backend.get_status(job_id)
    while snapshot is not None and snapshot.status != "completed" and time.time() < deadline:
        time.sleep(0.05)
        snapshot = backend.get_status(job_id)

    assert snapshot is not None
    assert snapshot.status == "completed"
    assert snapshot.current == 3
    assert snapshot.total == 3
    assert snapshot.passed == 2
    assert snapshot.failed == 1
    assert snapshot.percent == 100.0
