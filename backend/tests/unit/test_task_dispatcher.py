"""Unit tests for Celery task dispatch routing."""

from __future__ import annotations

from unittest.mock import patch


def test_dispatch_scan_routes_market_scans_to_market_queue():
    from app.infra.tasks.dispatcher import CeleryTaskDispatcher

    class _FakeAsyncResult:
        id = "task-123"

    with patch("app.tasks.scan_tasks.run_bulk_scan.apply_async", return_value=_FakeAsyncResult()) as mock_apply:
        dispatcher = CeleryTaskDispatcher()
        task_id = dispatcher.dispatch_scan(
            "scan-001",
            ["0700.HK"],
            {"min_price": 10},
            market="HK",
        )

    assert task_id == "task-123"
    mock_apply.assert_called_once_with(
        args=["scan-001", ["0700.HK"], {"min_price": 10}],
        kwargs={"market": "HK"},
        queue="user_scans_hk",
    )
