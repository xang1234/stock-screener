from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock


def test_run_post_scan_pipeline_uses_session_factory_and_closes(monkeypatch):
    import app.services.scan_execution as module

    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.first.return_value = SimpleNamespace(universe_key="all")
    session_factory = MagicMock(return_value=fake_db)

    peer_metrics = MagicMock()
    setup_distribution = MagicMock()
    cleanup = MagicMock()
    prewarm_delay = MagicMock()

    monkeypatch.setattr(module, "compute_industry_peer_metrics", peer_metrics)
    monkeypatch.setattr(module, "_log_setup_engine_distribution", setup_distribution)
    monkeypatch.setattr(module, "cleanup_old_scans", cleanup)
    monkeypatch.setitem(
        sys.modules,
        "app.tasks.cache_tasks",
        SimpleNamespace(prewarm_chart_cache_for_scan=SimpleNamespace(delay=prewarm_delay)),
    )

    module.run_post_scan_pipeline("scan-001", session_factory=session_factory)

    session_factory.assert_called_once_with()
    peer_metrics.assert_called_once_with(fake_db, "scan-001")
    setup_distribution.assert_called_once_with(fake_db, "scan-001")
    cleanup.assert_called_once_with(fake_db, "all")
    prewarm_delay.assert_called_once_with("scan-001", top_n=50)
    fake_db.close.assert_called_once_with()


def test_run_bulk_scan_via_use_case_threads_session_factory(monkeypatch):
    import app.infra.db.uow as uow_module
    import app.infra.tasks.cancellation as cancellation_module
    import app.infra.tasks.progress_sink as progress_module
    import app.services.scan_execution as module
    import app.wiring.bootstrap as bootstrap

    fake_result = SimpleNamespace(
        scan_id="scan-001",
        status="completed",
        total_scanned=10,
        passed=7,
        failed=3,
    )

    mock_use_case = MagicMock()
    mock_use_case.execute.return_value = fake_result
    task_instance = MagicMock()
    task_instance.request.id = "task-id-1"

    session_factory = MagicMock()
    cancel_instance = MagicMock()
    cancel_ctor = MagicMock(return_value=cancel_instance)
    uow_ctor = MagicMock(return_value=object())
    progress_ctor = MagicMock(return_value=object())
    post_pipeline = MagicMock()

    monkeypatch.setattr(bootstrap, "get_run_bulk_scan_use_case", lambda: mock_use_case)
    monkeypatch.setattr(cancellation_module, "DbCancellationToken", cancel_ctor)
    monkeypatch.setattr(uow_module, "SqlUnitOfWork", uow_ctor)
    monkeypatch.setattr(progress_module, "CeleryProgressSink", progress_ctor)
    monkeypatch.setattr(module, "run_post_scan_pipeline", post_pipeline)

    result = module.run_bulk_scan_via_use_case(
        task_instance,
        "scan-001",
        ["AAPL", "MSFT"],
        {"include_vcp": True},
        session_factory=session_factory,
    )

    cancel_ctor.assert_called_once_with(session_factory, "scan-001")
    uow_ctor.assert_called_once_with(session_factory)
    post_pipeline.assert_called_once_with("scan-001", session_factory=session_factory)
    cancel_instance.close.assert_called_once_with()
    assert result == {
        "scan_id": "scan-001",
        "completed": 10,
        "passed": 7,
        "failed": 3,
        "status": "completed",
        "scan_path": "use_case",
    }
