"""Unit tests for the _run_bulk_scan_via_use_case wrapper function.

Verifies that the thin Celery wrapper correctly wires adapters
and delegates to the use case.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch, call

import pytest

from app.use_cases.scanning.run_bulk_scan import RunBulkScanCommand, RunBulkScanResult


@dataclass
class _FakeResult:
    """Minimal stand-in for RunBulkScanResult."""
    scan_id: str = "scan-001"
    status: str = "completed"
    total_scanned: int = 10
    passed: int = 7
    failed: int = 3


# All patches target the lazy-import locations inside scan_tasks module
_WRAPPER_PATH = "app.tasks.scan_tasks"


class TestRunBulkScanViaUseCase:
    """Test the thin wrapper that delegates to RunBulkScanUseCase."""

    @patch(f"{_WRAPPER_PATH}._run_post_scan_pipeline")
    @patch(f"{_WRAPPER_PATH}.settings")
    def test_happy_path_returns_correct_dict(self, mock_settings, mock_pipeline):
        mock_settings.scan_usecase_chunk_size = 25

        fake_result = _FakeResult()
        mock_use_case = MagicMock()
        mock_use_case.execute.return_value = fake_result

        mock_uow_cls = MagicMock()
        mock_progress_cls = MagicMock()
        mock_cancel_cls = MagicMock()
        mock_cancel_instance = mock_cancel_cls.return_value

        task_instance = MagicMock()
        task_instance.request.id = "celery-task-id-123"

        with (
            patch(f"{_WRAPPER_PATH}.SessionLocal") as mock_session_local,
            patch("app.wiring.bootstrap.get_run_bulk_scan_use_case", return_value=mock_use_case),
            patch("app.infra.db.uow.SqlUnitOfWork", mock_uow_cls),
            patch("app.infra.tasks.progress_sink.CeleryProgressSink", mock_progress_cls),
            patch("app.infra.tasks.cancellation.DbCancellationToken", mock_cancel_cls),
        ):
            from app.tasks.scan_tasks import _run_bulk_scan_via_use_case

            result = _run_bulk_scan_via_use_case(
                task_instance, "scan-001", ["AAPL", "MSFT"], {"include_vcp": True}
            )

        assert result == {
            "scan_id": "scan-001",
            "completed": 10,
            "passed": 7,
            "failed": 3,
            "status": "completed",
            "scan_path": "use_case",
        }

    @patch(f"{_WRAPPER_PATH}._run_post_scan_pipeline")
    @patch(f"{_WRAPPER_PATH}.settings")
    def test_post_scan_pipeline_called_on_completed(self, mock_settings, mock_pipeline):
        mock_settings.scan_usecase_chunk_size = 25

        mock_use_case = MagicMock()
        mock_use_case.execute.return_value = _FakeResult(status="completed")

        task_instance = MagicMock()
        task_instance.request.id = "task-id"

        with (
            patch(f"{_WRAPPER_PATH}.SessionLocal"),
            patch("app.wiring.bootstrap.get_run_bulk_scan_use_case", return_value=mock_use_case),
            patch("app.infra.db.uow.SqlUnitOfWork"),
            patch("app.infra.tasks.progress_sink.CeleryProgressSink"),
            patch("app.infra.tasks.cancellation.DbCancellationToken"),
        ):
            from app.tasks.scan_tasks import _run_bulk_scan_via_use_case
            _run_bulk_scan_via_use_case(task_instance, "scan-001", ["AAPL"], {})

        mock_pipeline.assert_called_once_with("scan-001")

    @patch(f"{_WRAPPER_PATH}._run_post_scan_pipeline")
    @patch(f"{_WRAPPER_PATH}.settings")
    def test_post_scan_pipeline_skipped_on_cancelled(self, mock_settings, mock_pipeline):
        mock_settings.scan_usecase_chunk_size = 25

        mock_use_case = MagicMock()
        mock_use_case.execute.return_value = _FakeResult(status="cancelled")

        task_instance = MagicMock()
        task_instance.request.id = "task-id"

        with (
            patch(f"{_WRAPPER_PATH}.SessionLocal"),
            patch("app.wiring.bootstrap.get_run_bulk_scan_use_case", return_value=mock_use_case),
            patch("app.infra.db.uow.SqlUnitOfWork"),
            patch("app.infra.tasks.progress_sink.CeleryProgressSink"),
            patch("app.infra.tasks.cancellation.DbCancellationToken"),
        ):
            from app.tasks.scan_tasks import _run_bulk_scan_via_use_case
            _run_bulk_scan_via_use_case(task_instance, "scan-001", ["AAPL"], {})

        mock_pipeline.assert_not_called()

    @patch(f"{_WRAPPER_PATH}._run_post_scan_pipeline")
    @patch(f"{_WRAPPER_PATH}.settings")
    def test_cancel_close_called_even_on_exception(self, mock_settings, mock_pipeline):
        mock_settings.scan_usecase_chunk_size = 25

        mock_use_case = MagicMock()
        mock_use_case.execute.side_effect = RuntimeError("boom")

        mock_cancel_instance = MagicMock()

        task_instance = MagicMock()
        task_instance.request.id = "task-id"

        with (
            patch(f"{_WRAPPER_PATH}.SessionLocal"),
            patch("app.wiring.bootstrap.get_run_bulk_scan_use_case", return_value=mock_use_case),
            patch("app.infra.db.uow.SqlUnitOfWork"),
            patch("app.infra.tasks.progress_sink.CeleryProgressSink"),
            patch("app.infra.tasks.cancellation.DbCancellationToken", return_value=mock_cancel_instance),
        ):
            from app.tasks.scan_tasks import _run_bulk_scan_via_use_case

            with pytest.raises(RuntimeError, match="boom"):
                _run_bulk_scan_via_use_case(task_instance, "scan-001", ["AAPL"], {})

        mock_cancel_instance.close.assert_called_once()

    @patch(f"{_WRAPPER_PATH}._run_post_scan_pipeline")
    @patch(f"{_WRAPPER_PATH}.settings")
    def test_command_built_with_correct_params(self, mock_settings, mock_pipeline):
        mock_settings.scan_usecase_chunk_size = 25

        captured_cmd = None

        def capture_execute(uow, cmd, progress, cancel):
            nonlocal captured_cmd
            captured_cmd = cmd
            return _FakeResult()

        mock_use_case = MagicMock()
        mock_use_case.execute.side_effect = capture_execute

        task_instance = MagicMock()
        task_instance.request.id = "celery-123"

        with (
            patch(f"{_WRAPPER_PATH}.SessionLocal"),
            patch("app.wiring.bootstrap.get_run_bulk_scan_use_case", return_value=mock_use_case),
            patch("app.infra.db.uow.SqlUnitOfWork"),
            patch("app.infra.tasks.progress_sink.CeleryProgressSink"),
            patch("app.infra.tasks.cancellation.DbCancellationToken"),
        ):
            from app.tasks.scan_tasks import _run_bulk_scan_via_use_case
            _run_bulk_scan_via_use_case(
                task_instance, "scan-001", ["AAPL", "MSFT"],
                {"include_vcp": True}
            )

        assert captured_cmd.scan_id == "scan-001"
        assert captured_cmd.symbols == ["AAPL", "MSFT"]
        assert captured_cmd.criteria == {"include_vcp": True}
        assert captured_cmd.chunk_size == 25
        assert captured_cmd.correlation_id == "celery-123"

    @patch(f"{_WRAPPER_PATH}._run_post_scan_pipeline")
    @patch(f"{_WRAPPER_PATH}.settings")
    def test_none_criteria_defaults_to_empty_dict(self, mock_settings, mock_pipeline):
        """When criteria is None, the wrapper passes {} to the command."""
        mock_settings.scan_usecase_chunk_size = 25

        captured_cmd = None

        def capture_execute(uow, cmd, progress, cancel):
            nonlocal captured_cmd
            captured_cmd = cmd
            return _FakeResult()

        mock_use_case = MagicMock()
        mock_use_case.execute.side_effect = capture_execute

        task_instance = MagicMock()
        task_instance.request.id = "task-id"

        with (
            patch(f"{_WRAPPER_PATH}.SessionLocal"),
            patch("app.wiring.bootstrap.get_run_bulk_scan_use_case", return_value=mock_use_case),
            patch("app.infra.db.uow.SqlUnitOfWork"),
            patch("app.infra.tasks.progress_sink.CeleryProgressSink"),
            patch("app.infra.tasks.cancellation.DbCancellationToken"),
        ):
            from app.tasks.scan_tasks import _run_bulk_scan_via_use_case
            _run_bulk_scan_via_use_case(task_instance, "scan-001", ["AAPL"], None)

        assert captured_cmd.criteria == {}


class TestPostScanPipeline:
    """Test the extracted _run_post_scan_pipeline function."""

    @patch(f"{_WRAPPER_PATH}.SessionLocal")
    @patch(f"{_WRAPPER_PATH}.compute_industry_peer_metrics")
    @patch(f"{_WRAPPER_PATH}.cleanup_old_scans")
    def test_calls_peer_metrics_and_cleanup(
        self, mock_cleanup, mock_peer_metrics, mock_session_local
    ):
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db

        mock_scan = MagicMock()
        mock_scan.universe_key = "all"
        mock_db.query.return_value.filter.return_value.first.return_value = mock_scan

        from app.tasks.scan_tasks import _run_post_scan_pipeline

        with patch(f"{_WRAPPER_PATH}.cache_tasks", create=True):
            _run_post_scan_pipeline("scan-001")

        mock_peer_metrics.assert_called_once_with(mock_db, "scan-001")
        mock_cleanup.assert_called_once_with(mock_db, "all")
        mock_db.close.assert_called_once()

    @patch(f"{_WRAPPER_PATH}.SessionLocal")
    @patch(f"{_WRAPPER_PATH}.compute_industry_peer_metrics")
    def test_session_closed_on_error(self, mock_peer_metrics, mock_session_local):
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db
        mock_peer_metrics.side_effect = RuntimeError("db error")

        from app.tasks.scan_tasks import _run_post_scan_pipeline
        # Should not raise — errors are caught and logged
        _run_post_scan_pipeline("scan-001")

        mock_db.close.assert_called_once()
