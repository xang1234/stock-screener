"""Unit tests for CeleryProgressSink infra adapter.

Verifies that domain ProgressEvents are correctly translated into
Celery task state updates, and that errors are swallowed (progress
is non-critical).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.domain.scanning.models import ProgressEvent
from app.infra.tasks.progress_sink import CeleryProgressSink


class TestCeleryProgressSink:
    """Test the Celery adapter for the ProgressSink port."""

    def _make_event(self, **overrides) -> ProgressEvent:
        defaults = dict(
            current=50,
            total=100,
            passed=30,
            failed=10,
            throughput=2.5,
            eta_seconds=20,
        )
        defaults.update(overrides)
        return ProgressEvent(**defaults)

    def test_emit_calls_update_state_with_progress(self):
        task = MagicMock()
        sink = CeleryProgressSink(task)

        sink.emit(self._make_event())

        task.update_state.assert_called_once_with(
            state="PROGRESS",
            meta={
                "current": 50,
                "total": 100,
                "percent": 50.0,
                "passed": 30,
                "failed": 10,
                "throughput": 2.5,
                "eta_seconds": 20,
            },
        )

    def test_emit_calculates_percent_correctly(self):
        task = MagicMock()
        sink = CeleryProgressSink(task)

        sink.emit(self._make_event(current=1, total=3))

        meta = task.update_state.call_args.kwargs["meta"]
        assert abs(meta["percent"] - 33.333333) < 0.01

    def test_emit_handles_zero_total(self):
        task = MagicMock()
        sink = CeleryProgressSink(task)

        sink.emit(self._make_event(current=0, total=0))

        meta = task.update_state.call_args.kwargs["meta"]
        assert meta["percent"] == 0.0

    def test_emit_swallows_update_state_exception(self):
        task = MagicMock()
        task.update_state.side_effect = RuntimeError("Redis down")
        sink = CeleryProgressSink(task)

        # Should NOT raise
        sink.emit(self._make_event())

    def test_emit_passes_none_throughput_and_eta(self):
        task = MagicMock()
        sink = CeleryProgressSink(task)

        sink.emit(self._make_event(throughput=None, eta_seconds=None))

        meta = task.update_state.call_args.kwargs["meta"]
        assert meta["throughput"] is None
        assert meta["eta_seconds"] is None
