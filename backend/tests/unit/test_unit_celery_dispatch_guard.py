"""Guards that keep unit tests from dispatching real Celery broker messages."""
from __future__ import annotations

from celery.app.task import Task
import pytest


def test_unit_suite_replaces_celery_apply_async() -> None:
    assert getattr(Task.apply_async, "_stockscreener_unit_stub", False) is True


def test_unmocked_celery_delay_does_not_run_task_body(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.tasks.scan_tasks import finalize_scan_artifacts

    def fail_if_executed(*_args, **_kwargs):
        pytest.fail("unit Celery dispatch should not execute task bodies")

    monkeypatch.setattr(finalize_scan_artifacts, "run", fail_if_executed)

    result = finalize_scan_artifacts.delay("scan-001")

    assert result.id.startswith("unit-test-app.tasks.scan_tasks.finalize_scan_artifacts-")
    assert result.state == "PENDING"
