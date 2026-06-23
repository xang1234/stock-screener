from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from celery.exceptions import Retry
from celery.worker.request import Request


SMART_REFRESH_TASK = "app.tasks.cache_tasks.smart_refresh_cache"


def test_publish_runtime_activity_failure_marks_smart_refresh_failed(monkeypatch):
    import app.tasks.runtime_activity_failure_hooks as module

    db = MagicMock()
    failed_activity_calls = []
    lock = MagicMock()
    coordination = MagicMock()
    price_cache = MagicMock()

    monkeypatch.setattr(module, "SessionLocal", lambda: db)
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: lock)
    monkeypatch.setattr(module, "get_workload_coordination", lambda: coordination)
    monkeypatch.setattr(module, "get_price_cache", lambda: price_cache)
    monkeypatch.setattr(
        module,
        "mark_market_activity_failed",
        lambda db_arg, **kwargs: failed_activity_calls.append((db_arg, kwargs)),
    )

    module.publish_runtime_activity_failure(
        SMART_REFRESH_TASK,
        "lost-task",
        {"market": "us", "activity_lifecycle": "bootstrap"},
        RuntimeError("Worker exited prematurely: signal 9 (SIGKILL)"),
    )

    assert failed_activity_calls == [
        (
            db,
            {
                "market": "US",
                "stage_key": "prices",
                "lifecycle": "bootstrap",
                "task_name": SMART_REFRESH_TASK,
                "task_id": "lost-task",
                "message": (
                    "Task worker exited before cleanup: "
                    "Worker exited prematurely: signal 9 (SIGKILL)"
                ),
            },
        )
    ]
    price_cache.complete_warmup_heartbeat.assert_called_once_with("failed", market="US")
    lock.release.assert_called_once_with("lost-task", market="US")
    coordination.release_market_workload.assert_called_once_with("lost-task", market="US")
    coordination.release_external_fetch.assert_called_once_with("lost-task")
    db.close.assert_called_once_with()


def test_publish_runtime_activity_failure_keeps_shared_cleanup_scope_for_missing_market(
    monkeypatch,
):
    import app.tasks.runtime_activity_failure_hooks as module

    db = MagicMock()
    failed_activity_calls = []
    lock = MagicMock()
    coordination = MagicMock()
    price_cache = MagicMock()

    monkeypatch.setattr(module, "SessionLocal", lambda: db)
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: lock)
    monkeypatch.setattr(module, "get_workload_coordination", lambda: coordination)
    monkeypatch.setattr(module, "get_price_cache", lambda: price_cache)
    monkeypatch.setattr(
        module,
        "mark_market_activity_failed",
        lambda db_arg, **kwargs: failed_activity_calls.append((db_arg, kwargs)),
    )

    module.publish_runtime_activity_failure(
        SMART_REFRESH_TASK,
        "shared-task",
        {"mode": "auto"},
        RuntimeError("worker lost"),
    )

    assert failed_activity_calls[0][1]["market"] == "US"
    price_cache.complete_warmup_heartbeat.assert_called_once_with("failed", market=None)
    lock.release.assert_called_once_with("shared-task", market=None)
    coordination.release_market_workload.assert_called_once_with(
        "shared-task",
        market=None,
    )
    coordination.release_external_fetch.assert_called_once_with("shared-task")
    db.close.assert_called_once_with()


def test_publish_runtime_activity_failure_ignores_untracked_tasks(monkeypatch):
    import app.tasks.runtime_activity_failure_hooks as module

    monkeypatch.setattr(
        module,
        "SessionLocal",
        lambda: pytest.fail("untracked tasks must not open a database session"),
    )
    monkeypatch.setattr(
        module,
        "mark_market_activity_failed",
        lambda *args, **kwargs: pytest.fail("untracked tasks must not publish activity"),
    )

    module.publish_runtime_activity_failure(
        "app.tasks.cache_tasks.warm_spy_cache",
        "warmup-task",
        {"market": "US"},
        RuntimeError("boom"),
    )


def test_request_hook_delegates_failure_publication(monkeypatch):
    import app.tasks.runtime_activity_failure_hooks as module

    delegated_calls = []

    monkeypatch.setattr(
        Request,
        "on_failure",
        lambda self, exc_info, send_failed_event=True, return_ok=False: "super-result",
    )
    monkeypatch.setattr(
        module,
        "publish_runtime_activity_failure",
        lambda task_name, task_id, kwargs, exception: delegated_calls.append(
            {
                "task_name": task_name,
                "task_id": task_id,
                "kwargs": kwargs,
                "exception": exception,
            }
        ),
    )

    request = object.__new__(module.RuntimeActivityFailureRequest)
    request._task = SimpleNamespace(name=SMART_REFRESH_TASK)
    request.id = "request-task"
    request._kwargs = {"market": "HK", "activity_lifecycle": "bootstrap"}
    exception = RuntimeError("worker disappeared")

    result = request.on_failure(
        SimpleNamespace(exception=exception),
        send_failed_event=False,
        return_ok=True,
    )

    assert result == "super-result"
    assert delegated_calls == [
        {
            "task_name": SMART_REFRESH_TASK,
            "task_id": "request-task",
            "kwargs": {"market": "HK", "activity_lifecycle": "bootstrap"},
            "exception": exception,
        }
    ]


def test_request_hook_skips_failure_publication_for_retry(monkeypatch):
    import app.tasks.runtime_activity_failure_hooks as module

    delegated_calls = []

    monkeypatch.setattr(
        Request,
        "on_failure",
        lambda self, exc_info, send_failed_event=True, return_ok=False: "retry-result",
    )
    monkeypatch.setattr(
        module,
        "publish_runtime_activity_failure",
        lambda *args, **kwargs: delegated_calls.append((args, kwargs)),
    )

    request = object.__new__(module.RuntimeActivityFailureRequest)
    request._task = SimpleNamespace(name=SMART_REFRESH_TASK)
    request.id = "retry-task"
    request._kwargs = {"market": "US"}

    result = request.on_failure(SimpleNamespace(exception=Retry("retry later")))

    assert result == "retry-result"
    assert delegated_calls == []
