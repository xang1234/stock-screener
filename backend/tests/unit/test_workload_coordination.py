"""Unit tests for background-task lease coordination."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from celery.exceptions import Retry


def _make_coordination():
    with patch("app.tasks.workload_coordination.settings") as mock_settings:
        mock_settings.redis_host = "localhost"
        mock_settings.redis_port = 6379
        mock_settings.redis_db = 0
        mock_settings.data_fetch_lock_timeout = 7200

        with patch("app.tasks.workload_coordination.redis.Redis") as mock_redis_cls:
            mock_redis = MagicMock()
            mock_redis_cls.return_value = mock_redis
            mock_release_script = MagicMock()
            mock_redis.register_script.return_value = mock_release_script

            from app.tasks.workload_coordination import WorkloadCoordination

            coordination = WorkloadCoordination()

    return coordination, mock_redis, mock_release_script


def test_market_workload_lease_uses_market_scoped_key():
    coordination, mock_redis, _ = _make_coordination()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True

    acquired, is_reentrant = coordination.acquire_market_workload(
        "run_bulk_scan",
        "task-123",
        market="US",
    )

    assert (acquired, is_reentrant) == (True, False)
    args, kwargs = mock_redis.set.call_args
    assert args[0] == "market_workload:us"
    assert "run_bulk_scan:task-123:" in args[1]
    assert kwargs == {"nx": True, "ex": 7200}


def test_external_fetch_lease_uses_global_key():
    coordination, mock_redis, _ = _make_coordination()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True

    acquired, is_reentrant = coordination.acquire_external_fetch(
        "smart_refresh_cache",
        "task-123",
    )

    assert (acquired, is_reentrant) == (True, False)
    args, kwargs = mock_redis.set.call_args
    assert args[0] == "external_fetch_global"
    assert "smart_refresh_cache:task-123:" in args[1]
    assert kwargs == {"nx": True, "ex": 7200}


def test_disable_serialized_market_workload_bypasses_coordination():
    from app.tasks.workload_coordination import (
        disable_serialized_market_workload,
        serialized_market_workload,
    )

    task = SimpleNamespace(request=SimpleNamespace(id="task-123"))

    @serialized_market_workload("build_daily_snapshot")
    def my_func(self, market=None):
        return {"status": "ok", "market": market}

    with patch("app.wiring.bootstrap.get_workload_coordination") as mock_get_coordination:
        with disable_serialized_market_workload():
            result = my_func(task, market="HK")

    assert result == {"status": "ok", "market": "HK"}
    mock_get_coordination.assert_not_called()


@patch("app.wiring.bootstrap.get_workload_coordination")
@patch("app.wiring.bootstrap.get_data_fetch_lock")
def test_serialized_data_fetch_retries_when_external_fetch_lease_is_busy(
    mock_get_lock,
    mock_get_coordination,
):
    from app.tasks.data_fetch_lock import serialized_data_fetch

    mock_lock = MagicMock()
    mock_lock.acquire.return_value = (True, False)
    mock_get_lock.return_value = mock_lock

    mock_coordination = MagicMock()
    mock_coordination.acquire_market_workload.return_value = (True, False)
    mock_coordination.acquire_external_fetch.return_value = (False, False)
    mock_coordination.get_external_fetch_holder.return_value = {
        "task_name": "refresh_all_fundamentals",
        "task_id": "other-task",
    }
    mock_get_coordination.return_value = mock_coordination

    retry_calls = []

    def _retry(*, exc=None, countdown=None, max_retries=None):
        retry_calls.append(
            {
                "exc": exc,
                "countdown": countdown,
                "max_retries": max_retries,
            }
        )
        effective_max_retries = 3 if max_retries is None else max_retries
        if task.request.retries + 1 > effective_max_retries:
            raise exc if exc is not None else RuntimeError("max retries exceeded")
        raise Retry(message=str(exc))

    task = SimpleNamespace(
        request=SimpleNamespace(id="task-123", retries=3),
        retry=_retry,
    )

    @serialized_data_fetch("smart_refresh_cache")
    def my_func(self, market=None):
        return "ok"

    with patch("app.tasks.data_fetch_lock.logger") as mock_logger:
        with pytest.raises(Retry):
            my_func(task, market="US")

    assert len(retry_calls) == 1
    assert str(retry_calls[0]["exc"]) == (
        "waiting_for_external_fetch_global (refresh_all_fundamentals)"
    )
    assert retry_calls[0]["countdown"] == 120
    assert retry_calls[0]["max_retries"] is not None
    assert retry_calls[0]["max_retries"] > task.request.retries
    mock_logger.error.assert_not_called()

    mock_coordination.acquire_market_workload.assert_called_once_with(
        "smart_refresh_cache",
        "task-123",
        market="US",
    )
    mock_coordination.acquire_external_fetch.assert_called_once_with(
        "smart_refresh_cache",
        "task-123",
    )
    mock_coordination.release_market_workload.assert_called_once_with(
        "task-123",
        market="US",
    )
    mock_lock.release.assert_called_once_with("task-123", market="US")


@patch("app.wiring.bootstrap.get_workload_coordination")
def test_serialized_market_workload_uses_long_retry_budget_when_lease_is_busy(
    mock_get_coordination,
):
    from app.tasks.workload_coordination import serialized_market_workload

    mock_coordination = MagicMock()
    mock_coordination.acquire_market_workload.return_value = (False, False)
    mock_coordination.get_market_workload_holder.return_value = {
        "task_name": "calculate_daily_breadth",
        "task_id": "other-task",
    }
    mock_get_coordination.return_value = mock_coordination

    retry_calls = []

    def _retry(*, exc=None, countdown=None, max_retries=None):
        retry_calls.append(
            {
                "exc": exc,
                "countdown": countdown,
                "max_retries": max_retries,
            }
        )
        effective_max_retries = 3 if max_retries is None else max_retries
        if task.request.retries + 1 > effective_max_retries:
            raise exc if exc is not None else RuntimeError("max retries exceeded")
        raise Retry(message=str(exc))

    task = SimpleNamespace(
        request=SimpleNamespace(id="task-123", retries=3),
        retry=_retry,
    )

    @serialized_market_workload("run_bulk_scan")
    def my_func(self, market=None):
        return "ok"

    with pytest.raises(Retry):
        my_func(task, market="US")

    assert len(retry_calls) == 1
    assert str(retry_calls[0]["exc"]) == (
        "waiting_for_market_workload:US (calculate_daily_breadth)"
    )
    assert retry_calls[0]["countdown"] == 120
    assert retry_calls[0]["max_retries"] is not None
    assert retry_calls[0]["max_retries"] > task.request.retries
