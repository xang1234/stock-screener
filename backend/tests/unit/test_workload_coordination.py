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

    def _retry(*, exc=None, countdown=None, max_retries=None):
        raise Retry(message=str(exc))

    task = SimpleNamespace(
        request=SimpleNamespace(id="task-123", retries=0),
        retry=_retry,
    )

    @serialized_data_fetch("smart_refresh_cache")
    def my_func(self, market=None):
        return "ok"

    with pytest.raises(Retry):
        my_func(task, market="US")

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
