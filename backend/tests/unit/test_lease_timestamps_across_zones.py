"""Lease and heartbeat timestamps cross processes in different zones.

Docker workers run with TZ=America/New_York while the API runs in UTC, so a
timestamp one writes must read with its true age in the other.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from app.celery_app import _startup_heartbeat_is_stale
from app.services.cache.price_cache_warmup import PriceCacheWarmupStore
from app.services.operations_job_service import _age_seconds_from
from app.services.price_cache_service import PriceCacheService
from tests.unit.test_data_fetch_lock import _make_lock

pytestmark = pytest.mark.skipif(not hasattr(time, "tzset"), reason="needs time.tzset")


@contextmanager
def _process_tz(name):
    previous = os.environ.get("TZ")
    os.environ["TZ"] = name
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


def _store():
    redis_client = MagicMock()
    store = PriceCacheWarmupStore(
        logger=MagicMock(),
        redis_client=redis_client,
        metadata_key="cache:warmup:metadata",
        heartbeat_key="cache:warmup:heartbeat",
    )
    return store, redis_client


def test_a_fresh_worker_heartbeat_reads_fresh_in_the_api():
    store, redis_client = _store()
    with _process_tz("America/New_York"):
        store.update_warmup_heartbeat(current=1, total=10, market="US")
    payload = redis_client.setex.call_args.args[2]
    redis_client.get.return_value = payload
    updated_at = json.loads(payload)["updated_at"]

    with _process_tz("UTC"):
        assert store.get_heartbeat_info(market="US")["minutes"] < 1
        assert _age_seconds_from(updated_at) < 60
        assert _startup_heartbeat_is_stale(updated_at) is False


def test_a_just_completed_worker_heartbeat_reads_fresh_in_the_api():
    store, redis_client = _store()
    redis_client.get.return_value = None
    with _process_tz("America/New_York"):
        store.complete_warmup_heartbeat("completed", market="US")
    redis_client.get.return_value = redis_client.setex.call_args.args[2]

    with _process_tz("UTC"):
        assert store.get_heartbeat_info(market="US")["minutes"] < 1


def test_a_lock_a_worker_just_took_reads_fresh_in_the_api():
    lock, mock_redis, _, _ = _make_lock()
    mock_redis.set.return_value = True
    with _process_tz("America/New_York"):
        lock.acquire("refresh", "task-1", market="US")
    mock_redis.get.return_value = mock_redis.set.call_args.args[1].encode()

    with _process_tz("UTC"):
        holder = lock.get_current_holder(market="US")

    assert holder["task_id"] == "task-1"
    assert _age_seconds_from(holder["started_at"]) < 60


def _health_with_lock_held(started_at, warmup_meta):
    service = PriceCacheService(redis_client=MagicMock(), session_factory=MagicMock())
    service._get_heartbeat_info = lambda *a, **k: None
    service._get_warmup_metadata = lambda: warmup_meta
    service._get_spy_last_date = lambda: None
    service._get_expected_data_date = lambda market: None
    lock = MagicMock()
    lock.get_any_current_holder.return_value = {
        "task_id": "task-1",
        "task_name": "refresh",
        "started_at": started_at,
    }
    with patch("app.wiring.bootstrap.get_data_fetch_lock", return_value=lock):
        return service.get_cache_health_status(), lock


def test_a_lock_just_taken_without_heartbeat_reads_as_starting():
    started_at = datetime.now(timezone.utc).isoformat()
    status, _ = _health_with_lock_held(started_at, None)

    assert status["status"] == "updating"
    assert status["task_running"]["started_at"] == started_at


@pytest.mark.parametrize("aware_completion", [True, False])
def test_a_lock_outlived_by_its_completed_warmup_is_released(aware_completion):
    now = datetime.now(timezone.utc)
    completed_at = now - timedelta(minutes=1)
    if not aware_completion:
        completed_at = completed_at.replace(tzinfo=None)
    status, lock = _health_with_lock_held(
        (now - timedelta(minutes=10)).isoformat(),
        {"status": "completed", "completed_at": completed_at.isoformat()},
    )

    lock.force_release_all.assert_called_once()
    assert status["status"] == "stale"
