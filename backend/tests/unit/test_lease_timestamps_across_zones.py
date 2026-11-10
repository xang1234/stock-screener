"""Lease and heartbeat timestamps cross processes in different zones.

Docker workers run with TZ=America/New_York while the API runs in UTC, so a
timestamp one writes must read with its true age in the other.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
import time
from unittest.mock import MagicMock

import pytest

from app.celery_app import _startup_heartbeat_is_stale
from app.services.cache.price_cache_warmup import PriceCacheWarmupStore
from app.services.operations_job_service import _age_seconds_from
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
