"""GET /v1/cache/health judges a held lock by that lock's own market scope.

Real DataFetchLock and warmup store over a dict-backed Redis.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from app.services.price_cache_service import PriceCacheService
from app.tasks.data_fetch_lock import _RELEASE_LUA, DataFetchLock, _lock_key_for_market


class _DictRedis:
    """Just enough Redis for DataFetchLock and the warmup store."""

    def __init__(self):
        self.values: dict[str, str] = {}

    def ping(self):
        return True

    def get(self, key):
        value = self.values.get(key)
        return None if value is None else value.encode()

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.values:
            return None
        self.values[key] = value
        return True

    def setex(self, key, _ttl, value):
        self.values[key] = value

    def delete(self, *keys):
        return sum(self.values.pop(key, None) is not None for key in keys)

    def ttl(self, key):
        return 600 if key in self.values else -2

    def register_script(self, lua):
        def release(keys, args):
            if args[0] in self.values.get(keys[0], ""):
                return self.delete(keys[0])
            return 0

        return release if lua == _RELEASE_LUA else None


@pytest.fixture
def redis():
    return _DictRedis()


@pytest.fixture
def lock(redis):
    with patch("app.tasks.data_fetch_lock.redis.Redis", return_value=redis):
        return DataFetchLock()


@pytest.fixture
def service(redis):
    return PriceCacheService(redis_client=redis, session_factory=lambda: None)


def _health(service, lock):
    with patch("app.wiring.bootstrap.get_data_fetch_lock", return_value=lock):
        return service.get_cache_health_status()


def _holder_id(lock, market):
    return (lock.get_current_holder(market=market) or {}).get("task_id")


def test_hk_refresh_is_judged_by_its_own_heartbeat_not_the_shared_one(service, lock):
    lock.acquire("refresh_hk", "hk-1", market="HK")
    service.complete_warmup_heartbeat("completed")  # an earlier shared run
    service.update_warmup_heartbeat(5, 10, market="HK")

    status = _health(service, lock)

    assert status["status"] == "updating"
    assert status["task_running"]["task_id"] == "hk-1"
    assert status["task_running"]["current"] == 5
    assert _holder_id(lock, "HK") == "hk-1"


def test_shared_warmup_completion_does_not_release_a_silent_hk_lock(service, lock, redis):
    started_at = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    redis.values[_lock_key_for_market("HK")] = f"refresh_hk:hk-1:{started_at}"
    service.save_warmup_metadata("completed", 10, 10)  # a shared run, after the HK lock

    status = _health(service, lock)

    assert status["status"] == "stuck"
    assert _holder_id(lock, "HK") == "hk-1"


def test_finished_refresh_releases_only_its_own_lock(service, lock):
    lock.acquire("refresh_us", "us-1", market="US")
    lock.acquire("refresh_hk", "hk-1", market="HK")
    service.complete_warmup_heartbeat("completed", market="US")
    service.update_warmup_heartbeat(5, 10, market="HK")

    _health(service, lock)

    assert _holder_id(lock, "US") is None
    assert _holder_id(lock, "HK") == "hk-1"
