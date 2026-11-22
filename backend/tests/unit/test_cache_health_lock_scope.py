"""GET /v1/cache/health judges a held lock by its own market scope and never releases it.

Real DataFetchLock and warmup store over a dict-backed Redis.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from app.services.price_cache_service import PriceCacheService
from app.tasks.data_fetch_lock import LOCK_KEY, _RELEASE_LUA, DataFetchLock, _lock_key_for_market


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
    redis.values[_lock_key_for_market("HK")] = f"refresh_hk:hk-1:{_ten_minutes_ago()}"
    service.save_warmup_metadata("completed", 10, 10)  # a shared run, after the HK lock

    status = _health(service, lock)

    assert status["status"] == "stuck"
    assert _holder_id(lock, "HK") == "hk-1"


def _ten_minutes_ago():
    return (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()


def _new_hk_lock_after_an_earlier_hk_run(service, lock, redis):
    service.save_warmup_metadata("completed", 10, 10, market="HK")
    service.complete_warmup_heartbeat("completed", market="HK")
    lock.acquire("refresh_hk", "hk-2", market="HK")  # not yet publishing progress
    return {("hk-2", "HK")}


def _finished_us_refresh_beside_a_running_hk_refresh(service, lock, redis):
    lock.acquire("refresh_us", "us-1", market="US")
    lock.acquire("refresh_hk", "hk-1", market="HK")
    service.save_warmup_metadata("completed", 10, 10, market="US")
    service.complete_warmup_heartbeat("completed", market="US")
    service.update_warmup_heartbeat(5, 10, market="HK")
    return {("us-1", "US"), ("hk-1", "HK")}


def _silent_us_lock_whose_warmup_finished(service, lock, redis):
    redis.values[_lock_key_for_market("US")] = f"refresh_us:us-1:{_ten_minutes_ago()}"
    service.save_warmup_metadata("completed", 10, 10, market="US")
    return {("us-1", "US")}


def _finished_legacy_unsuffixed_lock(service, lock, redis):
    redis.values[LOCK_KEY] = f"refresh:legacy-1:{_ten_minutes_ago()}"
    service.save_warmup_metadata("completed", 10, 10)
    service.complete_warmup_heartbeat("completed")
    return {("legacy-1", None)}


def _lock_keys(redis):
    return {key: value for key, value in redis.values.items() if key.startswith(LOCK_KEY)}


@pytest.mark.parametrize(
    "arrange",
    [
        _new_hk_lock_after_an_earlier_hk_run,
        _finished_us_refresh_beside_a_running_hk_refresh,
        _silent_us_lock_whose_warmup_finished,
        _finished_legacy_unsuffixed_lock,
    ],
)
def test_health_reports_a_held_lock_without_touching_it(service, lock, redis, arrange):
    holders = arrange(service, lock, redis)
    redis.values[service._redis_last_update_key("SPY", market="US")] = "2099-01-02"  # SPY fresh
    keys_before = _lock_keys(redis)

    status = _health(service, lock)

    assert _lock_keys(redis) == keys_before
    assert status["status"] == "updating"
    assert status["can_refresh"] is False
    running = status["task_running"]
    assert (running["task_id"], running["market"]) in holders


@pytest.mark.parametrize("publishes_progress", [True, False])
def test_a_held_lock_reports_its_own_markets_last_warmup(service, lock, publishes_progress):
    service.save_warmup_metadata("completed", 10, 10)  # a shared run
    service.save_warmup_metadata("completed", 3, 3, market="HK")  # the previous HK run
    lock.acquire("refresh_hk", "hk-1", market="HK")
    if publishes_progress:
        service.update_warmup_heartbeat(5, 10, market="HK")

    status = _health(service, lock)

    assert status["status"] == "updating"
    assert status["last_warmup"]["market"] == "hk"


def test_an_earlier_runs_running_heartbeat_does_not_mark_a_new_lock_stuck(service, lock, monkeypatch):
    # The previous HK holder died mid-run 45 min ago, leaving a "running" heartbeat.
    monkeypatch.setattr(
        service, "_get_heartbeat_info", lambda market=None: {"status": "running", "minutes": 45}
    )
    lock.acquire("refresh_hk", "hk-2", market="HK")  # just acquired, no heartbeat of its own yet

    status = _health(service, lock)

    assert status["status"] == "updating"
    assert "can_force_cancel" not in status
    assert status["task_running"]["task_id"] == "hk-2"


def test_a_lock_held_long_after_its_warmup_finished_is_stuck(service, lock, redis, monkeypatch):
    # The warmup finished an hour ago, but post-warmup work hung while the
    # renewal thread kept the lease alive.
    def hours_ago(hours):
        return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

    redis.values[_lock_key_for_market("US")] = f"refresh_us:us-1:{hours_ago(2)}"
    # Weekly refresh writes this heartbeat before its post-warmup work, so it is
    # as old as the warmup; force-cancel accepts a terminal heartbeat that old.
    monkeypatch.setattr(
        service, "_get_heartbeat_info", lambda market=None: {"status": "completed", "minutes": 60}
    )
    monkeypatch.setattr(
        service, "get_warmup_metadata",
        lambda market=None: {"status": "completed", "completed_at": hours_ago(1)},
    )

    status = _health(service, lock)

    assert status["status"] == "stuck"
    assert status["can_force_cancel"] is True


def test_force_cancel_releases_only_the_stuck_markets_lock(service, lock):
    """POST /v1/cache/force-cancel on a silent US lock must leave HK's lock held."""
    from app.api.v1 import cache as cache_api

    lock.acquire("refresh_us", "us-1", market="US")
    lock.acquire("refresh_hk", "hk-1", market="HK")
    service.update_warmup_heartbeat(5, 10, market="HK")

    with patch.object(cache_api, "get_data_fetch_lock", return_value=lock), patch.object(
        cache_api, "get_price_cache", return_value=service
    ):
        result = cache_api.force_cancel_refresh()

    assert result["status"] == "cancelled"
    assert result["task_id"] == "us-1"
    assert _holder_id(lock, "US") is None
    assert _holder_id(lock, "HK") == "hk-1"


def test_force_cancel_releases_a_stuck_legacy_unsuffixed_lock(service, lock, redis):
    from app.api.v1 import cache as cache_api

    redis.values[LOCK_KEY] = f"refresh:legacy-1:{_ten_minutes_ago()}"

    with patch.object(cache_api, "get_data_fetch_lock", return_value=lock), patch.object(
        cache_api, "get_price_cache", return_value=service
    ):
        result = cache_api.force_cancel_refresh()

    assert result["status"] == "cancelled"
    assert LOCK_KEY not in redis.values
