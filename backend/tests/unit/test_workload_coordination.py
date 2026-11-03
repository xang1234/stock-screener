"""Unit tests for background-task lease coordination."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from celery.exceptions import Retry
from sqlalchemy.exc import OperationalError


def _postgres_recovery_error() -> OperationalError:
    return OperationalError(
        "select 1",
        {},
        Exception(
            "FATAL:  the database system is not yet accepting connections\n"
            "DETAIL:  Consistent recovery state has not been yet reached."
        ),
    )


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
def test_serialized_market_workload_retries_with_coordination_retry_budget(
    mock_get_coordination,
):
    from app.tasks.workload_coordination import serialized_market_workload

    mock_coordination = MagicMock()
    mock_coordination.acquire_market_workload.return_value = (False, False)
    mock_coordination.get_market_workload_holder.return_value = {
        "task_name": "calculate_daily_group_rankings_with_gapfill",
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
        raise Retry(message=str(exc))

    task = SimpleNamespace(
        request=SimpleNamespace(id="task-123", retries=1),
        retry=_retry,
    )
    body_called = False

    @serialized_market_workload("build_daily_snapshot")
    def my_func(self, market=None):
        nonlocal body_called
        body_called = True
        return {"status": "ok"}

    with pytest.raises(Retry):
        my_func(task, market="US")

    assert body_called is False
    assert retry_calls[0]["countdown"] == 30
    assert retry_calls[0]["max_retries"] == 10_000
    assert "waiting_for_market_workload:US" in str(retry_calls[0]["exc"])


@patch("app.wiring.bootstrap.get_workload_coordination")
def test_serialized_market_workload_retries_transient_postgres_recovery(
    mock_get_coordination,
):
    from app.tasks.workload_coordination import serialized_market_workload

    mock_coordination = MagicMock()
    mock_coordination.acquire_market_workload.return_value = (True, False)
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
        raise Retry(message=str(exc))

    task = SimpleNamespace(
        request=SimpleNamespace(id="task-123", retries=0),
        retry=_retry,
    )

    @serialized_market_workload("build_daily_snapshot")
    def my_func(self, market=None):
        raise _postgres_recovery_error()

    with pytest.raises(Retry):
        my_func(task, market="TW")

    assert retry_calls[0]["countdown"] == 5
    assert retry_calls[0]["max_retries"] == 12
    assert "database system is not yet accepting connections" in str(
        retry_calls[0]["exc"]
    )
    mock_coordination.release_market_workload.assert_called_once_with(
        "task-123",
        market="TW",
    )


@patch("app.wiring.bootstrap.get_workload_coordination")
@patch("app.wiring.bootstrap.get_data_fetch_lock")
def test_serialized_data_fetch_retries_when_external_fetch_lease_is_busy(
    mock_get_lock,
    mock_get_coordination,
):
    from app.tasks.data_fetch_lock import _serialized_data_fetch

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

    @_serialized_data_fetch("smart_refresh_cache")
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


def test_pipeline_chain_claim_is_an_atomic_market_scoped_lease():
    from app.tasks.workload_coordination import PIPELINE_CHAIN_TTL_SECONDS

    coordination, _, mock_script = _make_coordination()
    mock_script.return_value = 1

    assert coordination.claim_pipeline_chain(
        "token-1", market="US", task_ids=["stage-1", "stage-2"]
    ) is True

    # One script: SET NX on the claim (Beat and a manual Run cannot both win)
    # and, only if that won, the chain's stage ids under the token.
    kwargs = mock_script.call_args.kwargs
    assert kwargs["keys"] == ["pipeline_chain:us", "pipeline_chain_tasks:token-1"]
    value, task_ids, ttl = kwargs["args"]
    assert value.startswith("daily_market_pipeline:token-1:")
    assert json.loads(task_ids) == ["stage-1", "stage-2"]
    assert ttl == PIPELINE_CHAIN_TTL_SECONDS


def test_pipeline_chain_claim_reports_an_existing_holder():
    coordination, _, mock_script = _make_coordination()
    mock_script.return_value = 0

    assert coordination.claim_pipeline_chain("token-2", market="US", task_ids=["s"]) is False


def test_pipeline_chain_claim_is_a_heartbeat_safety_net_not_a_daily_deadline():
    # The heartbeat renews a live chain and frees a dead one (#605); the TTL
    # only bounds a claim the heartbeat stopped looking after.
    from app.tasks.workload_coordination import (
        PIPELINE_CHAIN_HEARTBEAT_SECONDS,
        PIPELINE_CHAIN_TTL_SECONDS,
    )

    assert 48 * 3600 <= PIPELINE_CHAIN_TTL_SECONDS <= 72 * 3600
    assert PIPELINE_CHAIN_HEARTBEAT_SECONDS * 10 < PIPELINE_CHAIN_TTL_SECONDS


def test_pipeline_chain_renewal_is_owner_checked_and_keeps_the_stage_ids():
    from app.tasks.workload_coordination import PIPELINE_CHAIN_TTL_SECONDS

    coordination, mock_redis, mock_script = _make_coordination()
    mock_script.return_value = 1

    assert coordination.renew_pipeline_chain("token-1", market="US") is True
    mock_script.assert_called_once_with(
        keys=["pipeline_chain:us"], args=[":token-1:", PIPELINE_CHAIN_TTL_SECONDS]
    )
    mock_redis.expire.assert_called_once_with(
        "pipeline_chain_tasks:token-1", PIPELINE_CHAIN_TTL_SECONDS
    )


def test_pipeline_chain_renewal_leaves_a_newer_claim_alone():
    coordination, mock_redis, mock_script = _make_coordination()
    mock_script.return_value = 0

    assert coordination.renew_pipeline_chain("token-1", market="US") is False
    mock_redis.expire.assert_not_called()


def test_pipeline_chain_stage_ids():
    coordination, mock_redis, _ = _make_coordination()
    mock_redis.get.return_value = b'["stage-1", "stage-2"]'

    assert coordination.get_pipeline_chain_task_ids("token-1") == ["stage-1", "stage-2"]
    mock_redis.get.assert_called_with("pipeline_chain_tasks:token-1")

    mock_redis.get.return_value = None
    assert coordination.get_pipeline_chain_task_ids("token-0") is None


def test_pipeline_chain_missing_clock_starts_once_and_clears():
    from app.tasks.workload_coordination import PIPELINE_CHAIN_TTL_SECONDS

    coordination, mock_redis, _ = _make_coordination()
    mock_redis.get.return_value = b"1000.0"

    with patch("app.tasks.workload_coordination.time.time", return_value=1240.0):
        assert coordination.pipeline_chain_missing("token-1") == (240.0, "1000.0")

    # SET NX EX in one command: the first miss starts the clock, later ones
    # keep it, and the key can never be left without a TTL.
    mock_redis.set.assert_called_once_with(
        "pipeline_chain_misses:token-1", "1240.0", nx=True, ex=PIPELINE_CHAIN_TTL_SECONDS
    )
    coordination.clear_pipeline_chain_missing("token-1")
    mock_redis.delete.assert_called_once_with("pipeline_chain_misses:token-1")


def test_pipeline_chain_release_is_owner_checked():
    coordination, _, mock_release_script = _make_coordination()
    mock_release_script.return_value = 1

    assert coordination.release_pipeline_chain("token-1", market="US") is True
    mock_release_script.assert_called_once_with(keys=["pipeline_chain:us"], args=[":token-1:"])


def test_pipeline_chain_missing_reads_no_marker_as_a_first_miss():
    # Cleared between the SET NX and the GET by a heartbeat that saw it alive.
    coordination, mock_redis, _ = _make_coordination()
    mock_redis.get.return_value = None

    assert coordination.pipeline_chain_missing("token-1") == (0.0, None)


@pytest.mark.parametrize("script_result, outcome", [(1, "released"), (2, "kept"), (0, "lost")])
def test_missing_pipeline_chain_release_checks_the_marker_read(script_result, outcome):
    coordination, _, mock_script = _make_coordination()
    mock_script.return_value = script_result

    assert (
        coordination.release_missing_pipeline_chain(
            "token-1", market="US", missing_since="1000.0", heartbeat="hb-1"
        )
        == outcome
    )
    mock_script.assert_called_once_with(
        keys=["pipeline_chain:us", "pipeline_chain_misses:token-1", "pipeline_chain_heartbeat"],
        args=[":token-1:", "1000.0", ":hb-1:"],
    )


def test_missing_pipeline_chain_release_lua_against_real_redis():
    # #627 acceptance: run the Lua on a real Redis. CI has none, so it skips there.
    import redis as redis_lib

    from app.tasks.workload_coordination import (
        _RELEASE_MISSING_PIPELINE_CHAIN_LUA,
        WorkloadCoordination,
    )

    client = redis_lib.Redis(port=6379, db=15, socket_connect_timeout=1, socket_timeout=1)
    try:
        client.ping()
    except (redis_lib.exceptions.ConnectionError, redis_lib.exceptions.TimeoutError):
        pytest.skip("no local Redis")
    coordination = object.__new__(WorkloadCoordination)
    coordination.redis = client
    coordination._release_missing_chain_script = client.register_script(
        _RELEASE_MISSING_PIPELINE_CHAIN_LUA
    )
    claim, marker = "pipeline_chain:us", "pipeline_chain_misses:tok-627"
    lock = "pipeline_chain_heartbeat"

    def release(since):
        return coordination.release_missing_pipeline_chain(
            "tok-627", market="US", missing_since=since, heartbeat="hb-1"
        )

    try:
        client.set(claim, "daily_market_pipeline:tok-627:2026-10-13T00:00:00+00:00")
        client.set(marker, "1000.0")
        # The heartbeat lock expired, then went to the next heartbeat: kept.
        assert release("1000.0") == "kept"
        client.set(lock, "heartbeat:hb-2:")
        assert release("1000.0") == "kept"
        client.set(lock, "heartbeat:hb-1:")
        # Replaced, then cleared: kept, untouched.
        assert release("999.0") == "kept"
        client.delete(marker)
        assert release("1000.0") == "kept"
        assert client.exists(claim)
        # Another token's claim: lost.
        client.set(marker, "1000.0")
        client.set(claim, "daily_market_pipeline:newer:2026-10-13T00:00:00+00:00")
        assert release("1000.0") == "lost"
        assert client.exists(claim)
        # The marker read, still there: released.
        client.set(claim, "daily_market_pipeline:tok-627:2026-10-13T00:00:00+00:00")
        assert release("1000.0") == "released"
        assert not client.exists(claim)
    finally:
        client.delete(claim, marker, lock)
        client.close()


def test_pipeline_chain_heartbeat_lock_is_set_nx_and_owner_checked():
    from app.tasks.workload_coordination import PIPELINE_CHAIN_HEARTBEAT_LOCK_SECONDS

    coordination, mock_redis, mock_release_script = _make_coordination()
    mock_redis.set.return_value = None

    assert coordination.acquire_pipeline_chain_heartbeat("hb-1") is False
    mock_redis.set.assert_called_once_with(
        "pipeline_chain_heartbeat", "heartbeat:hb-1:", nx=True,
        ex=PIPELINE_CHAIN_HEARTBEAT_LOCK_SECONDS,
    )
    coordination.release_pipeline_chain_heartbeat("hb-1")
    mock_release_script.assert_called_once_with(
        keys=["pipeline_chain_heartbeat"], args=[":hb-1:"]
    )


def test_pipeline_chain_holder_reads_token_and_start():
    coordination, mock_redis, _ = _make_coordination()
    mock_redis.get.return_value = b"daily_market_pipeline:token-1:2026-03-16T20:30:00+00:00"
    mock_redis.ttl.return_value = 1000

    holder = coordination.get_pipeline_chain_holder("US")

    assert holder["task_id"] == "token-1"
    assert holder["started_at"] == "2026-03-16T20:30:00+00:00"
    mock_redis.get.assert_called_with("pipeline_chain:us")


def test_pipeline_chain_holders_cover_every_market():
    from app.tasks.market_queues import SUPPORTED_MARKETS

    coordination, mock_redis, _ = _make_coordination()
    mock_redis.get.return_value = None

    holders = coordination.get_pipeline_chain_holders()

    assert set(holders) == {m.upper() for m in SUPPORTED_MARKETS}
    assert all(holder is None for holder in holders.values())


def test_pipeline_chain_starts_read_every_market_in_one_mget():
    coordination, mock_redis, _ = _make_coordination()
    mock_redis.mget.return_value = [
        b"daily_market_pipeline:tok-1:2026-10-16T20:30:00+00:00",
        None,
        b"garbled",
    ]

    starts = coordination.get_pipeline_chain_starts(["US", "hk", "JP"])

    mock_redis.mget.assert_called_once_with(["pipeline_chain:us", "pipeline_chain:hk", "pipeline_chain:jp"])
    mock_redis.get.assert_not_called()
    # Held but unparseable still reads as held ("").
    assert starts == {"US": "2026-10-16T20:30:00+00:00", "HK": None, "JP": ""}


def test_pipeline_catch_up_count_is_per_market_session_and_outlives_a_weekend():
    coordination, mock_redis, _ = _make_coordination()
    pipe = mock_redis.pipeline.return_value
    pipe.execute.return_value = [3, True]

    count = coordination.count_pipeline_catch_up("US", "2026-10-28")

    assert count == 3
    pipe.incr.assert_called_once_with("pipeline_catch_up:us:2026-10-28")
    pipe.expire.assert_called_once_with("pipeline_catch_up:us:2026-10-28", 14 * 24 * 3600)
