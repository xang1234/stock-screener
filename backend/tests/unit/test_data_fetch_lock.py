"""
Tests for data_fetch_lock.py — lock mechanism, decorator, and _impl functions.

All tests mock Redis (no live Redis needed) and mock CacheManager/PriceCacheService
(no external API calls).
"""
import logging
import pytest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from celery.exceptions import Retry, SoftTimeLimitExceeded
from sqlalchemy.exc import OperationalError

from app.tasks.data_fetch_lock import (
    DataFetchLock,
    LOCK_KEY,
    _lock_key_for_market,
    all_market_lock_keys,
    disable_serialized_data_fetch_lock,
    serialized_data_fetch,
)

# After bead 9.1 the no-market default path routes to the shared key, not the
# legacy unsuffixed LOCK_KEY. Legacy LOCK_KEY is retained only for startup
# cleanup of pre-9.1 stale locks.
SHARED_LOCK_KEY = _lock_key_for_market(None)  # "data_fetch_job_lock:shared"


def _postgres_recovery_error() -> OperationalError:
    return OperationalError(
        "select 1",
        {},
        Exception(
            "FATAL:  the database system is not yet accepting connections\n"
            "DETAIL:  Consistent recovery state has not been yet reached."
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lock(lock_value=None, ttl=3600):
    """Create a DataFetchLock with a mocked Redis client."""
    with patch("app.tasks.data_fetch_lock.settings") as mock_settings:
        mock_settings.redis_host = "localhost"
        mock_settings.redis_port = 6379
        mock_settings.redis_db = 0
        mock_settings.data_fetch_lock_timeout = 7200

        with patch("app.tasks.data_fetch_lock.redis.Redis") as MockRedis:
            mock_redis = MagicMock()
            MockRedis.return_value = mock_redis

            # register_script returns a callable script object
            mock_release_script = MagicMock()
            mock_extend_script = MagicMock()
            mock_redis.register_script.side_effect = [mock_release_script, mock_extend_script]

            lock = DataFetchLock()

            # Pre-set lock value if requested
            if lock_value is not None:
                mock_redis.get.return_value = lock_value.encode()
            else:
                mock_redis.get.return_value = None

            mock_redis.ttl.return_value = ttl

    return lock, mock_redis, mock_release_script, mock_extend_script


def _make_coordination():
    mock_coordination = MagicMock()
    mock_coordination.acquire_market_workload.return_value = (True, False)
    mock_coordination.acquire_external_fetch.return_value = (True, False)
    return mock_coordination


# ---------------------------------------------------------------------------
# Lock Mechanism Tests
# ---------------------------------------------------------------------------

class TestAcquire:
    """Tests for DataFetchLock.acquire()."""

    def test_acquire_fresh(self):
        """Acquiring a fresh (no holder) lock returns (True, False)."""
        lock, mock_redis, _, _ = _make_lock(lock_value=None)
        mock_redis.set.return_value = True

        result = lock.acquire("test_task", "task-123")

        assert result == (True, False)
        mock_redis.set.assert_called_once()
        args, kwargs = mock_redis.set.call_args
        assert args[0] == SHARED_LOCK_KEY
        assert "test_task:task-123:" in args[1]
        assert kwargs == {"nx": True, "ex": 7200}

    def test_acquire_fresh_verifies_nx_set(self):
        """Acquiring a fresh lock calls redis.set with nx=True."""
        lock, mock_redis, _, _ = _make_lock(lock_value=None)
        mock_redis.set.return_value = True

        lock.acquire("my_task", "id-abc")

        args, kwargs = mock_redis.set.call_args
        assert kwargs["nx"] is True
        assert args[0] == SHARED_LOCK_KEY
        assert "my_task:id-abc:" in args[1]

    def test_acquire_blocked(self):
        """Trying to acquire when a different task holds returns (False, False)."""
        lock, mock_redis, _, _ = _make_lock(
            lock_value="other_task:other-id:2024-01-01T00:00:00"
        )
        mock_redis.set.return_value = False  # NX fails

        result = lock.acquire("my_task", "my-id")

        assert result == (False, False)

    def test_acquire_reentrant(self):
        """Re-entrant acquire (same task_id) returns (True, True) without redis.set()."""
        lock, mock_redis, _, _ = _make_lock(
            lock_value="original_task:task-123:2024-01-01T00:00:00"
        )

        result = lock.acquire("another_name", "task-123")

        assert result == (True, True)
        # Should NOT attempt redis.set — the re-entrant path returns early
        mock_redis.set.assert_not_called()

    def test_acquire_reentrant_skips_unknown(self):
        """task_id='unknown' does NOT trigger re-entrant path."""
        lock, mock_redis, _, _ = _make_lock(
            lock_value="some_task:unknown:2024-01-01T00:00:00"
        )
        mock_redis.set.return_value = False

        result = lock.acquire("my_task", "unknown")

        # Should fall through to normal acquire, which fails
        assert result == (False, False)
        mock_redis.set.assert_called_once()


class TestRelease:
    """Tests for DataFetchLock.release()."""

    def test_release_atomic(self):
        """Release uses Lua script with correct key and match pattern."""
        lock, mock_redis, mock_release_script, _ = _make_lock()
        mock_release_script.return_value = 1  # Lua returns 1 on success

        result = lock.release("task-123")

        assert result is True
        mock_release_script.assert_called_once_with(
            keys=[SHARED_LOCK_KEY], args=[":task-123:"]
        )

    def test_release_wrong_owner(self):
        """Release fails when a different task_id holds the lock."""
        lock, mock_redis, mock_release_script, _ = _make_lock()
        mock_release_script.return_value = 0  # Lua returns 0 on mismatch

        result = lock.release("wrong-id")

        assert result is False


class TestExtendLock:
    """Tests for DataFetchLock.extend_lock()."""

    def test_extend_lock_success(self):
        """Extend uses Lua script and returns True on positive TTL."""
        lock, mock_redis, _, mock_extend_script = _make_lock()
        mock_extend_script.return_value = 4800  # new TTL

        result = lock.extend_lock("task-123", additional_seconds=1200)

        assert result is True
        mock_extend_script.assert_called_once_with(
            keys=[SHARED_LOCK_KEY], args=[":task-123:", 1200, 7200]
        )

    def test_extend_lock_custom_max_ttl(self):
        """Extend passes custom max_ttl to Lua script."""
        lock, mock_redis, _, mock_extend_script = _make_lock()
        mock_extend_script.return_value = 3600

        result = lock.extend_lock("task-123", additional_seconds=300, max_ttl=3600)

        assert result is True
        mock_extend_script.assert_called_once_with(
            keys=[SHARED_LOCK_KEY], args=[":task-123:", 300, 3600]
        )

    def test_extend_lock_wrong_owner(self):
        """Extend fails when a different task_id holds the lock."""
        lock, mock_redis, _, mock_extend_script = _make_lock()
        mock_extend_script.return_value = -1

        result = lock.extend_lock("wrong-id", additional_seconds=600)

        assert result is False


# ---------------------------------------------------------------------------
# Decorator Tests
# ---------------------------------------------------------------------------

class TestSerializedDataFetchDecorator:
    """Tests for the @serialized_data_fetch decorator."""

    @patch("app.wiring.bootstrap.get_data_fetch_lock")
    @patch("app.tasks.data_fetch_lock.settings")
    def test_decorator_releases_on_success(self, mock_settings, mock_get_instance):
        """Lock is released after successful completion."""
        mock_lock = MagicMock()
        mock_lock.acquire.return_value = (True, False)
        mock_lock.lock_timeout = 7200
        mock_get_instance.return_value = mock_lock

        @serialized_data_fetch("test_task")
        def my_func():
            return "ok"

        with patch(
            "app.wiring.bootstrap.get_workload_coordination",
            return_value=_make_coordination(),
        ):
            result = my_func()

        assert result == "ok"
        mock_lock.acquire.assert_called_once()
        mock_lock.release.assert_called_once()

    @patch("app.wiring.bootstrap.get_data_fetch_lock")
    @patch("app.tasks.data_fetch_lock.settings")
    def test_decorator_releases_on_error(self, mock_settings, mock_get_instance):
        """Lock is released after an exception."""
        mock_lock = MagicMock()
        mock_lock.acquire.return_value = (True, False)
        mock_lock.lock_timeout = 7200
        mock_get_instance.return_value = mock_lock

        @serialized_data_fetch("test_task")
        def my_func():
            raise ValueError("boom")

        with patch(
            "app.wiring.bootstrap.get_workload_coordination",
            return_value=_make_coordination(),
        ):
            with pytest.raises(ValueError, match="boom"):
                my_func()

        mock_lock.release.assert_called_once()

    @patch("app.wiring.bootstrap.get_data_fetch_lock")
    @patch("app.tasks.data_fetch_lock.settings")
    def test_decorator_releases_on_retry(self, mock_settings, mock_get_instance):
        """Lock is released when the task raises Celery Retry."""
        mock_lock = MagicMock()
        mock_lock.acquire.return_value = (True, False)
        mock_lock.lock_timeout = 7200
        mock_get_instance.return_value = mock_lock

        @serialized_data_fetch("test_task")
        def my_func():
            raise Retry("retry")

        with patch(
            "app.wiring.bootstrap.get_workload_coordination",
            return_value=_make_coordination(),
        ):
            with pytest.raises(Retry):
                my_func()

        mock_lock.release.assert_called_once()

    @patch("app.wiring.bootstrap.get_workload_coordination")
    @patch("app.wiring.bootstrap.get_data_fetch_lock")
    def test_busy_market_workload_retries_with_coordination_retry_budget(
        self,
        mock_get_lock,
        mock_get_coordination,
    ):
        mock_lock = MagicMock()
        mock_lock.acquire.return_value = (True, False)
        mock_get_lock.return_value = mock_lock

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

        @serialized_data_fetch("refresh_stock_universe")
        def my_func(self, market=None):
            return "ok"

        with pytest.raises(Retry):
            my_func(task, market="US")

        assert retry_calls[0]["countdown"] == 30
        assert retry_calls[0]["max_retries"] == 10_000
        assert "waiting_for_market_workload:US" in str(retry_calls[0]["exc"])
        mock_lock.release.assert_called_once_with("task-123", market="US")
        mock_coordination.release_market_workload.assert_not_called()
        mock_coordination.acquire_external_fetch.assert_not_called()

    @patch("app.wiring.bootstrap.get_workload_coordination")
    @patch("app.wiring.bootstrap.get_data_fetch_lock")
    def test_busy_external_fetch_retries_with_coordination_retry_budget_and_no_error_log(
        self,
        mock_get_lock,
        mock_get_coordination,
        caplog,
    ):
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
            raise Retry(message=str(exc))

        task = SimpleNamespace(
            request=SimpleNamespace(id="task-123", retries=0),
            retry=_retry,
        )

        @serialized_data_fetch("smart_refresh_cache")
        def my_func(self, market=None):
            return "ok"

        caplog.set_level(logging.ERROR)
        with pytest.raises(Retry):
            my_func(task, market="US")

        assert retry_calls[0]["countdown"] == 15
        assert retry_calls[0]["max_retries"] == 10_000
        assert "waiting_for_external_fetch_global" in str(retry_calls[0]["exc"])
        mock_coordination.release_market_workload.assert_called_once_with(
            "task-123",
            market="US",
        )
        mock_lock.release.assert_called_once_with("task-123", market="US")
        assert not any(
            "Error in data fetch task smart_refresh_cache" in record.message
            for record in caplog.records
        )

    @patch("app.wiring.bootstrap.get_workload_coordination")
    @patch("app.wiring.bootstrap.get_data_fetch_lock")
    def test_transient_postgres_recovery_retries_and_releases_leases(
        self,
        mock_get_lock,
        mock_get_coordination,
    ):
        mock_lock = MagicMock()
        mock_lock.acquire.return_value = (True, False)
        mock_get_lock.return_value = mock_lock

        mock_coordination = MagicMock()
        mock_coordination.acquire_market_workload.return_value = (True, False)
        mock_coordination.acquire_external_fetch.return_value = (True, False)
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

        @serialized_data_fetch("smart_refresh_cache")
        def my_func(self, market=None):
            raise _postgres_recovery_error()

        with pytest.raises(Retry):
            my_func(task, market="HK")

        assert retry_calls[0]["countdown"] == 5
        assert retry_calls[0]["max_retries"] == 12
        assert "database system is not yet accepting connections" in str(
            retry_calls[0]["exc"]
        )
        mock_coordination.release_external_fetch.assert_called_once_with("task-123")
        mock_coordination.release_market_workload.assert_called_once_with(
            "task-123",
            market="HK",
        )
        mock_lock.release.assert_called_once_with("task-123", market="HK")

    @patch("app.wiring.bootstrap.get_data_fetch_lock")
    @patch("app.tasks.data_fetch_lock.settings")
    def test_decorator_releases_on_soft_time_limit(self, mock_settings, mock_get_instance):
        """Lock is released when the task exceeds its soft time limit."""
        mock_lock = MagicMock()
        mock_lock.acquire.return_value = (True, False)
        mock_lock.lock_timeout = 7200
        mock_get_instance.return_value = mock_lock

        @serialized_data_fetch("test_task")
        def my_func():
            raise SoftTimeLimitExceeded()

        with patch(
            "app.wiring.bootstrap.get_workload_coordination",
            return_value=_make_coordination(),
        ):
            with pytest.raises(SoftTimeLimitExceeded):
                my_func()

        mock_lock.release.assert_called_once()

    @patch("app.wiring.bootstrap.get_data_fetch_lock")
    @patch("app.tasks.data_fetch_lock.settings")
    def test_decorator_skips_release_on_reentrant(self, mock_settings, mock_get_instance):
        """Re-entrant acquire does NOT call release."""
        mock_lock = MagicMock()
        mock_lock.acquire.return_value = (True, True)  # re-entrant
        mock_lock.lock_timeout = 7200
        mock_get_instance.return_value = mock_lock

        @serialized_data_fetch("test_task")
        def my_func():
            return "ok"

        with patch(
            "app.wiring.bootstrap.get_workload_coordination",
            return_value=_make_coordination(),
        ):
            result = my_func()

        assert result == "ok"
        mock_lock.release.assert_not_called()

    @patch("app.wiring.bootstrap.get_data_fetch_lock")
    @patch("app.tasks.data_fetch_lock.settings")
    def test_decorator_skips_when_lock_held(self, mock_settings, mock_get_instance):
        """When acquire fails, the decorated function returns a skip payload."""
        mock_lock = MagicMock()
        mock_lock.acquire.return_value = (False, False)
        mock_lock.get_current_holder.return_value = {
            'task_name': 'stale_task', 'task_id': 'stale-id',
        }
        mock_lock.get_current_task.return_value = {
            'task_name': 'smart_refresh_cache', 'task_id': 'live-task-id',
        }
        mock_get_instance.return_value = mock_lock
        executed = False

        @serialized_data_fetch("test_task")
        def my_func():
            nonlocal executed
            executed = True
            return "executed"

        result = my_func()

        assert executed is False
        assert result["status"] == "already_running"
        assert result["skipped"] is True
        assert result["task_name"] == "test_task"
        assert result["running_task_name"] == "smart_refresh_cache"
        assert result["task_id"] == "live-task-id"
        mock_lock.acquire.assert_called_once()  # no retry loop

    @patch("app.wiring.bootstrap.get_data_fetch_lock")
    @patch("app.tasks.data_fetch_lock.settings")
    def test_decorator_does_not_release_unacquired_lock(self, mock_settings, mock_get_instance):
        """When lock was never acquired, release() is NOT called."""
        mock_lock = MagicMock()
        mock_lock.acquire.return_value = (False, False)
        mock_lock.get_current_holder.return_value = None
        mock_get_instance.return_value = mock_lock

        @serialized_data_fetch("test_task")
        def my_func():
            return "done"

        my_func()

        mock_lock.release.assert_not_called()

    @patch("app.wiring.bootstrap.get_data_fetch_lock")
    def test_decorator_can_bypass_lock_for_in_process_workflows(self, mock_get_instance):
        """Export/bootstrap flows can skip the distributed lock in a single process."""
        @serialized_data_fetch("test_task")
        def my_func():
            return "ok"

        with disable_serialized_data_fetch_lock():
            result = my_func()

        assert result == "ok"
        mock_get_instance.assert_not_called()


# ---------------------------------------------------------------------------
# _impl Function Tests
# ---------------------------------------------------------------------------

class TestPrewarmScanCacheImpl:
    """Tests for _prewarm_scan_cache_impl."""

    def test_impl_with_no_task(self, monkeypatch):
        """_prewarm_scan_cache_impl works without Celery task context."""
        import app.tasks.cache_tasks as module

        monkeypatch.setattr(module, "format_market_status", lambda: "closed")
        _prewarm_scan_cache_impl = module._prewarm_scan_cache_impl

        mock_db = MagicMock()
        mock_session_local = MagicMock(return_value=mock_db)
        mock_cache = MagicMock()
        if hasattr(module, "get_session_factory"):
            mock_get_session_factory = MagicMock(return_value=mock_session_local)
            monkeypatch.setattr(module, "get_session_factory", mock_get_session_factory)
        else:
            mock_get_session_factory = None
            monkeypatch.setattr(module, "SessionLocal", mock_session_local)

        if hasattr(module, "get_cache_manager"):
            mock_get_cache_manager = MagicMock(return_value=mock_cache)
            monkeypatch.setattr(module, "get_cache_manager", mock_get_cache_manager)
        else:
            mock_get_cache_manager = None
            mock_cache_cls = MagicMock(return_value=mock_cache)
            monkeypatch.setattr(module, "CacheManager", mock_cache_cls)

        mock_cache.warm_all_caches.return_value = {
            'warmed': 2, 'failed': 0, 'already_cached': 1
        }

        result = _prewarm_scan_cache_impl(
            task=None,
            symbol_list=["AAPL", "MSFT", "GOOGL"],
            priority="low"
        )

        assert result['warmed'] == 2
        assert result['failed'] == 0
        assert result['already_cached'] == 1
        assert result['total'] == 3
        assert 'completed_at' in result
        # DB should be opened and closed by impl (owns_db=True)
        if mock_get_session_factory is not None:
            mock_get_session_factory.assert_called_once()
        mock_session_local.assert_called_once()
        if mock_get_cache_manager is not None:
            mock_get_cache_manager.assert_called_once_with(db=mock_db)
        mock_db.close.assert_called_once()

    def test_impl_with_shared_db(self, monkeypatch):
        """_prewarm_scan_cache_impl uses provided db and does not close it."""
        import app.tasks.cache_tasks as module

        monkeypatch.setattr(module, "format_market_status", lambda: "closed")
        _prewarm_scan_cache_impl = module._prewarm_scan_cache_impl

        mock_db = MagicMock()
        mock_cache = MagicMock()
        if hasattr(module, "get_cache_manager"):
            mock_get_cache_manager = MagicMock(return_value=mock_cache)
            monkeypatch.setattr(module, "get_cache_manager", mock_get_cache_manager)
        else:
            mock_get_cache_manager = None
            mock_cache_cls = MagicMock(return_value=mock_cache)
            monkeypatch.setattr(module, "CacheManager", mock_cache_cls)

        mock_cache.warm_all_caches.return_value = {
            'warmed': 1, 'failed': 0, 'already_cached': 0
        }

        result = _prewarm_scan_cache_impl(
            task=None,
            symbol_list=["AAPL"],
            priority="normal",
            db=mock_db
        )

        assert result['warmed'] == 1
        if mock_get_cache_manager is not None:
            mock_get_cache_manager.assert_called_once_with(db=mock_db)
        # DB should NOT be closed — caller owns it
        mock_db.close.assert_not_called()


class TestForceRefreshStaleIntradayImpl:
    """Tests for _force_refresh_stale_intraday_impl."""

    @patch("app.tasks.cache_tasks.format_market_status", return_value="closed")
    @patch("app.wiring.bootstrap.get_price_cache")
    @patch("app.services.bulk_data_fetcher.BulkDataFetcher.__init__", return_value=None)
    def test_impl_with_no_task(self, mock_bf_init, mock_pcs_get, mock_market):
        """_force_refresh_stale_intraday_impl works without Celery task context."""
        from app.tasks.cache_tasks import _force_refresh_stale_intraday_impl

        mock_pcs = MagicMock()
        mock_pcs_get.return_value = mock_pcs
        mock_pcs.get_stale_intraday_symbols.return_value = []  # no stale symbols

        result = _force_refresh_stale_intraday_impl(task=None, symbols=None)

        assert result['refreshed'] == 0
        assert result['message'] == 'No stale intraday data detected'

    @patch("app.tasks.cache_tasks.format_market_status", return_value="closed")
    @patch("app.tasks.cache_tasks._fetch_with_backoff")
    @patch("app.wiring.bootstrap.get_price_cache")
    @patch("app.services.bulk_data_fetcher.BulkDataFetcher.__init__", return_value=None)
    def test_impl_with_symbols(self, mock_bf_init, mock_pcs_get, mock_fetch, mock_market):
        """_force_refresh_stale_intraday_impl processes provided symbols."""
        from app.tasks.cache_tasks import _force_refresh_stale_intraday_impl
        import pandas as pd

        mock_pcs = MagicMock()
        mock_pcs_get.return_value = mock_pcs

        mock_price_df = MagicMock(spec=pd.DataFrame)
        mock_price_df.empty = False
        mock_price_df.__len__ = lambda self: 100

        mock_fetch.return_value = {
            "AAPL": {"has_error": False, "price_data": mock_price_df},
            "MSFT": {"has_error": True, "error": "rate limited"},
        }
        with patch("app.tasks.cache_tasks._filter_active_symbols", return_value=["AAPL", "MSFT"]):
            result = _force_refresh_stale_intraday_impl(
                task=None, symbols=["AAPL", "MSFT"]
            )

        assert result['refreshed'] == 1
        assert result['failed'] == 1
        assert result['total'] == 2


# ---------------------------------------------------------------------------
# Worker Startup Signal Tests
# ---------------------------------------------------------------------------

class TestClearStaleLockOnStartup:
    """Tests for the @worker_ready signal handler in celery_app.py."""

    @patch("app.wiring.bootstrap.get_data_fetch_lock")
    def test_clears_stale_lock_for_datafetch_worker(self, mock_get_instance):
        """Datafetch worker clears an existing stale lock on startup."""
        from app.celery_app import _clear_stale_data_fetch_lock

        mock_lock = MagicMock()
        mock_lock.get_current_task.return_value = {
            'task_name': 'calculate_daily_group_rankings',
            'task_id': 'dead-task-id',
            'started_at': '2026-01-01T00:00:00',
            'ttl_seconds': 5400,
            'last_heartbeat': '2026-01-01T00:00:00+00:00',
        }
        mock_get_instance.return_value = mock_lock

        sender = MagicMock()
        sender.hostname = 'datafetch@abc123'

        _clear_stale_data_fetch_lock(sender=sender)

        mock_lock.force_release.assert_called_once()

    @patch("app.wiring.bootstrap.get_data_fetch_lock")
    def test_no_op_when_no_lock_exists(self, mock_get_instance):
        """Datafetch worker does nothing when no lock is held."""
        from app.celery_app import _clear_stale_data_fetch_lock

        mock_lock = MagicMock()
        mock_lock.get_current_task.return_value = None
        mock_lock.get_current_holder.return_value = None
        mock_get_instance.return_value = mock_lock

        sender = MagicMock()
        sender.hostname = 'datafetch@abc123'

        _clear_stale_data_fetch_lock(sender=sender)

        mock_lock.force_release.assert_not_called()

    def test_general_worker_does_not_clear_lock(self):
        """General worker must NOT clear the datafetch lock."""
        from app.celery_app import _clear_stale_data_fetch_lock

        sender = MagicMock()
        sender.hostname = 'general@abc123'

        # If DataFetchLock were instantiated, the test would fail
        # because we don't mock it — the early return prevents that
        with patch("app.wiring.bootstrap.get_data_fetch_lock") as mock_gi:
            _clear_stale_data_fetch_lock(sender=sender)
            mock_gi.assert_not_called()

    @patch("app.celery_app._ensure_worker_runtime_services")
    def test_general_worker_initializes_runtime_services(self, mock_ensure_runtime):
        """General worker still initializes runtime services on worker_ready."""
        from app.celery_app import _clear_stale_data_fetch_lock

        sender = MagicMock()
        sender.hostname = 'general@abc123'

        with patch("app.wiring.bootstrap.get_data_fetch_lock") as mock_gi:
            _clear_stale_data_fetch_lock(sender=sender)
            mock_ensure_runtime.assert_called_once()
            mock_gi.assert_not_called()

    @patch("app.wiring.bootstrap.get_data_fetch_lock")
    def test_handles_redis_connection_error(self, mock_get_instance):
        """Signal handler catches exceptions and doesn't crash the worker."""
        from app.celery_app import _clear_stale_data_fetch_lock

        mock_get_instance.side_effect = ConnectionError("Redis unavailable")

        sender = MagicMock()
        sender.hostname = 'datafetch@abc123'

        # Should not raise — the handler catches all exceptions
        _clear_stale_data_fetch_lock(sender=sender)

    @patch("app.wiring.bootstrap.get_data_fetch_lock")
    def test_global_worker_does_not_drop_live_market_locks(self, mock_get_instance):
        """Global worker must not clear a live per-market lock during startup overlap."""
        from app.celery_app import _clear_stale_data_fetch_lock

        mock_lock = MagicMock()
        mock_lock.get_current_task.side_effect = lambda market=None: {
            'task_name': 'smart_refresh_cache',
            'task_id': 'live-task-id',
            'started_at': '2026-04-19T00:00:00+00:00',
            'ttl_seconds': 7100,
            'last_heartbeat': datetime.now(timezone.utc).isoformat(),
        } if market == 'US' else None
        mock_get_instance.return_value = mock_lock

        sender = MagicMock()
        sender.hostname = 'datafetch-global@abc123'

        _clear_stale_data_fetch_lock(sender=sender)

        mock_lock.force_release_all.assert_not_called()
        mock_lock.force_release.assert_not_called()

    @patch("app.wiring.bootstrap.get_data_fetch_lock")
    def test_global_worker_clears_only_stale_market_locks(self, mock_get_instance):
        """Global worker clears only scopes whose heartbeat is stale."""
        from app.celery_app import _clear_stale_data_fetch_lock

        stale_heartbeat = (datetime.now(timezone.utc) - timedelta(minutes=45)).isoformat()
        mock_lock = MagicMock()
        mock_lock.get_current_task.side_effect = lambda market=None: {
            'task_name': 'smart_refresh_cache',
            'task_id': 'stale-task-id',
            'started_at': '2026-04-19T00:00:00+00:00',
            'ttl_seconds': 5400,
            'last_heartbeat': stale_heartbeat,
        } if market == 'HK' else None
        mock_get_instance.return_value = mock_lock

        sender = MagicMock()
        sender.hostname = 'datafetch-global@abc123'

        _clear_stale_data_fetch_lock(sender=sender)

        mock_lock.force_release.assert_called_once_with(market='HK')
        mock_lock.force_release_all.assert_not_called()


class TestEnsureWorkerRuntimeServices:
    @patch("app.wiring.bootstrap.initialize_process_runtime_services")
    @patch("app.celery_app.os.getpid")
    def test_reuses_existing_process_runtime_when_pid_unchanged(
        self,
        mock_getpid,
        mock_initialize_runtime,
    ):
        """Do not replace process runtime when worker PID has not changed."""
        from app.celery_app import _ensure_worker_runtime_services, celery_app

        runtime = object()
        mock_getpid.return_value = 1234
        mock_initialize_runtime.return_value = runtime

        if hasattr(celery_app, "runtime_services"):
            delattr(celery_app, "runtime_services")
        if hasattr(celery_app, "runtime_services_pid"):
            delattr(celery_app, "runtime_services_pid")

        resolved = _ensure_worker_runtime_services()

        assert resolved is runtime
        assert getattr(celery_app, "runtime_services") is runtime
        assert getattr(celery_app, "runtime_services_pid") == 1234
        mock_initialize_runtime.assert_called_once_with(force=False)

    @patch("app.wiring.bootstrap.initialize_process_runtime_services")
    @patch("app.celery_app.os.getpid")
    def test_rebuilds_process_runtime_after_pid_change(
        self,
        mock_getpid,
        mock_initialize_runtime,
    ):
        """Prefork child process must force runtime rebuild after fork."""
        from app.celery_app import _ensure_worker_runtime_services, celery_app

        runtime = object()
        mock_getpid.return_value = 4321
        mock_initialize_runtime.return_value = runtime
        celery_app.runtime_services_pid = 1111

        resolved = _ensure_worker_runtime_services()

        assert resolved is runtime
        mock_initialize_runtime.assert_called_once_with(force=True)

    @patch("app.wiring.bootstrap.initialize_process_runtime_services")
    @patch("app.celery_app.os.getpid")
    def test_force_rebuild_without_pid_marker(
        self,
        mock_getpid,
        mock_initialize_runtime,
    ):
        """Explicit force rebuild must bypass missing runtime PID marker."""
        from app.celery_app import _ensure_worker_runtime_services, celery_app

        runtime = object()
        mock_getpid.return_value = 7777
        mock_initialize_runtime.return_value = runtime
        if hasattr(celery_app, "runtime_services_pid"):
            delattr(celery_app, "runtime_services_pid")

        resolved = _ensure_worker_runtime_services(force_rebuild=True)

        assert resolved is runtime
        mock_initialize_runtime.assert_called_once_with(force=True)


class TestWorkerProcessInit:
    @patch("app.celery_app._ensure_worker_runtime_services")
    @patch("app.database.engine.dispose")
    def test_worker_process_init_forces_runtime_rebuild(
        self,
        mock_engine_dispose,
        mock_ensure_runtime,
    ):
        """Prefork worker init must force runtime rebuild in child process."""
        from app.celery_app import _dispose_engine_after_fork

        _dispose_engine_after_fork()

        mock_engine_dispose.assert_called_once()
        mock_ensure_runtime.assert_called_once_with(force_rebuild=True)


# ---------------------------------------------------------------------------
# Per-market lock tests (bead StockScreenClaude-asia.9.1)
# ---------------------------------------------------------------------------

class TestPerMarketLockKeys:
    """Each market gets its own Redis key so cross-market refreshes can run in parallel."""

    def test_lock_key_per_market(self):
        assert _lock_key_for_market("US") == "data_fetch_job_lock:us"
        assert _lock_key_for_market("HK") == "data_fetch_job_lock:hk"
        assert _lock_key_for_market("JP") == "data_fetch_job_lock:jp"
        assert _lock_key_for_market("TW") == "data_fetch_job_lock:tw"

    def test_lock_key_for_none_is_shared(self):
        assert _lock_key_for_market(None) == "data_fetch_job_lock:shared"

    def test_lock_key_normalizes_case(self):
        assert _lock_key_for_market("hk") == "data_fetch_job_lock:hk"
        assert _lock_key_for_market(" JP ") == "data_fetch_job_lock:jp"

    def test_all_market_lock_keys_covers_markets_and_shared(self):
        keys = all_market_lock_keys()
        for m in ("us", "hk", "jp", "tw", "cn"):
            assert f"data_fetch_job_lock:{m}" in keys
        assert "data_fetch_job_lock:shared" in keys


class TestAcquirePerMarket:
    """Per-market acquire uses the market-specific Redis key."""

    def test_acquire_us_uses_us_key(self):
        lock, mock_redis, _, _ = _make_lock(lock_value=None)
        mock_redis.set.return_value = True

        lock.acquire("smart_refresh_cache", "task-us", market="US")

        args, kwargs = mock_redis.set.call_args
        assert args[0] == "data_fetch_job_lock:us"
        assert "smart_refresh_cache:task-us:" in args[1]

    def test_acquire_hk_uses_hk_key(self):
        lock, mock_redis, _, _ = _make_lock(lock_value=None)
        mock_redis.set.return_value = True

        lock.acquire("smart_refresh_cache", "task-hk", market="HK")

        args, _ = mock_redis.set.call_args
        assert args[0] == "data_fetch_job_lock:hk"

    def test_acquire_unknown_market_rejected(self):
        lock, _, _, _ = _make_lock(lock_value=None)
        with pytest.raises(ValueError):
            lock.acquire("smart_refresh_cache", "task-id", market="ZZ")


class TestReleasePerMarket:
    def test_release_targets_market_key(self):
        lock, _, mock_release_script, _ = _make_lock()
        mock_release_script.return_value = 1

        lock.release("task-hk", market="HK")

        mock_release_script.assert_called_once_with(
            keys=["data_fetch_job_lock:hk"], args=[":task-hk:"]
        )


class TestExtendPerMarket:
    def test_extend_targets_market_key(self):
        lock, _, _, mock_extend_script = _make_lock()
        mock_extend_script.return_value = 3600

        lock.extend_lock("task-jp", additional_seconds=300, market="JP")

        mock_extend_script.assert_called_once_with(
            keys=["data_fetch_job_lock:jp"], args=[":task-jp:", 300, 7200]
        )


class TestDecoratorPassesMarket:
    """Decorator pulls `market` from task kwargs and threads it to the lock."""

    @patch("app.wiring.bootstrap.get_data_fetch_lock")
    @patch("app.tasks.data_fetch_lock.settings")
    def test_decorator_passes_market_to_acquire(self, mock_settings, mock_get_lock):
        mock_lock = MagicMock()
        mock_lock.acquire.return_value = (True, False)
        mock_lock.lock_timeout = 7200
        mock_get_lock.return_value = mock_lock

        @serialized_data_fetch("test_task")
        def my_func(market=None):
            return "ok"

        with patch(
            "app.wiring.bootstrap.get_workload_coordination",
            return_value=_make_coordination(),
        ):
            my_func(market="HK")

        args, kwargs = mock_lock.acquire.call_args
        assert kwargs.get("market") == "HK"
        mock_lock.release.assert_called_once()
        release_kwargs = mock_lock.release.call_args.kwargs
        assert release_kwargs.get("market") == "HK"

    @patch("app.wiring.bootstrap.get_data_fetch_lock")
    @patch("app.tasks.data_fetch_lock.settings")
    def test_decorator_defaults_to_shared_when_no_market(self, mock_settings, mock_get_lock):
        mock_lock = MagicMock()
        mock_lock.acquire.return_value = (True, False)
        mock_lock.lock_timeout = 7200
        mock_get_lock.return_value = mock_lock

        @serialized_data_fetch("test_task")
        def my_func():
            return "ok"

        with patch(
            "app.wiring.bootstrap.get_workload_coordination",
            return_value=_make_coordination(),
        ):
            my_func()

        args, kwargs = mock_lock.acquire.call_args
        # Market arg should be None (→ shared) since the task has no market kwarg.
        assert kwargs.get("market") is None

    @patch("app.wiring.bootstrap.get_data_fetch_lock")
    @patch("app.tasks.data_fetch_lock.settings")
    def test_lock_contention_payload_includes_market(self, mock_settings, mock_get_lock):
        mock_lock = MagicMock()
        mock_lock.acquire.return_value = (False, False)
        mock_lock.get_current_holder.return_value = {
            'task_name': 'other', 'task_id': 'other-id',
        }
        mock_lock.get_current_task.return_value = {
            'task_name': 'other', 'task_id': 'other-id',
        }
        mock_get_lock.return_value = mock_lock

        @serialized_data_fetch("test_task")
        def my_func(market=None):
            return "ran"

        result = my_func(market="JP")

        assert result["status"] == "already_running"
        assert result["market"] == "jp"
