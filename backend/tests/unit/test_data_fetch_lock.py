"""
Tests for data_fetch_lock.py — lock mechanism, decorator, and _impl functions.

All tests mock Redis (no live Redis needed) and mock CacheManager/PriceCacheService
(no external API calls).
"""
import pytest
from unittest.mock import MagicMock, patch

from app.tasks.data_fetch_lock import DataFetchLock, serialized_data_fetch, LOCK_KEY


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
        assert args[0] == LOCK_KEY
        assert "test_task:task-123:" in args[1]
        assert kwargs == {"nx": True, "ex": 7200}

    def test_acquire_fresh_verifies_nx_set(self):
        """Acquiring a fresh lock calls redis.set with nx=True."""
        lock, mock_redis, _, _ = _make_lock(lock_value=None)
        mock_redis.set.return_value = True

        lock.acquire("my_task", "id-abc")

        args, kwargs = mock_redis.set.call_args
        assert kwargs["nx"] is True
        assert args[0] == LOCK_KEY
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
            keys=[LOCK_KEY], args=[":task-123:"]
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
            keys=[LOCK_KEY], args=[":task-123:", 1200]
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

    @patch("app.tasks.data_fetch_lock.DataFetchLock.get_instance")
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

        result = my_func()

        assert result == "ok"
        mock_lock.acquire.assert_called_once()
        mock_lock.release.assert_called_once()

    @patch("app.tasks.data_fetch_lock.DataFetchLock.get_instance")
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

        with pytest.raises(ValueError, match="boom"):
            my_func()

        mock_lock.release.assert_called_once()

    @patch("app.tasks.data_fetch_lock.DataFetchLock.get_instance")
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

        result = my_func()

        assert result == "ok"
        mock_lock.release.assert_not_called()

    @patch("app.tasks.data_fetch_lock.DataFetchLock.get_instance")
    @patch("app.tasks.data_fetch_lock.settings")
    def test_decorator_proceeds_when_lock_held(self, mock_settings, mock_get_instance):
        """When acquire fails, the decorated function still executes (no polling)."""
        mock_lock = MagicMock()
        mock_lock.acquire.return_value = (False, False)
        mock_lock.get_current_holder.return_value = {
            'task_name': 'stale_task', 'task_id': 'stale-id',
        }
        mock_get_instance.return_value = mock_lock

        @serialized_data_fetch("test_task")
        def my_func():
            return "executed"

        result = my_func()

        assert result == "executed"
        mock_lock.acquire.assert_called_once()  # no retry loop

    @patch("app.tasks.data_fetch_lock.DataFetchLock.get_instance")
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


# ---------------------------------------------------------------------------
# _impl Function Tests
# ---------------------------------------------------------------------------

class TestPrewarmScanCacheImpl:
    """Tests for _prewarm_scan_cache_impl."""

    @patch("app.tasks.cache_tasks.CacheManager")
    @patch("app.tasks.cache_tasks.SessionLocal")
    @patch("app.tasks.cache_tasks.format_market_status", return_value="closed")
    def test_impl_with_no_task(self, mock_market, mock_session_local, mock_cache_cls):
        """_prewarm_scan_cache_impl works without Celery task context."""
        from app.tasks.cache_tasks import _prewarm_scan_cache_impl

        mock_db = MagicMock()
        mock_session_local.return_value = mock_db
        mock_cache = MagicMock()
        mock_cache_cls.return_value = mock_cache
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
        mock_session_local.assert_called_once()
        mock_db.close.assert_called_once()

    @patch("app.tasks.cache_tasks.CacheManager")
    @patch("app.tasks.cache_tasks.format_market_status", return_value="closed")
    def test_impl_with_shared_db(self, mock_market, mock_cache_cls):
        """_prewarm_scan_cache_impl uses provided db and does not close it."""
        from app.tasks.cache_tasks import _prewarm_scan_cache_impl

        mock_db = MagicMock()
        mock_cache = MagicMock()
        mock_cache_cls.return_value = mock_cache
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
        # DB should NOT be closed — caller owns it
        mock_db.close.assert_not_called()


class TestForceRefreshStaleIntradayImpl:
    """Tests for _force_refresh_stale_intraday_impl."""

    @patch("app.tasks.cache_tasks.format_market_status", return_value="closed")
    @patch("app.services.price_cache_service.PriceCacheService.get_instance")
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
    @patch("app.services.price_cache_service.PriceCacheService.get_instance")
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

    @patch("app.tasks.data_fetch_lock.DataFetchLock.get_instance")
    def test_clears_stale_lock_for_datafetch_worker(self, mock_get_instance):
        """Datafetch worker clears an existing stale lock on startup."""
        from app.celery_app import _clear_stale_data_fetch_lock

        mock_lock = MagicMock()
        mock_lock.get_current_holder.return_value = {
            'task_name': 'calculate_daily_group_rankings',
            'task_id': 'dead-task-id',
            'started_at': '2026-01-01T00:00:00',
            'ttl_seconds': 5400,
        }
        mock_get_instance.return_value = mock_lock

        sender = MagicMock()
        sender.hostname = 'datafetch@abc123'

        _clear_stale_data_fetch_lock(sender=sender)

        mock_lock.force_release.assert_called_once()

    @patch("app.tasks.data_fetch_lock.DataFetchLock.get_instance")
    def test_no_op_when_no_lock_exists(self, mock_get_instance):
        """Datafetch worker does nothing when no lock is held."""
        from app.celery_app import _clear_stale_data_fetch_lock

        mock_lock = MagicMock()
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
        with patch("app.tasks.data_fetch_lock.DataFetchLock.get_instance") as mock_gi:
            _clear_stale_data_fetch_lock(sender=sender)
            mock_gi.assert_not_called()

    @patch("app.tasks.data_fetch_lock.DataFetchLock.get_instance")
    def test_handles_redis_connection_error(self, mock_get_instance):
        """Signal handler catches exceptions and doesn't crash the worker."""
        from app.celery_app import _clear_stale_data_fetch_lock

        mock_get_instance.side_effect = ConnectionError("Redis unavailable")

        sender = MagicMock()
        sender.hostname = 'datafetch@abc123'

        # Should not raise — the handler catches all exceptions
        _clear_stale_data_fetch_lock(sender=sender)
