"""
Tests for the Redis-backed distributed rate limiter.

Tests cover:
1. Redis available — basic flow (first call immediate, second call waits)
2. Redis available — timeout handling
3. Redis unavailable — fallback to in-process
4. Fallback recovery when Redis becomes available
5. Concurrent access (thread safety)
6. Jitter behavior
7. Different keys don't interfere
"""
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from app.services.rate_limiter import RedisRateLimiter, RateLimitTimeoutError


class TestRedisRateLimiterWithRedis:
    """Tests for when Redis is available."""

    def _make_limiter_with_mock_redis(self):
        """Create a limiter with a mock Redis client that simulates the Lua script."""
        limiter = RedisRateLimiter(jitter_ms=0)  # No jitter for deterministic tests

        mock_client = MagicMock()
        mock_client.ping.return_value = True

        # Track the "next_allowed" per key to simulate Lua script
        state = {}

        def mock_eval(script, num_keys, key, interval_str):
            interval = float(interval_str)
            now = time.time()
            next_allowed = state.get(key, 0.0)

            if now >= next_allowed:
                state[key] = now + interval
                return b'0'
            else:
                wait_time = next_allowed - now
                state[key] = next_allowed + interval
                return str(wait_time).encode()

        mock_client.eval = mock_eval

        # Patch _get_redis_client to return our mock
        limiter._get_redis_client = lambda: mock_client
        limiter._using_fallback = False

        return limiter

    def test_first_call_immediate(self):
        """First call should return immediately (wait ≈ 0)."""
        limiter = self._make_limiter_with_mock_redis()

        waited = limiter.wait("yfinance", min_interval_s=1.0)
        assert waited == 0.0

    def test_second_call_waits(self):
        """Second call should wait approximately min_interval_s."""
        limiter = self._make_limiter_with_mock_redis()

        limiter.wait("yfinance", min_interval_s=0.1)
        start = time.monotonic()
        waited = limiter.wait("yfinance", min_interval_s=0.1)
        elapsed = time.monotonic() - start

        # Should wait roughly 0.1s (with some tolerance)
        assert waited > 0.05
        assert elapsed >= 0.05

    def test_different_keys_independent(self):
        """Different rate limit keys should not interfere."""
        limiter = self._make_limiter_with_mock_redis()

        # First call to each key should be immediate
        waited1 = limiter.wait("yfinance", min_interval_s=1.0)
        waited2 = limiter.wait("finviz", min_interval_s=1.0)

        assert waited1 == 0.0
        assert waited2 == 0.0

    def test_timeout_raises_error(self):
        """Should raise RateLimitTimeoutError when wait exceeds timeout."""
        limiter = self._make_limiter_with_mock_redis()

        # First call establishes the baseline
        limiter.wait("yfinance", min_interval_s=10.0)

        # Second call would need to wait ~10s, but timeout is 0.1s
        with pytest.raises(RateLimitTimeoutError):
            limiter.wait("yfinance", min_interval_s=10.0, timeout_s=0.1)


class TestRedisRateLimiterFallback:
    """Tests for when Redis is unavailable."""

    def test_fallback_works(self):
        """When Redis is unavailable, in-process fallback should work."""
        limiter = RedisRateLimiter(jitter_ms=0)
        limiter._get_redis_client = lambda: None  # Redis unavailable

        # First call should be immediate
        waited = limiter.wait("yfinance", min_interval_s=0.1)
        assert waited == 0.0

        # Second call should wait
        start = time.monotonic()
        waited = limiter.wait("yfinance", min_interval_s=0.1)
        elapsed = time.monotonic() - start

        assert waited > 0.05
        assert elapsed >= 0.05

    def test_fallback_warning_logged(self):
        """Should log warning on first fallback."""
        limiter = RedisRateLimiter(jitter_ms=0)
        limiter._get_redis_client = lambda: None

        with patch("app.services.rate_limiter.logger") as mock_logger:
            limiter.wait("yfinance", min_interval_s=0.1)
            mock_logger.warning.assert_called()

    def test_fallback_recovery(self):
        """Should recover to Redis when connection restores."""
        limiter = RedisRateLimiter(jitter_ms=0, redis_retry_interval=0.0)

        # Start with Redis unavailable
        limiter._get_redis_client = lambda: None
        limiter.wait("yfinance", min_interval_s=0.1)
        assert limiter._using_fallback is True

        # Now make Redis available
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.eval.return_value = b'0'
        limiter._get_redis_client = lambda: mock_client

        # Next call should recover
        limiter.wait("yfinance", min_interval_s=0.1)
        assert limiter._using_fallback is False


class TestRedisRateLimiterConcurrency:
    """Tests for thread safety."""

    def test_concurrent_access(self):
        """Two threads calling wait() concurrently should be properly spaced."""
        limiter = RedisRateLimiter(jitter_ms=0)
        limiter._get_redis_client = lambda: None  # Use fallback for simplicity

        interval = 0.15
        timestamps = []
        barrier = threading.Barrier(2)

        def worker():
            barrier.wait()  # Synchronize start
            limiter.wait("yfinance", min_interval_s=interval)
            timestamps.append(time.monotonic())

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(timestamps) == 2

        # Timestamps should be spaced by approximately the interval
        timestamps.sort()
        gap = timestamps[1] - timestamps[0]
        assert gap >= interval * 0.7, f"Gap {gap:.3f}s < expected {interval * 0.7:.3f}s"


class TestRedisRateLimiterJitter:
    """Tests for jitter behavior."""

    def test_jitter_applied_when_waiting(self):
        """Jitter should be applied only when wait > 0."""
        limiter = RedisRateLimiter(jitter_ms=100)
        limiter._get_redis_client = lambda: None  # Use fallback

        # First call — no wait, no jitter
        waited_first = limiter.wait("test_jitter", min_interval_s=0.2)
        assert waited_first == 0.0  # No jitter when not waiting

        # Second call — has wait, should include jitter
        waited_second = limiter.wait("test_jitter", min_interval_s=0.2)
        # With jitter_ms=100, wait should be between 0.1 and 0.35 (interval + up to 0.1 jitter)
        assert waited_second > 0.1

    def test_no_jitter_when_zero(self):
        """No jitter should be added when jitter_ms=0."""
        limiter = RedisRateLimiter(jitter_ms=0)
        limiter._get_redis_client = lambda: None

        # Collect multiple wait times
        limiter.wait("test_nojitter", min_interval_s=0.05)

        # Multiple rapid calls - the wait times should be very consistent
        waits = []
        for _ in range(3):
            w = limiter.wait("test_nojitter", min_interval_s=0.05)
            waits.append(w)

        # Without jitter, all waits should be very close to 0.05
        for w in waits:
            assert 0.03 <= w <= 0.08, f"Wait {w:.3f}s outside expected range"


class TestRateLimitTimeoutError:
    """Tests for the custom exception."""

    def test_error_message(self):
        """Error should contain useful debugging info."""
        err = RateLimitTimeoutError("Rate limit wait for 'yfinance' would be 30.0s")
        assert "yfinance" in str(err)
        assert "30.0s" in str(err)
