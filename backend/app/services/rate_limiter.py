"""
Distributed rate limiter using Redis atomic operations.

Provides a single, unified rate limiting mechanism for all external API calls
(yfinance, finviz, etc.) across all Celery workers and processes.

Falls back to in-process limiter when Redis is unavailable, with automatic
recovery when the connection restores.
"""
import logging
import random
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class RateLimitTimeoutError(Exception):
    """Raised when rate limit wait would exceed the specified timeout."""
    pass


# Lua script for atomic rate limiting using Redis server time.
# KEYS[1] = rate limit key (e.g., "ratelimit:yfinance")
# ARGV[1] = min_interval_s (float, seconds between calls)
# Returns: seconds to wait as a string (0 = proceed immediately)
_LUA_RATE_LIMIT = """
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000

local next_allowed = tonumber(redis.call('GET', KEYS[1]) or 0)
local interval = tonumber(ARGV[1])

if now >= next_allowed then
    local new_next = now + interval
    local ttl = math.ceil(interval * 2 + 10)
    redis.call('SET', KEYS[1], tostring(new_next), 'EX', ttl)
    return '0'
else
    local new_next = next_allowed + interval
    local ttl = math.ceil(math.max(new_next - now, interval) + interval * 2)
    redis.call('SET', KEYS[1], tostring(new_next), 'EX', ttl)
    return tostring(next_allowed - now)
end
"""


class RedisRateLimiter:
    """
    Distributed rate limiter using Redis atomic operations.

    Falls back to in-process limiter when Redis is unavailable.
    Automatically recovers to Redis when connection restores.
    """

    def __init__(self, jitter_ms: int = 50, redis_retry_interval: float = 30.0):
        """
        Initialize the rate limiter.

        Args:
            jitter_ms: Maximum jitter in milliseconds to add when waiting.
                       Prevents thundering herd when multiple workers wake.
            redis_retry_interval: Seconds between Redis reconnection attempts
                                  when in fallback mode.
        """
        self._jitter_ms = jitter_ms
        self._redis_retry_interval = redis_retry_interval

        # Fallback state
        self._fallback_lock = threading.Lock()
        self._last_call_times: Dict[str, float] = {}
        self._using_fallback = False
        self._last_redis_retry = 0.0
        self._fallback_warned = False

    def _get_redis_client(self):
        """Get Redis client, returning None if unavailable."""
        try:
            from .redis_pool import get_redis_client
            client = get_redis_client()
            if client:
                client.ping()
                return client
        except Exception:
            pass
        return None

    def _try_redis(self, key: str, min_interval_s: float) -> Optional[float]:
        """
        Try to execute rate limiting via Redis.

        Returns:
            Wait time in seconds, or None if Redis is unavailable.
        """
        # If we're in fallback mode, check if it's time to retry Redis
        if self._using_fallback:
            now = time.monotonic()
            if now - self._last_redis_retry < self._redis_retry_interval:
                return None
            self._last_redis_retry = now

        client = self._get_redis_client()
        if client is None:
            if not self._using_fallback:
                self._using_fallback = True
                self._fallback_warned = False
            return None

        try:
            redis_key = f"ratelimit:{key}"
            result = client.eval(_LUA_RATE_LIMIT, 1, redis_key, str(min_interval_s))

            # If we were in fallback, we've recovered
            if self._using_fallback:
                self._using_fallback = False
                self._fallback_warned = False
                logger.info("Rate limiter: recovered Redis connection, switching back from fallback")

            # result is bytes from Redis
            if isinstance(result, bytes):
                return float(result.decode())
            return float(result)

        except Exception as e:
            logger.warning(f"Rate limiter: Redis error, falling back to in-process: {e}")
            self._using_fallback = True
            self._last_redis_retry = time.monotonic()
            return None

    def _fallback_wait(self, key: str, min_interval_s: float) -> float:
        """
        In-process fallback rate limiter using threading.Lock.

        Returns:
            Wait time in seconds.
        """
        if not self._fallback_warned:
            logger.warning(
                f"Rate limiter: using in-process fallback for '{key}' "
                f"(Redis unavailable, not distributed)"
            )
            self._fallback_warned = True

        with self._fallback_lock:
            now = time.monotonic()
            last = self._last_call_times.get(key, 0.0)
            elapsed = now - last

            if elapsed >= min_interval_s:
                self._last_call_times[key] = now
                return 0.0
            else:
                wait_time = min_interval_s - elapsed
                # Reserve the slot by advancing the timestamp
                self._last_call_times[key] = last + min_interval_s
                return wait_time

    def wait(
        self,
        key: str,
        min_interval_s: float,
        timeout_s: float = 120.0,
    ) -> float:
        """
        Block until rate limit allows the call. Returns actual time waited.

        Args:
            key: Rate limit key (e.g., "yfinance", "finviz", "yfinance:batch")
            min_interval_s: Minimum seconds between calls for this key
            timeout_s: Maximum seconds to wait before raising RateLimitTimeoutError

        Returns:
            Actual time waited in seconds

        Raises:
            RateLimitTimeoutError: If wait would exceed timeout_s
        """
        # Try Redis first
        wait_time = self._try_redis(key, min_interval_s)

        if wait_time is None:
            # Redis unavailable — use fallback
            wait_time = self._fallback_wait(key, min_interval_s)
            backend = "local"
        else:
            backend = "redis"

        # Check timeout before sleeping
        if wait_time > timeout_s:
            raise RateLimitTimeoutError(
                f"Rate limit wait for '{key}' would be {wait_time:.1f}s, "
                f"exceeding timeout of {timeout_s:.1f}s"
            )

        if wait_time > 0:
            # Add jitter to prevent thundering herd
            jitter = random.uniform(0, self._jitter_ms / 1000.0)
            total_wait = wait_time + jitter
            logger.debug(
                f"Rate limit: key={key} waited={total_wait:.3f}s backend={backend}"
            )
            time.sleep(total_wait)
            return total_wait

        logger.debug(f"Rate limit: key={key} waited=0.000s backend={backend}")
        return 0.0


# Module-level singleton
rate_limiter = RedisRateLimiter()
