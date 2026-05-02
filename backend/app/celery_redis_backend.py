"""Redis result backend extensions for Celery."""

from __future__ import annotations

from celery.backends.redis import RedisBackend
from redis.exceptions import BusyLoadingError


class RetryableRedisBackend(RedisBackend):
    """Redis backend that lets Celery retry transient Redis connection states."""

    def exception_safe_to_retry(self, exc: Exception) -> bool:
        """Return whether Celery may retry the result backend operation."""
        return isinstance(exc, BusyLoadingError) or isinstance(exc, self.connection_errors)


__all__ = ["RetryableRedisBackend"]
