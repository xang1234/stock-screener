"""Failure telemetry helpers for price cache symbol refresh operations."""

from __future__ import annotations


class PriceCacheFailureTelemetry:
    """Tracks per-symbol failure counters in Redis."""

    def __init__(
        self,
        *,
        logger,
        redis_client,
        key_template: str,
        ttl_seconds: int,
    ) -> None:
        self._logger = logger
        self._redis_client = redis_client
        self._key_template = key_template
        self._ttl_seconds = ttl_seconds

    def record_symbol_failure(self, symbol: str) -> int:
        if not self._redis_client:
            return 0
        try:
            key = self._key_template.format(symbol=symbol)
            count = self._redis_client.incr(key)
            self._redis_client.expire(key, self._ttl_seconds)
            return count
        except Exception as exc:
            self._logger.error("Error recording failure for %s: %s", symbol, exc)
            return 0

    def clear_symbol_failure(self, symbol: str) -> None:
        if not self._redis_client:
            return
        try:
            key = self._key_template.format(symbol=symbol)
            self._redis_client.delete(key)
        except Exception as exc:
            self._logger.error("Error clearing failure for %s: %s", symbol, exc)
