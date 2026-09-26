"""Shared provider pacing for every research HTTP attempt.

Research acquires the same ``RateBudgetPolicy`` keys as every other caller
(SEC uses ``sec_edgar``), in strict distributed mode: when Redis is
unavailable research refuses to proceed instead of pacing itself with a
process-local limiter. Waits happen outside any database transaction.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.rate_budget_policy import get_rate_budget_policy
from app.services.rate_limiter import (
    RateLimiterUnavailable,
    RateLimitTimeoutError,
    RedisRateLimiter,
)


class PacingUnavailable(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class RateTicket:
    provider: str
    key: str
    waited_seconds: float


class ResearchRateGate:
    def __init__(self, limiter: RedisRateLimiter | None = None, policy=None):
        self._limiter = limiter or RedisRateLimiter()
        self._policy = policy or get_rate_budget_policy()

    def acquire(
        self, provider: str, market: str | None = None, timeout_s: float = 60.0
    ) -> RateTicket:
        key = self._policy.provider_key(provider, market)
        interval = self._policy.get_rate_interval(provider, market)
        try:
            waited = self._limiter.wait(key, interval, timeout_s=timeout_s, strict=True)
        except RateLimiterUnavailable:
            raise PacingUnavailable("distributed_pacing_unavailable") from None
        except RateLimitTimeoutError:
            raise PacingUnavailable("pacing_wait_exceeded") from None
        if waited > 0:
            self._policy.record_throttle_wait(provider, market, waited)
        return RateTicket(provider=provider, key=key, waited_seconds=waited)
