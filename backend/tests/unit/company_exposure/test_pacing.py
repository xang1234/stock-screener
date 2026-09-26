from __future__ import annotations

import pytest

from app.services.company_exposure.pacing import PacingUnavailable, ResearchRateGate
from app.services.rate_budget_policy import RateBudgetPolicy
from app.services.rate_limiter import RateLimiterUnavailable, RedisRateLimiter


class _NoRedisLimiter(RedisRateLimiter):
    def _get_redis_client(self):
        return None


def test_strict_mode_refuses_process_local_fallback():
    limiter = _NoRedisLimiter()
    with pytest.raises(RateLimiterUnavailable):
        limiter.wait("sec_edgar:shared", 0.15, strict=True)
    # Legacy callers keep their fallback behaviour unchanged.
    assert limiter.wait("sec_edgar:shared", 0.0) >= 0.0


def test_research_gate_uses_existing_sec_key_and_fails_closed():
    gate = ResearchRateGate(limiter=_NoRedisLimiter(), policy=RateBudgetPolicy())
    with pytest.raises(PacingUnavailable) as raised:
        gate.acquire("sec_edgar")
    assert raised.value.code == "distributed_pacing_unavailable"
    assert RateBudgetPolicy.provider_key("sec_edgar", None) == "sec_edgar:shared"
    assert RateBudgetPolicy._global_interval_for("sec_edgar") == 0.15
    assert RateBudgetPolicy._global_interval_for("issuer_web") == 1.0
