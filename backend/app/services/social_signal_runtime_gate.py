"""Redis ownership lease and manual-dispatch cooldown for Social Signals."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import ceil

from app.domain.social_signals.records import SocialSourceBatch, SocialSourceOutcome


PROVIDER_LEASE_KEY = "social-signals:provider-read:lease"
MANUAL_COOLDOWN_KEY = "social-signals:manual-refresh:cooldown"
PROVIDER_COOLDOWN_KEY = "social-signals:provider:{provider}:cooldown"
LLM_REQUEST_LEASE_KEY = "social-signals:llm-request:lease"
LLM_REQUEST_SPACING_KEY = "social-signals:llm-request:spacing"
_PROVIDER_COOLDOWN_CODES = {
    "rate_limited", "reauthentication_required", "provider_error",
}

_RELEASE = """
local value = redis.call('get', KEYS[1])
if value == ARGV[1] then
  return redis.call('del', KEYS[1])
end
return 0
"""

_RENEW = """
local value = redis.call('get', KEYS[1])
if value == ARGV[1] then
  return redis.call('expire', KEYS[1], ARGV[2])
end
return 0
"""


class RedisSocialSignalGate:
    def __init__(self, redis_client):
        self.redis = redis_client
        self._release = redis_client.register_script(_RELEASE)
        self._renew = redis_client.register_script(_RENEW)

    def acquire(self, owner: str, ttl_seconds: int) -> bool:
        if not owner or ttl_seconds <= 0:
            raise ValueError("invalid_social_provider_lease")
        return bool(self.redis.set(PROVIDER_LEASE_KEY, owner, nx=True, ex=ttl_seconds))

    def release(self, owner: str) -> None:
        if owner:
            self._release(keys=[PROVIDER_LEASE_KEY], args=[owner])

    def renew(self, owner: str, ttl_seconds: int) -> bool:
        if not owner or ttl_seconds <= 0:
            raise ValueError("invalid_social_provider_lease")
        return bool(
            self._renew(
                keys=[PROVIDER_LEASE_KEY],
                args=[owner, ttl_seconds],
            )
        )

    def acquire_manual_cooldown(self, owner: str, ttl_seconds: int) -> tuple[bool, int]:
        if not owner or ttl_seconds <= 0:
            raise ValueError("invalid_social_manual_cooldown")
        accepted = bool(self.redis.set(MANUAL_COOLDOWN_KEY, owner, nx=True, ex=ttl_seconds))
        if accepted:
            return True, ttl_seconds
        remaining = self.redis.ttl(MANUAL_COOLDOWN_KEY)
        return False, max(1, int(remaining) if isinstance(remaining, int) and remaining > 0 else ttl_seconds)

    def release_manual_cooldown(self, owner: str) -> None:
        if owner:
            self._release(keys=[MANUAL_COOLDOWN_KEY], args=[owner])

    def provider_cooldown(self, provider: str, now: datetime):
        key = PROVIDER_COOLDOWN_KEY.format(provider=provider)
        remaining = self.redis.ttl(key)
        if not isinstance(remaining, int) or remaining <= 0:
            return None
        code = self.redis.get(key)
        if isinstance(code, bytes):
            code = code.decode("utf-8", errors="replace")
        if code not in _PROVIDER_COOLDOWN_CODES:
            code = "provider_error"
        return code, now + timedelta(seconds=remaining)

    def start_provider_cooldown(self, provider: str, code: str, reset_at: datetime, now: datetime):
        if code not in _PROVIDER_COOLDOWN_CODES:
            code = "provider_error"
        seconds = max(1, ceil((reset_at - now).total_seconds()))
        self.redis.set(PROVIDER_COOLDOWN_KEY.format(provider=provider), code, ex=seconds)


class RedisSocialLLMRequestGate:
    """One global model request lease plus a shared start-to-start interval."""

    def __init__(self, redis_client):
        self.redis = redis_client
        self._release = redis_client.register_script(_RELEASE)

    def acquire(self, owner: str, ttl_seconds: int) -> bool:
        if not owner or ttl_seconds <= 0:
            raise ValueError("invalid_social_llm_request_lease")
        return bool(self.redis.set(LLM_REQUEST_LEASE_KEY, owner, nx=True, ex=ttl_seconds))

    def wait_seconds(self) -> int:
        remaining = self.redis.ttl(LLM_REQUEST_SPACING_KEY)
        return max(0, remaining if isinstance(remaining, int) else 0)

    def mark_started(self, interval_seconds: float) -> None:
        if interval_seconds > 0:
            self.redis.set(
                LLM_REQUEST_SPACING_KEY, "1", ex=max(1, ceil(interval_seconds))
            )

    def release(self, owner: str) -> None:
        if owner:
            self._release(keys=[LLM_REQUEST_LEASE_KEY], args=[owner])


class SharedCooldownSocialProvider:
    """Persist adapter cooldowns so reconstructed workers remain fail-closed."""

    def __init__(self, provider_name, provider, gate, *, clock=None):
        self.provider_name = provider_name
        self.provider = provider
        self.gate = gate
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def read_source(self, request):
        now = self.clock()
        active = self.gate.provider_cooldown(self.provider_name, now)
        if active is not None:
            code, reset_at = active
            return SocialSourceBatch(request, (), SocialSourceOutcome(
                "failed", "failed", "limited", (code,), (), None, None,
                0, None, code, rate_limit_reset_at=reset_at,
            ))
        batch = self.provider.read_source(request)
        reset_at = batch.outcome.rate_limit_reset_at
        if batch.outcome.read_status == "failed" and reset_at is not None:
            self.gate.start_provider_cooldown(
                self.provider_name,
                batch.outcome.error_code or "provider_error",
                reset_at,
                now,
            )
        return batch


__all__ = [
    "MANUAL_COOLDOWN_KEY",
    "PROVIDER_COOLDOWN_KEY",
    "PROVIDER_LEASE_KEY",
    "LLM_REQUEST_LEASE_KEY",
    "LLM_REQUEST_SPACING_KEY",
    "RedisSocialLLMRequestGate",
    "RedisSocialSignalGate",
    "SharedCooldownSocialProvider",
]
