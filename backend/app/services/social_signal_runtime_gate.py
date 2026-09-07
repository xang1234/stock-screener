"""Redis ownership lease and manual-dispatch cooldown for Social Signals."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import ceil

from app.domain.social_signals.records import SocialSourceBatch, SocialSourceOutcome


PROVIDER_LEASE_KEY = "social-signals:provider-read:lease"
MANUAL_COOLDOWN_KEY = "social-signals:manual-refresh:cooldown"
PROVIDER_COOLDOWN_KEY = "social-signals:provider:{provider}:cooldown"
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


class RedisSocialSignalGate:
    def __init__(self, redis_client):
        self.redis = redis_client
        self._release = redis_client.register_script(_RELEASE)

    def acquire(self, owner: str, ttl_seconds: int) -> bool:
        if not owner or ttl_seconds <= 0:
            raise ValueError("invalid_social_provider_lease")
        return bool(self.redis.set(PROVIDER_LEASE_KEY, owner, nx=True, ex=ttl_seconds))

    def release(self, owner: str) -> None:
        if owner:
            self._release(keys=[PROVIDER_LEASE_KEY], args=[owner])

    def acquire_manual_cooldown(self, owner: str, ttl_seconds: int) -> tuple[bool, int]:
        if not owner or ttl_seconds <= 0:
            raise ValueError("invalid_social_manual_cooldown")
        accepted = bool(self.redis.set(MANUAL_COOLDOWN_KEY, owner, nx=True, ex=ttl_seconds))
        if accepted:
            return True, ttl_seconds
        remaining = self.redis.ttl(MANUAL_COOLDOWN_KEY)
        return False, max(1, int(remaining) if isinstance(remaining, int) and remaining > 0 else ttl_seconds)

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
    "RedisSocialSignalGate",
    "SharedCooldownSocialProvider",
]
