"""Provider×market circuit breaker for transient 429 storms.

When an upstream provider (yfinance, finviz) sustains 429s for a single
market, every in-flight batch independently sleeps through its exponential
backoff. The breaker stores `closed | open | half_open` state in Redis so
all Celery workers agree, and short-circuits doomed calls until the
cooldown expires. State is per-(provider, market), so an open circuit on
``finviz:hk`` doesn't pause ``finviz:us``.

Falls back silently to a no-op (always-closed) when Redis is unavailable
— callers continue to use the underlying rate-limiter backoff in that
case.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from ..config import settings
from ..tasks.market_queues import market_suffix
from .rate_limiter import RateLimitTimeoutError

logger = logging.getLogger(__name__)


class CircuitOpenError(RateLimitTimeoutError):
    """Raised when the breaker for (provider, market) is open.

    Subclasses ``RateLimitTimeoutError`` so existing call sites that catch
    the latter continue to handle this gracefully; new code can match the
    subclass to log "circuit open, skipping" cleanly.
    """

    def __init__(self, provider: str, market: Optional[str], reopens_in_s: float):
        self.provider = provider
        self.market = market
        self.reopens_in_s = reopens_in_s
        super().__init__(
            f"Circuit open for {provider}:{market_suffix(market)}; "
            f"reopens in {reopens_in_s:.1f}s"
        )


# Lua script: atomically read state and auto-promote open→half_open after
# cooldown. Returns the resolved state plus seconds-until-reopen for the
# error path. Half-open promotion uses SETNX so only one probe gets through.
#
# KEYS[1] = state hash key (e.g., "circuit:finviz:us")
# KEYS[2] = half-open probe lock key
# ARGV[1] = current epoch seconds (string)
# ARGV[2] = cooldown seconds (string)
#
# Returns: ARRAY {state_string, reopens_in_s_string}
_LUA_CHECK = """
local state = redis.call('HGET', KEYS[1], 'state')
if not state or state == 'closed' then
    return {'closed', '0'}
end

local opened_at = tonumber(redis.call('HGET', KEYS[1], 'opened_at') or '0')
local cooldown = tonumber(ARGV[2])
local now = tonumber(ARGV[1])
local elapsed = now - opened_at
local remaining = cooldown - elapsed

if state == 'open' then
    if elapsed >= cooldown then
        -- Promote to half_open. Set the probe lock; if SETNX wins we admit.
        redis.call('HSET', KEYS[1], 'state', 'half_open')
        local got = redis.call('SET', KEYS[2], '1', 'NX', 'EX', math.ceil(cooldown))
        if got then
            return {'half_open_probe', '0'}
        else
            return {'open', tostring(cooldown)}
        end
    else
        return {'open', tostring(remaining)}
    end
end

if state == 'half_open' then
    -- Already half-open; only one caller may proceed at a time.
    local got = redis.call('SET', KEYS[2], '1', 'NX', 'EX', math.ceil(cooldown))
    if got then
        return {'half_open_probe', '0'}
    else
        return {'open', tostring(math.max(remaining, 1))}
    end
end

return {'closed', '0'}
"""


class ProviderCircuitBreaker:
    """Distributed (provider, market) circuit breaker on top of Redis."""

    _COOLDOWN_DEFAULTS_S = {
        "US": 120,
        "HK": 300,
        "IN": 300,
        "JP": 300,
        "TW": 300,
    }

    def __init__(self, redis_client_factory=None):
        self._redis_client_factory = redis_client_factory
        self._fallback_lock = threading.Lock()
        self._fallback_state: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(getattr(settings, "circuit_breaker_enabled", True))

    @property
    def threshold(self) -> int:
        return int(getattr(settings, "circuit_breaker_threshold", 3) or 3)

    def cooldown_s(self, market: Optional[str]) -> int:
        if market:
            attr = f"circuit_breaker_cooldown_{market.lower()}"
            override = getattr(settings, attr, None)
            # Explicit ``is not None`` so a configured ``0`` cooldown
            # (typically only used in tests) is honoured.
            if override is not None:
                return int(override)
            return self._COOLDOWN_DEFAULTS_S.get(market.upper(), 300)
        return 120

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------
    @staticmethod
    def state_key(provider: str, market: Optional[str]) -> str:
        return f"circuit:{provider}:{market_suffix(market)}"

    @staticmethod
    def probe_key(provider: str, market: Optional[str]) -> str:
        return f"circuit:{provider}:{market_suffix(market)}:probe"

    @staticmethod
    def events_key(provider: str, market: Optional[str], day: Optional[str] = None) -> str:
        from datetime import datetime
        day = day or datetime.now().strftime("%Y%m%d")
        return f"circuit:events:{provider}:{market_suffix(market)}:{day}"

    # ------------------------------------------------------------------
    # Redis access
    # ------------------------------------------------------------------
    def _redis(self):
        if self._redis_client_factory is not None:
            try:
                return self._redis_client_factory()
            except Exception:
                return None
        try:
            from .redis_pool import get_redis_client
            return get_redis_client()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------
    def check(self, provider: str, market: Optional[str]) -> str:
        """Return resolved state: ``closed``, ``open``, or ``half_open_probe``.

        ``open`` means callers should bail (raise ``CircuitOpenError`` or
        skip). ``half_open_probe`` admits exactly one probe call; the
        caller must invoke ``record_success`` or ``record_429`` afterwards
        so the breaker can transition.
        """
        if not self.enabled:
            return "closed"
        client = self._redis()
        if client is None:
            return self._fallback_check(provider, market)
        state_key = self.state_key(provider, market)
        probe_key = self.probe_key(provider, market)
        cooldown = self.cooldown_s(market)
        try:
            result = client.eval(
                _LUA_CHECK, 2, state_key, probe_key,
                str(time.time()), str(cooldown),
            )
            state, _remaining = self._decode_check_result(result)
            return state
        except Exception as exc:
            logger.debug("CircuitBreaker.check: Redis error (%s); treating as closed", exc)
            return "closed"

    def raise_if_open(self, provider: str, market: Optional[str]) -> str:
        """Like ``check`` but raises ``CircuitOpenError`` when open.

        Returns the resolved state when the call may proceed (``closed`` or
        ``half_open_probe``).
        """
        state = self.check(provider, market)
        if state == "open":
            raise CircuitOpenError(provider, market, float(self.cooldown_s(market)))
        return state

    def record_429(self, provider: str, market: Optional[str]) -> None:
        """Increment consecutive-429 counter; trip to ``open`` at threshold."""
        if not self.enabled:
            return
        client = self._redis()
        if client is None:
            self._fallback_record_429(provider, market)
            return
        state_key = self.state_key(provider, market)
        events_key = self.events_key(provider, market)
        threshold = self.threshold
        cooldown = self.cooldown_s(market)
        ttl = max(int(cooldown * 2), 60)
        try:
            pipe = client.pipeline()
            pipe.hincrby(state_key, "consecutive_429", 1)
            pipe.hget(state_key, "state")
            pipe.expire(state_key, ttl)
            results = pipe.execute()
            consecutive = int(results[0] or 0)
            current_state = (results[1] or b"closed")
            if isinstance(current_state, bytes):
                current_state = current_state.decode()

            if current_state == "half_open":
                # Probe failed → reopen and reset opened_at so the cooldown
                # window restarts.
                self._transition(client, state_key, "half_open", "open", consecutive)
                pipe2 = client.pipeline()
                pipe2.hset(state_key, mapping={"opened_at": str(time.time())})
                pipe2.delete(self.probe_key(provider, market))
                pipe2.incr(events_key)
                pipe2.expire(events_key, 90 * 86400)
                pipe2.expire(state_key, ttl)
                pipe2.execute()
                logger.info(
                    "CircuitBreaker: half-open probe failed for %s:%s; "
                    "reopening (consecutive_429=%d)",
                    provider, market_suffix(market), consecutive,
                )
                return

            if current_state != "open" and consecutive >= threshold:
                pipe2 = client.pipeline()
                pipe2.hset(state_key, mapping={
                    "state": "open",
                    "opened_at": str(time.time()),
                })
                pipe2.incr(events_key)
                pipe2.expire(events_key, 90 * 86400)
                pipe2.expire(state_key, ttl)
                pipe2.execute()
                logger.warning(
                    "CircuitBreaker: tripping OPEN for %s:%s (consecutive_429=%d, "
                    "threshold=%d, cooldown=%ds)",
                    provider, market_suffix(market),
                    consecutive, threshold, cooldown,
                )
        except Exception as exc:
            logger.debug("CircuitBreaker.record_429: Redis error: %s", exc)

    def record_success(self, provider: str, market: Optional[str]) -> None:
        """Reset counter; promote half_open → closed when applicable."""
        if not self.enabled:
            return
        client = self._redis()
        if client is None:
            self._fallback_record_success(provider, market)
            return
        state_key = self.state_key(provider, market)
        try:
            current_state = client.hget(state_key, "state")
            if isinstance(current_state, bytes):
                current_state = current_state.decode()

            if current_state == "half_open":
                self._transition(client, state_key, "half_open", "closed", 0)
                client.delete(self.probe_key(provider, market))
                logger.info(
                    "CircuitBreaker: closing %s:%s after successful probe",
                    provider, market_suffix(market),
                )
                events_key = self.events_key(provider, market)
                client.incr(events_key)
                client.expire(events_key, 90 * 86400)
            elif current_state == "closed" or current_state is None:
                # Reset the consecutive counter so transient single 429s
                # don't accumulate forever.
                client.hset(state_key, "consecutive_429", 0)
        except Exception as exc:
            logger.debug("CircuitBreaker.record_success: Redis error: %s", exc)

    def reset(self, provider: str, market: Optional[str]) -> None:
        """Force-reset state (admin/test helper)."""
        client = self._redis()
        if client is None:
            self._fallback_state.pop(self._fallback_key(provider, market), None)
            return
        try:
            client.delete(self.state_key(provider, market))
            client.delete(self.probe_key(provider, market))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _decode_check_result(result) -> tuple[str, float]:
        try:
            state, remaining = result[0], result[1]
            if isinstance(state, bytes):
                state = state.decode()
            if isinstance(remaining, bytes):
                remaining = remaining.decode()
            return state, float(remaining)
        except Exception:
            return "closed", 0.0

    def _transition(self, client, state_key: str, from_state: str, to_state: str, consecutive: int) -> None:
        try:
            client.hset(state_key, mapping={
                "state": to_state,
                "consecutive_429": str(consecutive),
            })
        except Exception as exc:
            logger.debug("CircuitBreaker._transition: %s", exc)

    # --- in-process fallback (Redis-unavailable) ----------------------
    def _fallback_key(self, provider: str, market: Optional[str]) -> str:
        return self.state_key(provider, market)

    def _fallback_check(self, provider: str, market: Optional[str]) -> str:
        with self._fallback_lock:
            entry = self._fallback_state.get(self._fallback_key(provider, market))
            if not entry:
                return "closed"
            state = entry["state"]
            if state == "open":
                if time.time() - entry["opened_at"] >= self.cooldown_s(market):
                    entry["state"] = "half_open"
                    entry["probe_taken"] = True
                    return "half_open_probe"
                return "open"
            if state == "half_open":
                if entry.get("probe_taken"):
                    return "open"
                entry["probe_taken"] = True
                return "half_open_probe"
            return "closed"

    def _fallback_record_429(self, provider: str, market: Optional[str]) -> None:
        with self._fallback_lock:
            key = self._fallback_key(provider, market)
            entry = self._fallback_state.setdefault(
                key,
                {"state": "closed", "consecutive_429": 0, "opened_at": 0.0},
            )
            entry["consecutive_429"] += 1
            if entry["state"] == "half_open":
                entry["state"] = "open"
                entry["opened_at"] = time.time()
                entry.pop("probe_taken", None)
            elif entry["consecutive_429"] >= self.threshold:
                entry["state"] = "open"
                entry["opened_at"] = time.time()

    def _fallback_record_success(self, provider: str, market: Optional[str]) -> None:
        with self._fallback_lock:
            key = self._fallback_key(provider, market)
            entry = self._fallback_state.get(key)
            if not entry:
                return
            if entry["state"] == "half_open":
                entry["state"] = "closed"
            entry["consecutive_429"] = 0
            entry.pop("probe_taken", None)


# Module-level singleton
_default_breaker: Optional[ProviderCircuitBreaker] = None
_default_breaker_lock = threading.Lock()


def get_circuit_breaker() -> ProviderCircuitBreaker:
    """Return the process-wide ProviderCircuitBreaker singleton."""
    global _default_breaker
    if _default_breaker is None:
        with _default_breaker_lock:
            if _default_breaker is None:
                _default_breaker = ProviderCircuitBreaker()
    return _default_breaker
