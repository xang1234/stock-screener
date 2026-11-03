"""Redis-backed workload leases for Celery task coordination."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import json
import logging
import time
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

from celery.exceptions import Retry

try:
    import redis  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    redis = None

from ..config import settings
from .lease_renewal import (
    LEASE_REDIS_CONNECT_TIMEOUT_SECONDS,
    LEASE_REDIS_SOCKET_TIMEOUT_SECONDS,
    RENEW_LEASE_LUA,
    LeaseNotHeld,
    keep_leases_alive,
    lease_ttl_seconds,
)
from .market_queues import SUPPORTED_MARKETS, market_suffix, normalize_market
from .transient_database import retry_transient_database_error
from .workload_fence import LeaseLost, workload_fence

EXTERNAL_FETCH_GLOBAL_KEY = "external_fetch_global"
MARKET_WORKLOAD_PREFIX = "market_workload"
PIPELINE_CHAIN_PREFIX = "pipeline_chain"
PIPELINE_CHAIN_TASKS_PREFIX = "pipeline_chain_tasks"
PIPELINE_CHAIN_MISSES_PREFIX = "pipeline_chain_misses"
PIPELINE_CHAIN_HEARTBEAT_KEY = "pipeline_chain_heartbeat"
PIPELINE_CATCH_UP_PREFIX = "pipeline_catch_up"
# Held from dispatch until the daily pipeline chain's last stage or first
# failure. renew_daily_market_pipeline_claims renews it while any stage of
# the chain is still queued, reserved or running, and releases it once the
# chain is dead (#605), so a chain queued behind a long data-fetch backlog
# keeps it and a revoked one loses it. This TTL is only the safety net for a
# heartbeat outage: such a claim never locks the market for longer.
PIPELINE_CHAIN_TTL_SECONDS = 48 * 3600
PIPELINE_CHAIN_HEARTBEAT_SECONDS = 300
# Outlives a weekend of hourly catch-up ticks for one session.
PIPELINE_CATCH_UP_TTL_SECONDS = 96 * 3600
# How long heartbeats must find no trace of a chain before its claim is
# released. One check can fall between a worker taking a stage and storing its
# state, so it takes two; half an interval apart, so two heartbeats that
# overlap (a slow one and the next) cannot both count one such gap.
PIPELINE_CHAIN_MISSING_RELEASE_SECONDS = PIPELINE_CHAIN_HEARTBEAT_SECONDS / 2
# Keeps heartbeats from overlapping (#627). A heartbeat killed while holding
# it delays the next beats by at most this long; one slower than this can no
# longer free a chain it found missing.
PIPELINE_CHAIN_HEARTBEAT_LOCK_SECONDS = 6 * PIPELINE_CHAIN_HEARTBEAT_SECONDS
COORDINATION_WAIT_MAX_RETRIES = 10_000
logger = logging.getLogger(__name__)
_SERIALIZED_MARKET_WORKLOAD_DISABLED: ContextVar[bool] = ContextVar(
    "serialized_market_workload_disabled",
    default=False,
)

_RELEASE_LUA = """
local val = redis.call('get', KEYS[1])
if val and string.find(val, ARGV[1], 1, true) then
    return redis.call('del', KEYS[1])
end
return 0
"""


# Claim the chain and record its task ids in one step, so the heartbeat never
# sees a claim without them.
_CLAIM_PIPELINE_CHAIN_LUA = """
if redis.call('set', KEYS[1], ARGV[1], 'NX', 'EX', ARGV[3]) then
    redis.call('set', KEYS[2], ARGV[2], 'EX', ARGV[3])
    return 1
end
return 0
"""


# Release a chain heartbeats found missing, but only while the missing marker
# that heartbeat read is still in place (an overlapping heartbeat that saw the
# chain alive meanwhile cleared it) and the heartbeat still holds the
# heartbeat lock (one that outlived it may overlap a beat that just saw the
# chain alive but has not cleared the marker yet) (#627).
# 0: not this token's claim; 1: released; 2: kept.
_RELEASE_MISSING_PIPELINE_CHAIN_LUA = """
local val = redis.call('get', KEYS[1])
if not (val and string.find(val, ARGV[1], 1, true)) then
    return 0
end
if redis.call('get', KEYS[2]) ~= ARGV[2] then
    return 2
end
local beat = redis.call('get', KEYS[3])
if not (beat and string.find(beat, ARGV[3], 1, true)) then
    return 2
end
return redis.call('del', KEYS[1])
"""


# Keyed by token, not market, and left to expire after release: a heartbeat
# that read an old holder still finds that chain's ids, and its owner-checked
# renew or release then leaves the newer claim alone.
def _pipeline_chain_tasks_key(token: str) -> str:
    return f"{PIPELINE_CHAIN_TASKS_PREFIX}:{token}"


def _pipeline_chain_misses_key(token: str) -> str:
    return f"{PIPELINE_CHAIN_MISSES_PREFIX}:{token}"


def _market_workload_key(market: Optional[str]) -> str:
    return f"{MARKET_WORKLOAD_PREFIX}:{market_suffix(market)}"


def _pipeline_chain_key(market: Optional[str]) -> str:
    return f"{PIPELINE_CHAIN_PREFIX}:{market_suffix(market)}"


def _pipeline_catch_up_key(market: Optional[str], session: str) -> str:
    return f"{PIPELINE_CATCH_UP_PREFIX}:{market_suffix(market)}:{session}"


def _parse_task_id(lock_value: bytes) -> Optional[str]:
    try:
        parts = lock_value.decode().split(":", 2)
        if len(parts) >= 2:
            return parts[1]
    except Exception:
        pass
    return None


@contextmanager
def disable_serialized_market_workload() -> Iterator[None]:
    """Temporarily bypass Redis-backed market-workload leases."""
    token = _SERIALIZED_MARKET_WORKLOAD_DISABLED.set(True)
    try:
        yield
    finally:
        _SERIALIZED_MARKET_WORKLOAD_DISABLED.reset(token)


class WorkloadCoordination:
    """Coordinates global external fetch and per-market workload leases."""

    def __init__(self) -> None:
        if redis is None:
            raise RuntimeError("Redis package is not installed; WorkloadCoordination is unavailable")
        self.redis = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            socket_connect_timeout=LEASE_REDIS_CONNECT_TIMEOUT_SECONDS,
            socket_timeout=LEASE_REDIS_SOCKET_TIMEOUT_SECONDS,
        )
        # Short lease, renewed by the holder's heartbeat (see lease_renewal).
        self.lock_timeout = lease_ttl_seconds(
            getattr(settings, "data_fetch_lock_timeout", None)
        )
        self._release_script = self.redis.register_script(_RELEASE_LUA)
        self._renew_script = self.redis.register_script(RENEW_LEASE_LUA)
        self._claim_chain_script = self.redis.register_script(_CLAIM_PIPELINE_CHAIN_LUA)
        self._release_missing_chain_script = self.redis.register_script(
            _RELEASE_MISSING_PIPELINE_CHAIN_LUA
        )

    def _acquire(self, key: str, task_name: str, task_id: str) -> Tuple[bool, bool]:
        if task_id != "unknown":
            current = self.redis.get(key)
            if current:
                holder_task_id = _parse_task_id(current)
                if holder_task_id and holder_task_id == task_id:
                    return (True, True)

        lock_value = f"{task_name}:{task_id}:{datetime.now(timezone.utc).isoformat()}"
        acquired = self.redis.set(key, lock_value, nx=True, ex=self.lock_timeout)
        return (bool(acquired), False)

    def _release(self, key: str, task_id: str) -> bool:
        return bool(self._release_script(keys=[key], args=[f":{task_id}:"]))

    def _renew(self, key: str, task_id: str) -> bool:
        return bool(
            self._renew_script(keys=[key], args=[f":{task_id}:", self.lock_timeout])
        )

    def _holder(self, key: str) -> Optional[Dict[str, Any]]:
        current = self.redis.get(key)
        if not current:
            return None
        try:
            parts = current.decode().split(":")
            if len(parts) >= 3:
                return {
                    "task_name": parts[0],
                    "task_id": parts[1],
                    "started_at": ":".join(parts[2:]),
                    "ttl_seconds": self.redis.ttl(key),
                    "lease_key": key,
                }
        except Exception:
            return {"raw": current.decode(), "lease_key": key}
        return {"raw": current.decode(), "lease_key": key}

    def acquire_external_fetch(self, task_name: str, task_id: str) -> Tuple[bool, bool]:
        return self._acquire(EXTERNAL_FETCH_GLOBAL_KEY, task_name, task_id)

    def release_external_fetch(self, task_id: str) -> bool:
        return self._release(EXTERNAL_FETCH_GLOBAL_KEY, task_id)

    def renew_external_fetch(self, task_id: str) -> bool:
        return self._renew(EXTERNAL_FETCH_GLOBAL_KEY, task_id)

    def get_external_fetch_holder(self) -> Optional[Dict[str, Any]]:
        return self._holder(EXTERNAL_FETCH_GLOBAL_KEY)

    def acquire_market_workload(
        self,
        task_name: str,
        task_id: str,
        *,
        market: Optional[str],
    ) -> Tuple[bool, bool]:
        return self._acquire(_market_workload_key(market), task_name, task_id)

    def release_market_workload(self, task_id: str, *, market: Optional[str]) -> bool:
        return self._release(_market_workload_key(market), task_id)

    def renew_market_workload(self, task_id: str, *, market: Optional[str]) -> bool:
        return self._renew(_market_workload_key(market), task_id)

    def get_market_workload_holder(self, market: Optional[str]) -> Optional[Dict[str, Any]]:
        return self._holder(_market_workload_key(market))

    def claim_pipeline_chain(
        self, token: str, *, market: Optional[str], task_ids: Sequence[str]
    ) -> bool:
        """Claim the market's daily pipeline chain; False if one already holds it.

        The per-stage market workload lease is free between stages, so it
        cannot stop a second chain; this lease spans the whole chain.
        ``task_ids`` are the chain's stages in order, which the heartbeat
        reads to tell whether the chain is still alive.
        """
        value = f"daily_market_pipeline:{token}:{datetime.now(timezone.utc).isoformat()}"
        return bool(
            self._claim_chain_script(
                keys=[_pipeline_chain_key(market), _pipeline_chain_tasks_key(token)],
                args=[value, json.dumps(list(task_ids)), PIPELINE_CHAIN_TTL_SECONDS],
            )
        )

    def renew_pipeline_chain(self, token: str, *, market: Optional[str]) -> bool:
        """Restart the claim's TTL, only if this token still holds it."""
        renewed = bool(
            self._renew_script(
                keys=[_pipeline_chain_key(market)],
                args=[f":{token}:", PIPELINE_CHAIN_TTL_SECONDS],
            )
        )
        if renewed:
            self.redis.expire(_pipeline_chain_tasks_key(token), PIPELINE_CHAIN_TTL_SECONDS)
        return renewed

    def count_pipeline_catch_up(self, market: Optional[str], session: str) -> int:
        """Record one catch-up dispatch of the market's session and return how
        many it has had."""
        pipe = self.redis.pipeline()
        pipe.incr(_pipeline_catch_up_key(market, session))
        pipe.expire(_pipeline_catch_up_key(market, session), PIPELINE_CATCH_UP_TTL_SECONDS)
        count, _ = pipe.execute()
        return int(count)

    def get_pipeline_chain_task_ids(self, token: str) -> Optional[list[str]]:
        """The chain's stage ids in order; None for a claim taken before #605."""
        raw = self.redis.get(_pipeline_chain_tasks_key(token))
        return None if raw is None else json.loads(raw)

    def acquire_pipeline_chain_heartbeat(self, token: str) -> bool:
        """False while another heartbeat runs."""
        return bool(
            self.redis.set(
                PIPELINE_CHAIN_HEARTBEAT_KEY, f"heartbeat:{token}:", nx=True,
                ex=PIPELINE_CHAIN_HEARTBEAT_LOCK_SECONDS,
            )
        )

    def release_pipeline_chain_heartbeat(self, token: str) -> bool:
        return self._release(PIPELINE_CHAIN_HEARTBEAT_KEY, token)

    def pipeline_chain_missing(self, token: str) -> Tuple[float, Optional[str]]:
        """(seconds since heartbeats started finding no trace of the chain,
        the missing marker read); 0 seconds when this check is the first.
        Pass the marker to ``release_missing_pipeline_chain``."""
        key = _pipeline_chain_misses_key(token)
        now = time.time()
        self.redis.set(key, repr(now), nx=True, ex=PIPELINE_CHAIN_TTL_SECONDS)
        since = self.redis.get(key)
        if since is None:  # cleared by a heartbeat that saw the chain alive
            return 0.0, None
        since = since.decode()
        return max(now - float(since), 0.0), since

    def clear_pipeline_chain_missing(self, token: str) -> None:
        self.redis.delete(_pipeline_chain_misses_key(token))

    def release_pipeline_chain(self, token: str, *, market: Optional[str]) -> bool:
        """Release only if this token still holds it (never a newer chain's)."""
        return self._release(_pipeline_chain_key(market), token)

    def release_missing_pipeline_chain(
        self, token: str, *, market: Optional[str], missing_since: str, heartbeat: str
    ) -> str:
        """Release a chain found missing, atomically with a check that this
        token still holds the claim, its missing marker still reads
        ``missing_since``, and ``heartbeat`` still holds the heartbeat lock.
        "released", "kept" (the marker changed, or the heartbeat lock expired:
        another beat may have seen the chain alive), or "lost"."""
        result = self._release_missing_chain_script(
            keys=[
                _pipeline_chain_key(market),
                _pipeline_chain_misses_key(token),
                PIPELINE_CHAIN_HEARTBEAT_KEY,
            ],
            args=[f":{token}:", missing_since, f":{heartbeat}:"],
        )
        return {1: "released", 2: "kept"}.get(int(result), "lost")

    def get_pipeline_chain_holder(self, market: Optional[str]) -> Optional[Dict[str, Any]]:
        return self._holder(_pipeline_chain_key(market))

    def get_pipeline_chain_starts(self, markets) -> dict[str, Optional[str]]:
        """{MARKET: started_at, or None when free} in one MGET round trip, so
        a hung Redis costs one socket timeout, not one per market."""
        codes = [normalize_market(m) for m in markets]
        values = self.redis.mget([_pipeline_chain_key(c) for c in codes]) if codes else []
        starts: dict[str, Optional[str]] = {}
        for code, value in zip(codes, values):
            if value is None:
                starts[code] = None
                continue
            parts = value.decode().split(":", 2)
            # "daily_market_pipeline:<token>:<iso>"; held even if unparseable.
            starts[code] = parts[2] if len(parts) == 3 else ""
        return starts

    def get_pipeline_chain_holders(self) -> dict[str, Optional[Dict[str, Any]]]:
        return {
            normalize_market(market): self.get_pipeline_chain_holder(market)
            for market in SUPPORTED_MARKETS
        }

    def get_market_workload_holders(self) -> dict[str, Optional[Dict[str, Any]]]:
        holders: dict[str, Optional[Dict[str, Any]]] = {}
        for market in SUPPORTED_MARKETS:
            holders[normalize_market(market)] = self.get_market_workload_holder(market)
        return holders


def _coordination_retry(task: Any, message: str) -> None:
    retries = getattr(getattr(task, "request", None), "retries", 0) or 0
    countdown = min(15 * (2 ** retries), 300)
    raise task.retry(
        exc=RuntimeError(message),
        countdown=countdown,
        max_retries=COORDINATION_WAIT_MAX_RETRIES,
    )


def _wait_for_market_workload(task: Any, coordination: Any, market: Optional[str]):
    """Retry (or report waiting) while another task holds the market lease."""
    holder = coordination.get_market_workload_holder(market) or {}
    wait_reason = f"waiting_for_market_workload:{normalize_market(market)}"
    if task is not None and hasattr(task, "retry"):
        _coordination_retry(
            task,
            f"{wait_reason} ({holder.get('task_name', 'unknown')})",
        )
    return {
        "status": "waiting",
        "wait_reason": wait_reason,
        "running_task_name": holder.get("task_name"),
        "running_task_id": holder.get("task_id"),
    }


def serialized_market_workload(task_name: str):
    """Serialize compute/write work per market without the global fetch lease."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if _SERIALIZED_MARKET_WORKLOAD_DISABLED.get():
                return func(*args, **kwargs)

            market_value: Optional[str] = kwargs.get("market")
            task = args[0] if args and hasattr(args[0], "request") else None
            task_id = getattr(getattr(task, "request", None), "id", None) or "unknown"

            from ..wiring.bootstrap import get_workload_coordination

            coordination = get_workload_coordination()
            acquired, is_reentrant = coordination.acquire_market_workload(
                task_name,
                task_id,
                market=market_value,
            )
            if not acquired:
                return _wait_for_market_workload(task, coordination, market_value)

            # Renew on the reentrant path too: a Celery retry or redelivery
            # reuses the task id, so the "reentrant" lease may be a leftover
            # from a previous attempt with no renewer of its own. Renewing an
            # already-renewed lease is idempotent.
            renewals = [
                (
                    _market_workload_key(market_value),
                    lambda: coordination.renew_market_workload(
                        task_id, market=market_value
                    ),
                )
            ]
            try:
                with keep_leases_alive(renewals) as lease_lost, workload_fence(
                    _market_workload_key(market_value), lease_lost
                ):
                    return func(*args, **kwargs)
            except Retry:
                raise
            except (LeaseNotHeld, LeaseLost):
                # The lease was gone before the body started (a leftover
                # same-id lease that expired and was taken), or was lost
                # mid-run and a fenced commit was rejected (LeaseLost): wait
                # for the new holder like any busy lease.
                return _wait_for_market_workload(task, coordination, market_value)
            except Exception as exc:
                retry_transient_database_error(
                    task,
                    task_name,
                    exc,
                    logger=logger,
                )
                raise
            finally:
                if acquired and not is_reentrant:
                    coordination.release_market_workload(task_id, market=market_value)

        return wrapper

    return decorator
