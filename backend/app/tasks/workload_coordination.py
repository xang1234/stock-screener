"""Redis-backed workload leases for Celery task coordination."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Dict, Iterator, Optional, Tuple

try:
    import redis  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    redis = None

from ..config import settings
from .market_queues import SUPPORTED_MARKETS, market_suffix, normalize_market

EXTERNAL_FETCH_GLOBAL_KEY = "external_fetch_global"
MARKET_WORKLOAD_PREFIX = "market_workload"
COORDINATION_WAIT_MAX_RETRIES = 10_000
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


def _market_workload_key(market: Optional[str]) -> str:
    return f"{MARKET_WORKLOAD_PREFIX}:{market_suffix(market)}"


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
        )
        self.lock_timeout = getattr(settings, "data_fetch_lock_timeout", 7200)
        self._release_script = self.redis.register_script(_RELEASE_LUA)

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

    def get_market_workload_holder(self, market: Optional[str]) -> Optional[Dict[str, Any]]:
        return self._holder(_market_workload_key(market))

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
                holder = coordination.get_market_workload_holder(market_value) or {}
                wait_reason = f"waiting_for_market_workload:{normalize_market(market_value)}"
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

            try:
                return func(*args, **kwargs)
            finally:
                if acquired and not is_reentrant:
                    coordination.release_market_workload(task_id, market=market_value)

        return wrapper

    return decorator
