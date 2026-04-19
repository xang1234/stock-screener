"""
Distributed lock and queue management for data-fetching tasks.

Ensures only one data-fetching job runs at a time to prevent API rate limiting
from yfinance, finviz, and other external data sources.
"""
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from typing import Any, Dict, Iterator, Optional, Tuple

from celery.exceptions import Retry

try:
    import redis  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in desktop packaging
    redis = None

from ..config import settings
from .market_queues import SUPPORTED_MARKETS, market_suffix, normalize_market

logger = logging.getLogger(__name__)

# Celery treats retry(max_retries=None) as "use the task default", which is
# too small for lease-contention waits during startup/bootstrap. Use a large
# explicit ceiling so these coordination waits back off for a long time before
# ever becoming hard failures.
_COORDINATION_WAIT_MAX_RETRIES = 10_000

# Per-market keys take the form "data_fetch_job_lock:<market_lower>" (e.g.
# data_fetch_job_lock:hk); market-agnostic tasks use "data_fetch_job_lock:shared".
# LOCK_KEY (unsuffixed) is retained only for one-shot cleanup of pre-9.1 stale
# locks on worker startup; no runtime path writes to it.
LOCK_KEY = "data_fetch_job_lock"


def _lock_key_for_market(market: Optional[str]) -> str:
    """Return the Redis lock key for a given market, or :shared for None."""
    return f"{LOCK_KEY}:{market_suffix(market)}"


def all_market_lock_keys() -> list[str]:
    """All known per-market lock keys plus the shared key (for worker startup scans)."""
    return [_lock_key_for_market(m) for m in SUPPORTED_MARKETS] + [
        _lock_key_for_market(None)
    ]


_SERIALIZED_DATA_FETCH_LOCK_DISABLED: ContextVar[bool] = ContextVar(
    "serialized_data_fetch_lock_disabled",
    default=False,
)

# Lua script for atomic release: only deletes if the task_id field matches.
# Prevents TOCTOU race where lock TTL expires between GET and DEL.
_RELEASE_LUA = """
local val = redis.call('get', KEYS[1])
if val and string.find(val, ARGV[1], 1, true) then
    return redis.call('del', KEYS[1])
end
return 0
"""

# Lua script for atomic extend: only extends TTL if the task_id field matches.
# Caps TTL at max_ttl (ARGV[3]) to prevent unbounded growth during long tasks.
_EXTEND_LUA = """
local val = redis.call('get', KEYS[1])
if val and string.find(val, ARGV[1], 1, true) then
    local ttl = redis.call('ttl', KEYS[1])
    if ttl > 0 then
        local max_ttl = tonumber(ARGV[3]) or 7200
        local new_ttl = math.min(ttl + tonumber(ARGV[2]), max_ttl)
        redis.call('expire', KEYS[1], new_ttl)
        return new_ttl
    end
end
return -1
"""


def _parse_lock_task_id(lock_value: bytes) -> Optional[str]:
    """Extract the task_id field from a lock value string (task_name:task_id:timestamp)."""
    try:
        parts = lock_value.decode().split(":", 2)
        if len(parts) >= 2:
            return parts[1]
    except Exception:
        pass
    return None


def _build_lock_contention_payload(
    requested_task_name: str,
    current_task: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return a standardized response when another fetch task already holds the lock."""
    current_task = current_task or {}
    holder_name = current_task.get("task_name", "unknown")
    holder_task_id = current_task.get("task_id")
    message = f"Data fetch already in progress ({holder_name})"
    return {
        "status": "already_running",
        "skipped": True,
        "task_name": requested_task_name,
        "running_task_name": holder_name,
        "task_id": holder_task_id,
        "running_task_id": holder_task_id,
        "message": message,
    }


def _retry_coordination_wait(task: Any, message: str) -> None:
    retries = getattr(getattr(task, "request", None), "retries", 0) or 0
    countdown = min(15 * (2 ** retries), 300)
    raise task.retry(
        exc=RuntimeError(message),
        countdown=countdown,
        max_retries=_COORDINATION_WAIT_MAX_RETRIES,
    )


@contextmanager
def disable_serialized_data_fetch_lock():
    """Temporarily bypass the distributed Redis lock for in-process workflows."""
    token = _SERIALIZED_DATA_FETCH_LOCK_DISABLED.set(True)
    try:
        yield
    finally:
        _SERIALIZED_DATA_FETCH_LOCK_DISABLED.reset(token)


class DataFetchLock:
    """
    Redis-based distributed lock for data-fetching tasks.

    Provides visibility into which task currently holds the lock
    and when it started. Works in conjunction with Celery queue
    serialization for defense-in-depth.
    """

    def __init__(self):
        if redis is None:
            raise RuntimeError("Redis package is not installed; DataFetchLock is unavailable")
        self.redis = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db
        )
        self.lock_timeout = getattr(settings, 'data_fetch_lock_timeout', 7200)
        self._release_script = self.redis.register_script(_RELEASE_LUA)
        self._extend_script = self.redis.register_script(_EXTEND_LUA)

    @contextmanager
    def external_hold(
        self,
        task_name: str,
        task_id: str,
        market: Optional[str] = None,
    ) -> Iterator[bool]:
        """Acquire the per-market lock for the duration of the context.

        Releases on exit if (and only if) the acquire actually took the lock.
        Used by chaos/integration tests that need to simulate a stuck task
        holding a market's lock while other code runs.
        """
        acquired, is_reentrant = self.acquire(task_name, task_id, market=market)
        try:
            yield acquired
        finally:
            if acquired and not is_reentrant:
                self.release(task_id, market=market)

    def acquire(
        self,
        task_name: str,
        task_id: str,
        market: Optional[str] = None,
    ) -> Tuple[bool, bool]:
        """
        Try to acquire the lock for the given market scope.

        Args:
            task_name: Name of the task acquiring the lock
            task_id: Celery task ID
            market: Market code (US/HK/JP/TW) or None for the shared key.
                Lock keys are per-market so e.g. a US refresh and HK refresh
                can run in parallel.

        Returns:
            (success, is_reentrant) — success=True if lock acquired or re-entrant,
            is_reentrant=True if the lock was already held by the same task_id.
        """
        key = _lock_key_for_market(market)
        # Re-entrant check: if we already hold the lock, allow through
        if task_id != 'unknown':
            current = self.redis.get(key)
            if current:
                holder_task_id = _parse_lock_task_id(current)
                if holder_task_id and holder_task_id == task_id:
                    logger.info(
                        "Re-entrant lock acquire by %s (task_id=%s, key=%s)",
                        task_name, task_id, key,
                    )
                    return (True, True)

        # Normal acquire
        lock_value = f"{task_name}:{task_id}:{datetime.now().isoformat()}"
        acquired = self.redis.set(
            key,
            lock_value,
            nx=True,  # Only set if not exists
            ex=self.lock_timeout
        )
        if acquired:
            logger.info(
                "Data fetch lock acquired by %s (task_id=%s, key=%s)",
                task_name, task_id, key,
            )
            return (True, False)
        else:
            current = self.get_current_holder(market=market) or {}
            logger.info(
                "Data fetch lock already held by %s (task_id=%s, key=%s). Task %s will wait.",
                current.get("task_name", "unknown"),
                current.get("task_id", "unknown"),
                key,
                task_name,
            )
            return (False, False)

    def release(self, task_id: str, market: Optional[str] = None) -> bool:
        """
        Atomically release the lock for the given market scope if we own it.
        """
        key = _lock_key_for_market(market)
        match_pattern = f":{task_id}:"
        result = self._release_script(keys=[key], args=[match_pattern])
        if result:
            logger.info("Data fetch lock released by task_id=%s (key=%s)", task_id, key)
            return True
        return False

    def force_release(self, market: Optional[str] = None) -> bool:
        """Force release the lock regardless of owner for the given market scope."""
        key = _lock_key_for_market(market)
        result = self.redis.delete(key)
        if result:
            logger.warning("Data fetch lock force released (key=%s)", key)
        return bool(result)

    def get_current_holder(self, market: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get info about the current lock holder for the given market scope."""
        key = _lock_key_for_market(market)
        current = self.redis.get(key)
        if not current:
            return None

        try:
            parts = current.decode().split(':')
            if len(parts) >= 3:
                return {
                    'task_name': parts[0],
                    'task_id': parts[1],
                    'started_at': ':'.join(parts[2:]),  # Rejoin ISO timestamp
                    'ttl_seconds': self.redis.ttl(key),
                    'lock_key': key,
                }
        except Exception as e:
            logger.warning(f"Failed to parse lock value: {e}")

        return {'raw': current.decode(), 'lock_key': key}

    def get_current_task(self, market: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get info about the current running task including progress for the given market scope.

        Enhanced version of get_current_holder that includes heartbeat data.
        Per-market heartbeat key: the `market` arg scopes both lock-holder
        lookup and heartbeat read to the same market.
        """
        holder = self.get_current_holder(market=market)
        if not holder or 'task_name' not in holder:
            return None

        # Per-market heartbeat key via the public constructor on the store.
        try:
            from ..services.price_cache_service import WARMUP_HEARTBEAT_KEY
            from ..services.cache.price_cache_warmup import scoped_heartbeat_key

            heartbeat_key = scoped_heartbeat_key(WARMUP_HEARTBEAT_KEY, market)
            heartbeat_json = self.redis.get(heartbeat_key)
            if heartbeat_json:
                import json
                heartbeat = json.loads(heartbeat_json)
                holder['current'] = heartbeat.get('current')
                holder['total'] = heartbeat.get('total')
                holder['progress'] = heartbeat.get('percent')
                holder['last_heartbeat'] = heartbeat.get('updated_at')
        except Exception as e:
            logger.debug(f"Could not get heartbeat for task: {e}")

        return holder

    def is_locked(self, market: Optional[str] = None) -> bool:
        """Check if lock is currently held for the given market scope."""
        return self.redis.exists(_lock_key_for_market(market)) > 0

    def is_any_locked(self) -> bool:
        """Check if ANY market's lock (or shared) is held. Useful for dashboards."""
        for key in all_market_lock_keys():
            if self.redis.exists(key) > 0:
                return True
        # Also include legacy unsuffixed key defensively.
        return self.redis.exists(LOCK_KEY) > 0

    def get_any_current_holder(self) -> Optional[Dict[str, Any]]:
        """Return holder info from the first market lock that is held.

        Used by status/health API endpoints that need to report ANY active
        data-fetch task across all market scopes, without knowing which
        market is currently active. The legacy unsuffixed key is also checked
        so status remains consistent with ``is_any_locked()`` during a
        rolling upgrade when old tasks may still hold the pre-9.1 key.
        """
        for key in [*all_market_lock_keys(), LOCK_KEY]:
            current = self.redis.get(key)
            if not current:
                continue
            try:
                parts = current.decode().split(':')
                if len(parts) >= 3:
                    return {
                        'task_name': parts[0],
                        'task_id': parts[1],
                        'started_at': ':'.join(parts[2:]),
                        'ttl_seconds': self.redis.ttl(key),
                        'lock_key': key,
                    }
            except Exception:
                return {'raw': current.decode(), 'lock_key': key}
        return None

    def get_any_current_task(self) -> Optional[Dict[str, Any]]:
        """Return the current task (with heartbeat) from any held market lock.

        Companion to ``get_any_current_holder``, augmented with heartbeat
        progress data from the matching market-scoped heartbeat key.
        """
        holder = self.get_any_current_holder()
        if not holder or 'task_name' not in holder:
            return None

        try:
            from ..services.price_cache_service import WARMUP_HEARTBEAT_KEY
            from ..services.cache.price_cache_warmup import scoped_heartbeat_key
            # Derive market from the lock key: "data_fetch_job_lock:hk" → "hk"
            # Unsuffixed legacy LOCK_KEY is treated as shared scope.
            lock_key = holder.get('lock_key', '')
            suffix = lock_key.rsplit(':', 1)[-1] if ':' in lock_key else ''
            market_str = None if lock_key == LOCK_KEY or suffix in ('shared', '') else suffix
            heartbeat_key = scoped_heartbeat_key(WARMUP_HEARTBEAT_KEY, market_str)
            heartbeat_json = self.redis.get(heartbeat_key)
            if heartbeat_json:
                import json
                heartbeat = json.loads(heartbeat_json)
                holder['current'] = heartbeat.get('current')
                holder['total'] = heartbeat.get('total')
                holder['progress'] = heartbeat.get('percent')
                holder['last_heartbeat'] = heartbeat.get('updated_at')
        except Exception as e:
            logger.debug("Could not get heartbeat in get_any_current_task: %s", e)

        return holder

    def force_release_all(self) -> int:
        """Force-release ALL market and shared lock keys.

        Used by admin force-release endpoints to unstick any market lock
        regardless of which market is currently held. Returns the number of
        keys deleted.
        """
        count = 0
        for key in all_market_lock_keys():
            if self.redis.delete(key):
                count += 1
                logger.warning("Data fetch lock force released (key=%s)", key)
        # Also clear legacy unsuffixed key.
        if self.redis.delete(LOCK_KEY):
            count += 1
            logger.warning("Legacy data fetch lock force released (key=%s)", LOCK_KEY)
        return count

    def extend_lock(
        self,
        task_id: str,
        additional_seconds: int = 300,
        max_ttl: int = 7200,
        market: Optional[str] = None,
    ) -> bool:
        """Atomically extend the lock timeout if we own it, for the given market scope."""
        key = _lock_key_for_market(market)
        match_pattern = f":{task_id}:"
        new_ttl = self._extend_script(
            keys=[key], args=[match_pattern, additional_seconds, max_ttl]
        )
        if new_ttl > 0:
            logger.info(
                "Data fetch lock extended by %ss (new TTL: %ss, cap: %ss, key=%s)",
                additional_seconds, new_ttl, max_ttl, key,
            )
            return True
        return False


def serialized_data_fetch(task_name: str):
    """Decorator for tasks that fetch external data.

    Acquires a per-market distributed lock before running. The market is
    pulled from the task's ``market`` kwarg; tasks without it lock on the
    shared key. This enables cross-market parallelism (US refresh can run
    while HK refresh is in-flight) while preventing intra-market contention.
    Supports re-entrant calls by task_id.

    Example:
        @celery_app.task(bind=True, name='app.tasks.cache_tasks.smart_refresh_cache')
        @serialized_data_fetch('smart_refresh_cache')
        def smart_refresh_cache(self, mode="auto", market=None):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            market_value: Optional[str] = kwargs.get("market")
            market_label = normalize_market(market_value).lower()

            if _SERIALIZED_DATA_FETCH_LOCK_DISABLED.get():
                logger.info(
                    "Bypassing distributed data-fetch lock for %s (market=%s)",
                    task_name, market_label,
                )
                return func(*args, **kwargs)

            from ..wiring.bootstrap import get_data_fetch_lock, get_workload_coordination

            lock = get_data_fetch_lock()
            coordination = get_workload_coordination()

            task_id = 'unknown'
            if args and hasattr(args[0], 'request'):
                task_id = args[0].request.id or 'unknown'
            task_instance = args[0] if args and hasattr(args[0], "request") else None

            acquired, is_reentrant = lock.acquire(task_name, task_id, market=market_value)
            if not acquired:
                current = lock.get_current_task(market=market_value) or lock.get_current_holder(
                    market=market_value
                )
                logger.warning(
                    "Data fetch lock held by %s (task_id=%s, market=%s) — skipping duplicate task %s.",
                    current.get("task_name", "unknown") if current else "unknown",
                    current.get("task_id", "unknown") if current else "unknown",
                    market_label,
                    task_name,
                )
                payload = _build_lock_contention_payload(task_name, current)
                payload["market"] = market_label
                return payload

            workload_acquired = False
            workload_reentrant = False
            external_acquired = False
            external_reentrant = False

            try:
                workload_acquired, workload_reentrant = coordination.acquire_market_workload(
                    task_name,
                    task_id,
                    market=market_value,
                )
                if not workload_acquired:
                    current = coordination.get_market_workload_holder(market_value) or {}
                    message = (
                        f"waiting_for_market_workload:{normalize_market(market_value)} "
                        f"({current.get('task_name', 'unknown')})"
                    )
                    if task_instance is not None and hasattr(task_instance, "retry"):
                        _retry_coordination_wait(task_instance, message)
                    return {
                        "status": "waiting",
                        "wait_reason": f"waiting_for_market_workload:{normalize_market(market_value)}",
                        "running_task_name": current.get("task_name"),
                        "running_task_id": current.get("task_id"),
                    }

                external_acquired, external_reentrant = coordination.acquire_external_fetch(
                    task_name,
                    task_id,
                )
                if not external_acquired:
                    current = coordination.get_external_fetch_holder() or {}
                    message = (
                        "waiting_for_external_fetch_global "
                        f"({current.get('task_name', 'unknown')})"
                    )
                    if task_instance is not None and hasattr(task_instance, "retry"):
                        _retry_coordination_wait(task_instance, message)
                    return {
                        "status": "waiting",
                        "wait_reason": "waiting_for_external_fetch_global",
                        "running_task_name": current.get("task_name"),
                        "running_task_id": current.get("task_id"),
                    }

                logger.info(
                    "Starting data fetch task: %s (task_id=%s, market=%s)",
                    task_name, task_id, market_label,
                )
                start_time = datetime.now()

                result = func(*args, **kwargs)

                duration = (datetime.now() - start_time).total_seconds()
                logger.info(
                    "Completed data fetch task: %s in %.2fs (market=%s)",
                    task_name, duration, market_label,
                )
                return result
            except Retry:
                raise
            except Exception as e:
                logger.error(
                    "Error in data fetch task %s (market=%s): %s",
                    task_name, market_label, e,
                    exc_info=True,
                )
                raise
            finally:
                if external_acquired and not external_reentrant:
                    coordination.release_external_fetch(task_id)
                if workload_acquired and not workload_reentrant:
                    coordination.release_market_workload(task_id, market=market_value)
                if acquired and not is_reentrant:
                    lock.release(task_id, market=market_value)

        return wrapper
    return decorator
