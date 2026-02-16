"""
Distributed lock and queue management for data-fetching tasks.

Ensures only one data-fetching job runs at a time to prevent API rate limiting
from yfinance, finviz, and other external data sources.
"""
import redis
import logging
import time
from functools import wraps
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

from ..config import settings

logger = logging.getLogger(__name__)

LOCK_KEY = "data_fetch_job_lock"

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
_EXTEND_LUA = """
local val = redis.call('get', KEYS[1])
if val and string.find(val, ARGV[1], 1, true) then
    local ttl = redis.call('ttl', KEYS[1])
    if ttl > 0 then
        local new_ttl = ttl + tonumber(ARGV[2])
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


class DataFetchLock:
    """
    Redis-based distributed lock for data-fetching tasks.

    Provides visibility into which task currently holds the lock
    and when it started. Works in conjunction with Celery queue
    serialization for defense-in-depth.
    """

    _instance = None

    def __init__(self):
        self.redis = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db
        )
        self.lock_timeout = getattr(settings, 'data_fetch_lock_timeout', 7200)
        self._release_script = self.redis.register_script(_RELEASE_LUA)
        self._extend_script = self.redis.register_script(_EXTEND_LUA)

    @classmethod
    def get_instance(cls) -> 'DataFetchLock':
        """Get singleton instance of DataFetchLock."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def acquire(self, task_name: str, task_id: str) -> Tuple[bool, bool]:
        """
        Try to acquire the lock.

        Args:
            task_name: Name of the task acquiring the lock
            task_id: Celery task ID

        Returns:
            (success, is_reentrant) — success=True if lock acquired or re-entrant,
            is_reentrant=True if the lock was already held by the same task_id.
        """
        # Re-entrant check: if we already hold the lock, allow through
        if task_id != 'unknown':
            current = self.redis.get(LOCK_KEY)
            if current:
                holder_task_id = _parse_lock_task_id(current)
                if holder_task_id and holder_task_id == task_id:
                    logger.info(f"Re-entrant lock acquire by {task_name} (task_id={task_id})")
                    return (True, True)

        # Normal acquire
        lock_value = f"{task_name}:{task_id}:{datetime.now().isoformat()}"
        acquired = self.redis.set(
            LOCK_KEY,
            lock_value,
            nx=True,  # Only set if not exists
            ex=self.lock_timeout
        )
        if acquired:
            logger.info(f"Data fetch lock acquired by {task_name} (task_id={task_id})")
            return (True, False)
        else:
            current = self.get_current_holder() or {}
            logger.info(
                f"Data fetch lock already held by {current.get('task_name', 'unknown')} "
                f"(task_id={current.get('task_id', 'unknown')}). "
                f"Task {task_name} will wait in queue."
            )
            return (False, False)

    def release(self, task_id: str) -> bool:
        """
        Atomically release the lock if we own it.

        Uses a Lua script to check ownership and delete in one atomic
        operation, preventing TOCTOU races.

        Args:
            task_id: Celery task ID that should own the lock

        Returns:
            True if lock was released, False if we didn't own it
        """
        # Lua script matches ":task_id:" to ensure exact field match
        match_pattern = f":{task_id}:"
        result = self._release_script(keys=[LOCK_KEY], args=[match_pattern])
        if result:
            logger.info(f"Data fetch lock released by task_id={task_id}")
            return True
        return False

    def force_release(self) -> bool:
        """
        Force release the lock regardless of owner.
        Use with caution - only for stuck locks.

        Returns:
            True if lock was deleted, False if no lock existed
        """
        result = self.redis.delete(LOCK_KEY)
        if result:
            logger.warning("Data fetch lock force released")
        return bool(result)

    def get_current_holder(self) -> Optional[Dict[str, Any]]:
        """
        Get info about the current lock holder.

        Returns:
            Dict with task_name, task_id, started_at, ttl_seconds
            or None if no lock is held
        """
        current = self.redis.get(LOCK_KEY)
        if not current:
            return None

        try:
            parts = current.decode().split(':')
            if len(parts) >= 3:
                return {
                    'task_name': parts[0],
                    'task_id': parts[1],
                    'started_at': ':'.join(parts[2:]),  # Rejoin ISO timestamp
                    'ttl_seconds': self.redis.ttl(LOCK_KEY)
                }
        except Exception as e:
            logger.warning(f"Failed to parse lock value: {e}")

        return {'raw': current.decode()}

    def get_current_task(self) -> Optional[Dict[str, Any]]:
        """
        Get info about the current running task including progress.

        Enhanced version of get_current_holder that includes heartbeat data.

        Returns:
            Dict with task_name, task_id, started_at, ttl_seconds, and progress info
            or None if no task is running
        """
        holder = self.get_current_holder()
        if not holder or 'task_name' not in holder:
            return None

        # Try to get heartbeat/progress info
        try:
            from ..services.price_cache_service import WARMUP_HEARTBEAT_KEY
            heartbeat_json = self.redis.get(WARMUP_HEARTBEAT_KEY)
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

    def is_locked(self) -> bool:
        """Check if lock is currently held."""
        return self.redis.exists(LOCK_KEY) > 0

    def extend_lock(self, task_id: str, additional_seconds: int = 3600) -> bool:
        """
        Atomically extend the lock timeout if we own it.

        Uses a Lua script to check ownership and extend TTL in one
        atomic operation.

        Args:
            task_id: Celery task ID that should own the lock
            additional_seconds: Seconds to add to current TTL

        Returns:
            True if lock was extended, False if we didn't own it
        """
        match_pattern = f":{task_id}:"
        new_ttl = self._extend_script(
            keys=[LOCK_KEY], args=[match_pattern, additional_seconds]
        )
        if new_ttl > 0:
            logger.info(f"Data fetch lock extended by {additional_seconds}s (new TTL: {new_ttl}s)")
            return True
        return False


def serialized_data_fetch(task_name: str):
    """
    Decorator for tasks that fetch external data.

    Acquires a global lock before running and releases it after completion.
    Combined with Celery queue routing (concurrency=1), this ensures
    only one data-fetching task runs at a time.

    The lock provides visibility into which task is currently running,
    while the queue handles actual serialization.

    Supports re-entrant calls: if the same task_id already holds the lock,
    the decorator allows through without blocking and skips release on exit.

    Args:
        task_name: Human-readable name for the task (for logging/status)

    Example:
        @celery_app.task(bind=True, name='app.tasks.cache_tasks.daily_cache_warmup')
        @serialized_data_fetch('daily_cache_warmup')
        def daily_cache_warmup(self):
            # task code here...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            lock = DataFetchLock.get_instance()

            # Get task ID from Celery request if available
            task_id = 'unknown'
            if args and hasattr(args[0], 'request'):
                task_id = args[0].request.id or 'unknown'

            # Acquire lock (for visibility - queue handles actual serialization)
            acquired, is_reentrant = lock.acquire(task_name, task_id)
            if not acquired:
                wait_seconds = getattr(settings, "data_fetch_lock_wait_seconds", lock.lock_timeout)
                poll_interval = 5
                start_time = datetime.now()
                logger.info(
                    f"Waiting for data fetch lock (task={task_name}, task_id={task_id}) "
                    f"up to {wait_seconds}s"
                )
                while not acquired:
                    time.sleep(poll_interval)
                    acquired, is_reentrant = lock.acquire(task_name, task_id)
                    if (datetime.now() - start_time).total_seconds() > wait_seconds:
                        logger.error(
                            f"Timed out waiting for data fetch lock (task={task_name}, task_id={task_id})"
                        )
                        raise RuntimeError("Timed out waiting for data fetch lock")

            try:
                logger.info(f"Starting data fetch task: {task_name} (task_id={task_id})")
                start_time = datetime.now()

                result = func(*args, **kwargs)

                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"Completed data fetch task: {task_name} in {duration:.2f}s")

                return result
            except Exception as e:
                logger.error(f"Error in data fetch task {task_name}: {e}", exc_info=True)
                raise
            finally:
                # Only release if we actually acquired (not re-entrant)
                if acquired and not is_reentrant:
                    lock.release(task_id)

        return wrapper
    return decorator
