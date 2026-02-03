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
from typing import Optional, Dict, Any

from ..config import settings

logger = logging.getLogger(__name__)

LOCK_KEY = "data_fetch_job_lock"


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

    @classmethod
    def get_instance(cls) -> 'DataFetchLock':
        """Get singleton instance of DataFetchLock."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def acquire(self, task_name: str, task_id: str) -> bool:
        """
        Try to acquire the lock.

        Args:
            task_name: Name of the task acquiring the lock
            task_id: Celery task ID

        Returns:
            True if lock was acquired, False if already held by another task
        """
        lock_value = f"{task_name}:{task_id}:{datetime.now().isoformat()}"
        acquired = self.redis.set(
            LOCK_KEY,
            lock_value,
            nx=True,  # Only set if not exists
            ex=self.lock_timeout
        )
        if acquired:
            logger.info(f"Data fetch lock acquired by {task_name} (task_id={task_id})")
        else:
            current = self.get_current_holder()
            logger.info(
                f"Data fetch lock already held by {current.get('task_name', 'unknown')} "
                f"(task_id={current.get('task_id', 'unknown')}). "
                f"Task {task_name} will wait in queue."
            )
        return bool(acquired)

    def release(self, task_id: str) -> bool:
        """
        Release the lock if we own it.

        Args:
            task_id: Celery task ID that should own the lock

        Returns:
            True if lock was released, False if we didn't own it
        """
        current = self.redis.get(LOCK_KEY)
        if current and task_id in current.decode():
            self.redis.delete(LOCK_KEY)
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

    def is_locked(self) -> bool:
        """Check if lock is currently held."""
        return self.redis.exists(LOCK_KEY) > 0

    def extend_lock(self, task_id: str, additional_seconds: int = 3600) -> bool:
        """
        Extend the lock timeout if we own it.

        Args:
            task_id: Celery task ID that should own the lock
            additional_seconds: Seconds to add to current TTL

        Returns:
            True if lock was extended, False if we didn't own it
        """
        current = self.redis.get(LOCK_KEY)
        if current and task_id in current.decode():
            current_ttl = self.redis.ttl(LOCK_KEY)
            if current_ttl > 0:
                new_ttl = current_ttl + additional_seconds
                self.redis.expire(LOCK_KEY, new_ttl)
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
            acquired = lock.acquire(task_name, task_id)
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
                    acquired = lock.acquire(task_name, task_id)
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
                if acquired:
                    lock.release(task_id)

        return wrapper
    return decorator
