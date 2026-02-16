"""
Shared Redis Connection Pool for all cache services.

Provides a singleton connection pool that all cache services share,
reducing connection overhead and preventing connection exhaustion.

Benefits:
- Single pool with max_connections limit (default: 10)
- Automatic connection reuse across services
- Consistent timeout settings
- Thread-safe singleton pattern
"""
import logging
from typing import Optional
import redis
from redis import ConnectionPool, Redis

from ..config import settings

logger = logging.getLogger(__name__)

# Module-level pool instances (singletons)
_pool: Optional[ConnectionPool] = None
_bulk_pool: Optional[ConnectionPool] = None


def get_redis_pool() -> Optional[ConnectionPool]:
    """
    Get the shared Redis connection pool (singleton).

    Creates the pool on first call with settings from config.
    Returns None if Redis connection fails.

    Pool configuration:
    - max_connections: 10 (sufficient for typical worker count)
    - socket_connect_timeout: 2s (fast fail on connection issues)
    - socket_timeout: 2s (prevent long-running operations from blocking)
    """
    global _pool

    if _pool is not None:
        return _pool

    try:
        _pool = ConnectionPool(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.cache_redis_db,  # DB 2 for cache
            max_connections=10,
            socket_connect_timeout=2,
            socket_timeout=2,
            decode_responses=False  # Binary mode for pickle
        )

        # Verify pool works by getting a test connection
        test_client = Redis(connection_pool=_pool)
        test_client.ping()

        logger.info(
            f"Redis connection pool initialized: "
            f"{settings.redis_host}:{settings.redis_port}/db{settings.cache_redis_db} "
            f"(max_connections=10)"
        )
        return _pool

    except Exception as e:
        logger.warning(f"Failed to create Redis connection pool: {e}")
        _pool = None
        return None


def get_bulk_redis_pool() -> Optional[ConnectionPool]:
    """
    Get a separate Redis connection pool with longer timeouts for bulk pipeline operations.

    Normal operations use get_redis_pool() (2s timeout) for fast-fail behavior.
    Bulk pipelines (e.g., fetching 7000+ symbols) transfer multi-GB responses
    and need a longer timeout to complete without false failures.

    Returns None if Redis connection fails.
    """
    global _bulk_pool

    if _bulk_pool is not None:
        return _bulk_pool

    try:
        bulk_timeout = getattr(settings, 'redis_bulk_socket_timeout', 30)
        _bulk_pool = ConnectionPool(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.cache_redis_db,
            max_connections=5,
            socket_connect_timeout=5,
            socket_timeout=bulk_timeout,
            decode_responses=False
        )

        test_client = Redis(connection_pool=_bulk_pool)
        test_client.ping()

        logger.info(
            f"Redis bulk connection pool initialized: "
            f"{settings.redis_host}:{settings.redis_port}/db{settings.cache_redis_db} "
            f"(max_connections=5, socket_timeout={bulk_timeout}s)"
        )
        return _bulk_pool

    except Exception as e:
        logger.warning(f"Failed to create Redis bulk connection pool: {e}")
        _bulk_pool = None
        return None


def get_bulk_redis_client() -> Optional[Redis]:
    """
    Get a Redis client from the bulk connection pool (longer timeouts).

    Use this for bulk pipeline operations (e.g., get_many with thousands of keys).
    Falls back to the normal pool if the bulk pool is unavailable.
    """
    pool = get_bulk_redis_pool()

    if pool is None:
        logger.debug("Bulk Redis pool unavailable, falling back to normal pool")
        return get_redis_client()

    try:
        return Redis(connection_pool=pool, decode_responses=False)
    except Exception as e:
        logger.warning(f"Failed to create bulk Redis client, falling back to normal pool: {e}")
        return get_redis_client()


def get_redis_client() -> Optional[Redis]:
    """
    Get a Redis client using the shared connection pool.

    Returns:
        Redis client if pool is available, None otherwise

    Usage:
        client = get_redis_client()
        if client:
            client.set("key", "value")
    """
    pool = get_redis_pool()

    if pool is None:
        return None

    try:
        return Redis(connection_pool=pool, decode_responses=False)
    except Exception as e:
        logger.warning(f"Failed to create Redis client from pool: {e}")
        return None


def reset_pool() -> None:
    """
    Reset all connection pools (for testing or reconnection).

    This closes all connections in both pools and clears the singletons.
    The next call to get_redis_pool()/get_bulk_redis_pool() will create fresh pools.
    """
    global _pool, _bulk_pool

    if _pool is not None:
        try:
            _pool.disconnect()
            logger.info("Redis connection pool disconnected")
        except Exception as e:
            logger.warning(f"Error disconnecting pool: {e}")
        finally:
            _pool = None

    if _bulk_pool is not None:
        try:
            _bulk_pool.disconnect()
            logger.info("Redis bulk connection pool disconnected")
        except Exception as e:
            logger.warning(f"Error disconnecting bulk pool: {e}")
        finally:
            _bulk_pool = None
