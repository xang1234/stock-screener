"""Fixtures for the per-market load/soak harness (bead asia.9.3).

Provides:
- A fixed synthetic universe (200/100/100/80 symbols per US/HK/JP/TW) so
  benchmark numbers don't drift with live universe state.
- The deterministic YFinanceSimulator, with monkeypatch wiring into
  ``BulkDataFetcher`` so the real fetch path runs against the mock.
- A Redis-availability check that skips the load suite when Redis isn't
  running locally (CI provides Redis via a service container).
- A reset hook that clears all rate-limiter and 429-counter Redis keys
  before each run to make measurements independent.
"""
from __future__ import annotations

import os
from typing import Dict, List

import pytest


# Synthetic universe sizes — small enough to run in seconds, large enough
# to exercise per-market batch sizing differences (US 50, HK 25, JP 25, TW 20).
# Multiples of the per-market batch size so the loop body runs ~4-5 batches
# per market, enough to exercise backoff/throttle paths without making the
# test slow.
SYNTHETIC_UNIVERSE_SIZES: Dict[str, int] = {
    "US": 200,   # 4 batches x 50
    "HK": 100,   # 4 batches x 25
    "JP": 100,   # 4 batches x 25
    "TW": 80,    # 4 batches x 20
}


def _redis_available() -> bool:
    """Check that a Redis instance is reachable via the shared pool."""
    from redis.exceptions import RedisError
    from app.services.redis_pool import get_redis_client
    try:
        client = get_redis_client()
        if client is None:
            return False
        client.ping()
        return True
    except RedisError:
        return False


@pytest.fixture(scope="session")
def redis_required():
    """Skip load tests when Redis is unavailable.

    The harness requires real Redis for the rate-limiter atomic ops and
    429 counters. Mocking would defeat the purpose — the load test exists
    to exercise actual Redis-backed behavior under concurrency.
    """
    if not _redis_available():
        pytest.skip(
            "Load tests require a running Redis instance "
            "(set REDIS_HOST/REDIS_PORT or run a local redis-server)."
        )


@pytest.fixture(scope="session")
def synthetic_universe() -> Dict[str, List[str]]:
    """Return a stable per-market symbol list for benchmark reproducibility."""
    universe: Dict[str, List[str]] = {}
    for market, size in SYNTHETIC_UNIVERSE_SIZES.items():
        # Deterministic, market-prefixed symbols (LOAD_US_0001 etc).
        universe[market] = [f"LOAD_{market}_{i:04d}" for i in range(size)]
    return universe


@pytest.fixture
def reset_redis_state(redis_required):
    """Clear load-harness-owned Redis keys before/after each run.

    Ensures back-to-back load runs don't pollute each other's measurements.
    Cleanup is scoped to the exact per-market suffixes (us/hk/jp/tw) used by
    the synthetic universe so it never touches unrelated keys on a shared Redis
    instance (e.g., a staging server running live workers).

    Cleanup spans both Redis DBs the rate-budget plumbing uses:
    - DB 0 (``settings.redis_db``): exact per-market lock keys.
    - DB 2 (``settings.cache_redis_db``): ratelimit/429/throttle/warmup keys
      scoped to each load-test market suffix.
    Uses pipelined ``UNLINK`` (lazy delete) to avoid a per-key DEL round-trip.
    """
    import redis
    from app.config import settings
    from app.services.redis_pool import get_redis_client

    cache_client = get_redis_client()
    # Lock keys live in DB 0; redis_pool only exposes the cache DB so the
    # lock-DB client must be constructed directly here.
    lock_client = redis.Redis(
        host=settings.redis_host, port=settings.redis_port, db=settings.redis_db,
    )

    # Restrict cleanup to the exact market suffixes the load harness uses so we
    # never wipe unrelated state on a shared Redis instance (e.g., a staging
    # server where the same DB is used by live workers).
    _LOAD_MARKET_SUFFIXES = [m.lower() for m in SYNTHETIC_UNIVERSE_SIZES]

    def _clear():
        if cache_client is None:
            return
        pipe_cache = cache_client.pipeline()
        pipe_lock = lock_client.pipeline()
        for suffix in _LOAD_MARKET_SUFFIXES:
            # Exact lock key per market (DB 0).
            pipe_lock.unlink(f"data_fetch_job_lock:{suffix}")
            # Rate-limiter + 429 + throttle keys scoped to this market (DB 2).
            for key in cache_client.scan_iter(match=f"ratelimit:*:{suffix}*"):
                pipe_cache.unlink(key)
            # Warmup heartbeat key scoped to this market (DB 2).
            for key in cache_client.scan_iter(match=f"cache:warmup:*:{suffix}*"):
                pipe_cache.unlink(key)
        pipe_cache.execute()
        pipe_lock.execute()

    _clear()
    yield
    _clear()


@pytest.fixture
def simulator_seed() -> int:
    """Default RNG seed for the YFinanceSimulator. Override via env for
    intentional re-runs with different randomness."""
    return int(os.getenv("LOAD_TEST_SEED", "42"))


@pytest.fixture
def live_yfinance() -> bool:
    """When True, the harness uses the real yfinance instead of the simulator.

    Off by default for CI/dev. Set ``LOAD_TEST_LIVE=1`` for the occasional
    out-of-band evidence-pack run that produces real-traffic numbers.
    """
    return os.getenv("LOAD_TEST_LIVE", "").lower() in ("1", "true", "yes")
