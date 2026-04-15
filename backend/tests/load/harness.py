"""Shared harness helpers for the load + chaos test suites (beads 9.3, 9.4).

Extracted from ``test_per_market_load.py`` so the fault-injection isolation
tests in ``test_failure_isolation.py`` can reuse the same parallel runner,
yf monkey-patch wiring, and Redis counter accessor without duplication.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
from unittest.mock import patch

from .measurement import ResourceSampler
from .yfinance_simulator import MultiMarketSimulator


def no_sleep(_seconds: float) -> None:
    """Drop-in for ``time.sleep`` that doesn't actually wait.

    Patched into ``BulkDataFetcher`` and ``RedisRateLimiter`` so adaptive
    backoffs (60-480s) and inter-batch waits don't make the harness
    multi-minute. Per-batch latency is recorded in each sub-simulator;
    tail-latency reflects requested sleep, not actual wait.
    """
    return None


def make_simulator_module(simulator):
    """Wrap a simulator in a module-shaped namespace with ``.download``.

    Used to monkey-patch ``app.services.bulk_data_fetcher.yf``. The
    ``Tickers`` attribute raises explicitly if any future code path
    accidentally pulls fundamentals through the load harness — silent
    AttributeError or NoneType-not-callable would be much harder to
    diagnose than a typed message.
    """
    def _tickers_unsupported(*args, **kwargs):
        raise NotImplementedError(
            "yf.Tickers is not supported by the load harness simulator. "
            "Add a fundamentals scenario before exercising this path."
        )

    class _Mod:
        download = staticmethod(simulator.download)
        Tickers = staticmethod(_tickers_unsupported)
    return _Mod


def read_429_counter(market: str) -> int:
    """Read the per-market×provider 429 counter populated by RateBudgetPolicy.

    Delegates key construction to ``RateBudgetPolicy.counter_key_429`` so the
    harness stays in sync with the policy's write side.
    """
    try:
        from app.services.redis_pool import get_redis_client
        from app.services.rate_budget_policy import get_rate_budget_policy
        client = get_redis_client()
        if client is None:
            return 0
        val = client.get(get_rate_budget_policy().counter_key_429("yfinance", market))
        return int(val) if val else 0
    except Exception:
        return 0


def run_one_market(
    market: str,
    symbols: List[str],
    sampler: ResourceSampler,
) -> Tuple[str, float, int, int]:
    """Run a single market's refresh workload using the real fetcher path.

    Returns ``(market, wall_clock_s, symbols_processed, transient_failures)``.
    The yfinance module is expected to be patched by the caller before this
    runs (see ``run_parallel_refresh``).
    """
    from app.services.bulk_data_fetcher import BulkDataFetcher

    fetcher = BulkDataFetcher()
    start = time.monotonic()
    sampler.sample()

    try:
        results = fetcher.fetch_prices_in_batches(symbols, period="2y", market=market)
    except Exception as exc:
        raise RuntimeError(f"fetch_prices_in_batches({market}) crashed: {exc}") from exc

    sampler.sample()
    wall_clock_s = time.monotonic() - start
    transient_failures = sum(
        1 for v in results.values() if isinstance(v, dict) and v.get("has_error")
    )
    return (market, wall_clock_s, len(results), transient_failures)


def run_parallel_refresh(
    multi_sim: MultiMarketSimulator,
    universe_per_market: Dict[str, List[str]],
    sampler: ResourceSampler,
) -> List[Tuple[str, float, int, int]]:
    """Spawn one worker per market and run all in parallel against the simulator.

    Patches ``yf`` once with the multi-market simulator (thread-safe) and
    no-ops ``time.sleep`` in the fetcher + rate-limiter modules. Returns the
    per-market result tuples in completion order; callers typically re-sort
    for stable presentation.
    """
    from app.services import bulk_data_fetcher as bdf_module
    from app.services import rate_limiter as rl_module

    results: List[Tuple[str, float, int, int]] = []
    yf_mod = make_simulator_module(multi_sim)

    with patch.object(bdf_module, "yf", yf_mod), \
            patch.object(bdf_module.time, "sleep", no_sleep), \
            patch.object(rl_module.time, "sleep", no_sleep):
        with ThreadPoolExecutor(max_workers=len(universe_per_market)) as pool:
            futures = {
                pool.submit(run_one_market, market, symbols, sampler): market
                for market, symbols in universe_per_market.items()
            }
            for future in as_completed(futures):
                results.append(future.result())

    return results
