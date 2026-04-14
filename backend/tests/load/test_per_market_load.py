"""Per-market parallel-refresh load test (bead asia.9.3).

Simulates the production scenario where 4 per-market beat workers fire
``smart_refresh_cache`` for US/HK/JP/TW simultaneously during the Sunday
weekly refresh window. Exercises the real RateBudgetPolicy +
RedisRateLimiter + per-market lock plumbing from 9.1+9.2 against a
deterministic yfinance simulator.

Outputs a JSON snapshot under ``tests/load/baselines/`` and compares against
the committed baseline. Fails the test (and CI gate) when wall-clock or 429
counts regress beyond thresholds.

Run normally:
    pytest tests/load/test_per_market_load.py -v -m load

Update the committed baseline (only when intentional):
    LOAD_TEST_UPDATE_BASELINE=1 pytest tests/load/test_per_market_load.py -v -m load

Run against real yfinance (out-of-band):
    LOAD_TEST_LIVE=1 pytest tests/load/test_per_market_load.py -v -m load
"""
from __future__ import annotations

import os
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

import pytest

from app.services.bulk_data_fetcher import BulkDataFetcher
from app.services.rate_budget_policy import get_rate_budget_policy
from app.tasks.market_queues import SUPPORTED_MARKETS

from .measurement import (
    LoadRunSnapshot,
    MarketMetrics,
    ResourceSampler,
    compare_to_baseline,
    read_snapshot,
    write_snapshot,
)
from .yfinance_simulator import build_multi_market_simulator


def _no_sleep(_seconds: float) -> None:
    """Drop-in for ``time.sleep`` that doesn't actually wait.

    Patched into ``BulkDataFetcher`` and the simulator so the load harness
    measures *logical* pipeline behavior (call counts, lock contention,
    backoff branch coverage) rather than literal wall-clock waits. The
    fetcher's backoff path can ask for 60-480s sleeps; honoring those
    would make the test useless for CI.
    """
    return None


pytestmark = pytest.mark.load


BASELINE_PATH = Path(__file__).parent / "baselines" / "per_market_load.json"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _run_one_market(
    market: str,
    symbols: List[str],
    sampler: ResourceSampler,
) -> tuple[str, float, int, int]:
    """Run a single market's refresh workload using the real fetcher path.

    The yfinance module is already patched globally to a thread-safe
    multi-market simulator before this function is called, so all 4 markets
    run concurrently against the same patched module without race conditions.
    Returns (market, wall_clock_s, symbols_processed, transient_failures).
    """
    fetcher = BulkDataFetcher()

    start = time.monotonic()
    sampler.sample()

    try:
        results = fetcher.fetch_prices_in_batches(
            symbols,
            period="2y",
            market=market,
        )
    except Exception as exc:
        raise RuntimeError(f"fetch_prices_in_batches({market}) crashed: {exc}") from exc

    sampler.sample()
    wall_clock_s = time.monotonic() - start
    transient_failures = sum(
        1 for v in results.values() if isinstance(v, dict) and v.get("has_error")
    )
    return (market, wall_clock_s, len(results), transient_failures)


def _make_simulator_module(simulator):
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


def _read_429_counter(market: str) -> int:
    """Read the per-market×provider 429 counter populated by RateBudgetPolicy.

    Delegates key construction to ``RateBudgetPolicy.counter_key_429`` so the
    harness stays in sync with the policy's write side automatically.
    """
    try:
        from app.services.redis_pool import get_redis_client
        client = get_redis_client()
        if client is None:
            return 0
        val = client.get(get_rate_budget_policy().counter_key_429("yfinance", market))
        return int(val) if val else 0
    except Exception:
        return 0


@pytest.mark.load
def test_parallel_per_market_weekly_refresh(
    redis_required,
    reset_redis_state,
    synthetic_universe,
    simulator_seed,
    live_yfinance,
):
    """Run all 4 markets in parallel, snapshot metrics, regression-gate vs baseline.

    This is the canonical 9.3 load scenario: it mirrors the Sunday weekly
    refresh window when ``weekly-full-refresh-{us,hk,jp,tw}`` beat entries
    fire simultaneously. The test:

    1. Spins up one ThreadPoolExecutor worker per market.
    2. Each worker runs ``BulkDataFetcher.fetch_prices_in_batches(market=...)``
       against a deterministic yfinance simulator.
    3. The real per-market RateBudgetPolicy + RedisRateLimiter + lock
       partitioning from 9.1+9.2 governs concurrency.
    4. After all workers complete, snapshots wall-clock, 429 count,
       tail-latency, and resource samples to JSON.
    5. Compares against the committed baseline. Fails the test if any
       gated metric regresses beyond its threshold.
    """
    if live_yfinance:
        pytest.skip(
            "live_yfinance mode not implemented in this test (use the dedicated "
            "manual evidence-pack runner; see tests/load/README.md)."
        )

    # Force a fresh weights cache so the rate budget reflects the synthetic
    # universe sizes, not whatever the in-process cache had from prior tests.
    policy = get_rate_budget_policy()
    policy.invalidate_weights_cache()

    multi_sim = build_multi_market_simulator(seed=simulator_seed, sleep_fn=_no_sleep)
    sampler = ResourceSampler(interval_s=0.5)
    market_results: List[tuple[str, float, int, int]] = []

    # Patch yf module ONCE (thread-safe via MultiMarketSimulator) and patch
    # time.sleep across bulk_data_fetcher + rate_limiter so adaptive backoffs
    # (60-480s) and inter-batch waits don't make the harness multi-minute.
    # Per-batch latency is recorded in each sub-simulator; tail-latency
    # reflects requested sleep, not actual wait.
    from app.services import bulk_data_fetcher as bdf_module
    from app.services import rate_limiter as rl_module

    yf_mod = _make_simulator_module(multi_sim)
    with sampler, \
            patch.object(bdf_module, "yf", yf_mod), \
            patch.object(bdf_module.time, "sleep", _no_sleep), \
            patch.object(rl_module.time, "sleep", _no_sleep):
        with ThreadPoolExecutor(max_workers=len(SUPPORTED_MARKETS)) as pool:
            futures = {
                pool.submit(
                    _run_one_market,
                    market,
                    synthetic_universe[market],
                    sampler,
                ): market
                for market in SUPPORTED_MARKETS
            }
            for future in as_completed(futures):
                market_results.append(future.result())

    # Build per-market metrics by joining run results with simulator stats.
    sim_stats = multi_sim.stats
    market_metrics: List[MarketMetrics] = []
    for market, wall_s, symbols_processed, transient_failures in market_results:
        stats = sim_stats[market]
        counter_429 = _read_429_counter(market)
        market_metrics.append(MarketMetrics(
            market=market,
            wall_clock_s=round(wall_s, 3),
            symbols_processed=symbols_processed,
            yfinance_calls=stats["calls"],
            rate_limit_429s=max(counter_429, stats["rate_limited"]),
            transient_failures=transient_failures,
            p50_batch_latency_s=round(stats["p50_latency_s"], 4),
            p95_batch_latency_s=round(stats["p95_latency_s"], 4),
            p99_batch_latency_s=round(stats["p99_latency_s"], 4),
        ))

    # Stable order: by SUPPORTED_MARKETS index (US, HK, JP, TW).
    market_metrics.sort(key=lambda m: SUPPORTED_MARKETS.index(m.market))

    snapshot = LoadRunSnapshot(
        run_id=str(uuid.uuid4()),
        git_sha=_git_sha(),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        scenario="parallel_per_market_weekly_refresh",
        seed=simulator_seed,
        universe_size_per_market={m: len(syms) for m, syms in synthetic_universe.items()},
        markets=market_metrics,
        worker_resources=sampler.metrics(),
    )

    # Always write the current run for human inspection, regardless of pass/fail.
    current_path = BASELINE_PATH.parent / "current_run.json"
    write_snapshot(snapshot, current_path)
    print(f"\nLoad run snapshot written to: {current_path}")
    for m in market_metrics:
        print(
            f"  {m.market}: {m.wall_clock_s:.2f}s wall  "
            f"{m.rate_limit_429s} 429s  "
            f"{m.symbols_processed} symbols  "
            f"p95={m.p95_batch_latency_s:.3f}s"
        )

    # Bootstrap the baseline if missing (first run on a fresh checkout).
    if BASELINE_PATH.exists() is False or os.getenv("LOAD_TEST_UPDATE_BASELINE") == "1":
        write_snapshot(snapshot, BASELINE_PATH)
        pytest.skip(
            f"Baseline written to {BASELINE_PATH}. "
            f"Re-run without LOAD_TEST_UPDATE_BASELINE to gate against it."
        )

    baseline = read_snapshot(BASELINE_PATH)
    assert baseline is not None, "Baseline file should exist after bootstrap"
    report = compare_to_baseline(snapshot, baseline)

    print("\n" + report.format_summary())

    assert not report.has_regressions, (
        f"Load run regressed vs baseline:\n{report.format_summary()}\n"
        f"If this regression is intentional, re-run with "
        f"LOAD_TEST_UPDATE_BASELINE=1 to update the committed baseline."
    )
