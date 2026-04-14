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
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pytest

from app.services.rate_budget_policy import get_rate_budget_policy
from app.tasks.market_queues import SUPPORTED_MARKETS

from .harness import no_sleep, read_429_counter, run_parallel_refresh
from .measurement import (
    LoadRunSnapshot,
    MarketMetrics,
    ResourceSampler,
    compare_to_baseline,
    read_snapshot,
    write_snapshot,
)
from .yfinance_simulator import build_multi_market_simulator


pytestmark = pytest.mark.load


BASELINE_PATH = Path(__file__).parent / "baselines" / "per_market_load.json"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


# Shared parallel-runner + simulator wiring lives in ``harness.py``.


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

    multi_sim = build_multi_market_simulator(seed=simulator_seed, sleep_fn=no_sleep)
    sampler = ResourceSampler(interval_s=0.5)

    with sampler:
        market_results = run_parallel_refresh(multi_sim, synthetic_universe, sampler)

    # Build per-market metrics by joining run results with simulator stats.
    sim_stats = multi_sim.stats
    market_metrics: List[MarketMetrics] = []
    for market, wall_s, symbols_processed, transient_failures in market_results:
        stats = sim_stats[market]
        counter_429 = read_429_counter(market)
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
