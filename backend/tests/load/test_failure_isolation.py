"""Fault-injection isolation tests (bead asia.9.4).

Validates the operational guarantee that 9.1 (per-market queues + locks) and
9.2 (per-market rate budgets) deliver: a failure in one market is contained
to that market and does not degrade throughput, success rate, or rate-budget
state of the other markets.

Each test injects ONE specific fault into ONE victim market, then runs the
parallel 4-market refresh and asserts:

  1. The non-affected markets process all their symbols (no cascading failure).
  2. The non-affected markets show no unexpected 429 counter increments.
  3. The harness does not crash (the affected market may fail, but the
     fixture-level test machinery completes).
  4. The non-affected markets' yfinance_calls match the healthy baseline
     exactly (deterministic sanity — fault should not change other markets'
     batch behavior).

Assertions 1-3 are CI-gated; assertion 4 is a stronger correctness check
that's also gated since it's deterministic.

Run:
    make gate-7-chaos
    pytest backend/tests/load/test_failure_isolation.py -v -m load
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.tasks.market_queues import SUPPORTED_MARKETS

from .fault_injection import (
    hold_market_lock,
    inject_provider_exhaustion,
    inject_provider_hard_failure,
)
from .harness import no_sleep, read_429_counter, run_parallel_refresh
from .measurement import ResourceSampler, read_snapshot
from .yfinance_simulator import build_multi_market_simulator


pytestmark = pytest.mark.load


HEALTHY_BASELINE_PATH = (
    Path(__file__).parent / "baselines" / "per_market_load.json"
)

# Pick a victim market for each test so the assertions are explicit about
# which market is "the affected one". Spreading across markets exercises the
# isolation guarantee in different positions of the supported-markets tuple.
VICTIM_PROVIDER_EXHAUSTION = "HK"
VICTIM_PROVIDER_HARD_FAILURE = "JP"
VICTIM_LOCK_STUCK = "TW"

# Fail loudly if a victim is removed from the supported-markets contract;
# otherwise the chaos tests would silently target a non-existent simulator.
assert all(
    v in SUPPORTED_MARKETS
    for v in (VICTIM_PROVIDER_EXHAUSTION, VICTIM_PROVIDER_HARD_FAILURE, VICTIM_LOCK_STUCK)
), "A victim market constant is no longer in SUPPORTED_MARKETS"


_HEALTHY_BASELINE_CACHE = None


def _healthy_baseline_for(market: str):
    """Look up the per-market healthy baseline from the 9.3 load snapshot.

    Cached at module level so the JSON file is read at most once per pytest
    session (called per non-affected market per chaos test = up to 9 calls
    without the cache).
    """
    global _HEALTHY_BASELINE_CACHE
    if _HEALTHY_BASELINE_CACHE is None:
        snapshot = read_snapshot(HEALTHY_BASELINE_PATH)
        if snapshot is None:
            pytest.skip(
                f"No healthy baseline at {HEALTHY_BASELINE_PATH}. "
                "Run `make load-baseline-update` first."
            )
        _HEALTHY_BASELINE_CACHE = snapshot
    for m in _HEALTHY_BASELINE_CACHE.markets:
        if m.market == market:
            return m
    pytest.fail(f"Market {market} missing from healthy baseline")


def _assert_market_isolated(
    market: str,
    result_tuple,
    sim_stats: dict,
):
    """Assert the four isolation guarantees for a non-affected market.

    1. All symbols processed (no transient failures from cross-market leak).
    2. No unexpected 429s in the per-market counter.
    3. yfinance_calls matches healthy baseline exactly (deterministic).
    4. (Implicit) No exception bubbled out of the runner.
    """
    _, wall_s, symbols_processed, transient_failures = result_tuple
    healthy = _healthy_baseline_for(market)

    # (1) All symbols processed AND transient-failure count matches the
    # healthy baseline (the simulator has baseline failure_probability for
    # every market, so isolation means "same as healthy", not "zero").
    assert transient_failures == healthy.transient_failures, (
        f"[{market}] {transient_failures} symbols failed under fault, "
        f"healthy baseline had {healthy.transient_failures} — cross-market "
        f"fault leaked extra failures into this market."
    )
    assert symbols_processed == healthy.symbols_processed, (
        f"[{market}] processed {symbols_processed}/{healthy.symbols_processed} "
        f"symbols — incomplete refresh under cross-market fault."
    )

    # (2) 429 counter unchanged.
    counter_429 = read_429_counter(market)
    assert counter_429 == 0, (
        f"[{market}] 429 counter incremented to {counter_429} despite fault "
        f"being in another market — the per-market rate-key isolation broke."
    )
    sim_429 = sim_stats[market]["rate_limited"]
    assert sim_429 == healthy.rate_limit_429s, (
        f"[{market}] simulator-injected 429s = {sim_429}, baseline = "
        f"{healthy.rate_limit_429s}. Cross-market fault should not change this."
    )

    # (3) Deterministic batch behavior unchanged.
    sim_calls = sim_stats[market]["calls"]
    assert sim_calls == healthy.yfinance_calls, (
        f"[{market}] yfinance_calls = {sim_calls}, baseline = "
        f"{healthy.yfinance_calls}. Cross-market fault leaked into this "
        f"market's batch loop."
    )


def _run_with_simulator(
    multi_sim, synthetic_universe, sampler
):
    """Run the parallel refresh and return ``(results_by_market, sim_stats)``."""
    results = run_parallel_refresh(multi_sim, synthetic_universe, sampler)
    by_market = {tup[0]: tup for tup in results}
    return by_market, multi_sim.stats


# ---------------------------------------------------------------------------
# Fault scenarios
# ---------------------------------------------------------------------------

def test_provider_exhaustion_isolation(
    redis_required, reset_redis_state, synthetic_universe, simulator_seed,
):
    """Victim market sees 100% yfinance 429s; other 3 markets must complete normally.

    Validates: per-market ``yfinance:<market>`` rate-limit keys (9.2) prevent
    one market's 429 storm from incrementing other markets' counters or
    consuming their rate budgets.
    """
    multi_sim = build_multi_market_simulator(seed=simulator_seed, sleep_fn=no_sleep)
    inject_provider_exhaustion(multi_sim, victim=VICTIM_PROVIDER_EXHAUSTION)
    sampler = ResourceSampler(interval_s=0.5)

    with sampler:
        by_market, sim_stats = _run_with_simulator(multi_sim, synthetic_universe, sampler)

    # Assertion (3) on the affected market: harness must not crash, victim
    # is allowed to fail or partially succeed.
    assert VICTIM_PROVIDER_EXHAUSTION in by_market, (
        "Harness crashed under provider exhaustion — affected market did not "
        "produce a result tuple."
    )

    # Assertions (1), (2), (4) on non-affected markets.
    for market in SUPPORTED_MARKETS:
        if market == VICTIM_PROVIDER_EXHAUSTION:
            continue
        _assert_market_isolated(market, by_market[market], sim_stats)


def test_provider_hard_failure_isolation(
    redis_required, reset_redis_state, synthetic_universe, simulator_seed,
):
    """Victim market raises non-429 exceptions on every call; others must complete.

    Validates: a non-rate-limit upstream failure (network error, malformed
    response, etc.) on one market does not poison the shared ``BulkDataFetcher``
    instance or block other markets via shared state.
    """
    multi_sim = build_multi_market_simulator(seed=simulator_seed, sleep_fn=no_sleep)
    inject_provider_hard_failure(multi_sim, victim=VICTIM_PROVIDER_HARD_FAILURE)
    sampler = ResourceSampler(interval_s=0.5)

    with sampler:
        by_market, sim_stats = _run_with_simulator(multi_sim, synthetic_universe, sampler)

    assert VICTIM_PROVIDER_HARD_FAILURE in by_market, (
        "Harness crashed under provider hard failure."
    )

    for market in SUPPORTED_MARKETS:
        if market == VICTIM_PROVIDER_HARD_FAILURE:
            continue
        _assert_market_isolated(market, by_market[market], sim_stats)


def test_lock_stuck_isolation(
    redis_required, reset_redis_state, synthetic_universe, simulator_seed,
):
    """Holding one market's lock externally must not contaminate other markets.

    Validates the lock-key isolation half of 9.1: holding
    ``data_fetch_job_lock:<victim>`` does not affect any other market's
    fetcher operations or shared mutable state. Specifically, other markets'
    batch loops, RateBudgetPolicy intervals, and 429 counters operate
    exactly as in the no-fault baseline despite the victim's lock being
    pinned in Redis.

    Note: the harness calls ``fetch_prices_in_batches`` directly (not via
    the ``@serialized_data_fetch`` decorator), so the victim's own runner
    proceeds despite the held lock. This test does NOT exercise the
    decorator's lock-contention path — that's an integration-test concern
    against the actual Celery task. What this test guarantees is the
    narrower but operationally meaningful property that *holding the lock
    has no cross-market side effects on the fetcher itself*.
    """
    multi_sim = build_multi_market_simulator(seed=simulator_seed, sleep_fn=no_sleep)
    sampler = ResourceSampler(interval_s=0.5)

    with hold_market_lock(victim=VICTIM_LOCK_STUCK), sampler:
        by_market, sim_stats = _run_with_simulator(multi_sim, synthetic_universe, sampler)

    assert VICTIM_LOCK_STUCK in by_market, "Harness crashed under lock-stuck scenario."

    for market in SUPPORTED_MARKETS:
        if market == VICTIM_LOCK_STUCK:
            continue
        _assert_market_isolated(market, by_market[market], sim_stats)
