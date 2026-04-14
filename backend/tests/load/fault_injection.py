"""Fault injection helpers for chaos-lite isolation tests (bead asia.9.4).

Each helper mutates one market's behavior in a known way. The companion
isolation tests then assert that **non-affected markets continue operating
normally** — that's the operational guarantee 9.1 (per-market queues +
locks) and 9.2 (per-market rate budgets) are supposed to deliver.

The fault helpers don't crash anything; they just change the simulator's
per-market profile or pre-acquire a lock externally. Tests assert post-
conditions: wall-clock for non-affected markets stays within +20% of the
no-fault baseline, the affected market fails as expected, and the harness
itself doesn't crash.
"""
from __future__ import annotations

from dataclasses import replace

from .yfinance_simulator import MultiMarketSimulator


def _current_profile(simulator: MultiMarketSimulator, market: str):
    """Return the current profile for ``market`` via the public stats API."""
    return simulator._sims[market].profile  # noqa: SLF001 — read-only inspection


def inject_provider_exhaustion(simulator: MultiMarketSimulator, victim: str) -> None:
    """Force the victim market to return 100% 429 responses.

    Tests the assumption that per-market rate-limit keys (``yfinance:<market>``)
    don't propagate one market's 429 storm into others' counters or queues.
    """
    simulator.set_profile(
        victim,
        replace(
            _current_profile(simulator, victim),
            rate_limit_probability=1.0,
            failure_probability=0.0,
        ),
    )


def inject_provider_hard_failure(simulator: MultiMarketSimulator, victim: str) -> None:
    """Force the victim market to raise non-429 exceptions on every call.

    Tests that an upstream provider fault on one market doesn't poison the
    shared `BulkDataFetcher` instance or the rate-limiter for other markets.
    """
    simulator.set_profile(
        victim,
        replace(
            _current_profile(simulator, victim),
            rate_limit_probability=0.0,
            failure_probability=1.0,
        ),
    )


def hold_market_lock(victim: str, task_id: str = "stuck-task-id"):
    """Externally acquire ``data_fetch_job_lock:<victim>`` for the test scope.

    Simulates the operational scenario where a previous task crashed without
    releasing its market lock (e.g., container OOM-killed mid-fetch). The
    isolation guarantee 9.1 introduced is that this stuck lock blocks ONLY the
    affected market; other markets' Beat tasks still acquire their own keys.

    Thin wrapper over ``DataFetchLock.external_hold`` so chaos tests can call
    a one-arg helper without instantiating the lock themselves.
    """
    from app.wiring.bootstrap import get_data_fetch_lock

    return get_data_fetch_lock().external_hold(
        "stuck_external_holder", task_id, market=victim,
    )
