"""Beat schedule per-market fan-out (bead StockScreenClaude-asia.9.1).

Verifies that every market-scoped beat entry carries both a market kwarg AND
an explicit queue option, and that the queue matches the market. Without this
check it's easy for a new beat entry to silently land on the shared queue.
"""
from __future__ import annotations

import pytest

from app.celery_app import celery_app
from app.tasks.market_queues import (
    SHARED_DATA_FETCH_QUEUE,
    SUPPORTED_MARKETS,
    data_fetch_queue_for_market,
)


# Beat entry name prefixes that MUST be fanned out per market.
MARKET_SCOPED_PREFIXES = (
    "daily-smart-refresh-",
    "daily-breadth-calculation-",
    "daily-group-ranking-calculation-",
    "daily-feature-snapshot-",
    "weekly-full-refresh-",
    "weekly-fundamental-refresh-",
    "weekly-universe-refresh-",
)


def _market_entries():
    schedule = celery_app.conf.beat_schedule or {}
    for name, entry in schedule.items():
        for prefix in MARKET_SCOPED_PREFIXES:
            if name.startswith(prefix):
                suffix = name[len(prefix):]  # e.g. "us"
                yield name, entry, suffix.upper()


class TestBeatScheduleFanout:
    def test_each_market_scoped_entry_has_market_kwarg(self):
        for name, entry, expected_market in _market_entries():
            kwargs = entry.get("kwargs", {})
            assert kwargs.get("market") == expected_market, (
                f"beat entry {name!r} missing/mismatched market kwarg "
                f"(got {kwargs.get('market')!r}, expected {expected_market!r})"
            )

    def test_each_market_scoped_entry_has_explicit_queue(self):
        for name, entry, expected_market in _market_entries():
            opts = entry.get("options") or {}
            queue = opts.get("queue")
            expected_queue = data_fetch_queue_for_market(expected_market)
            assert queue == expected_queue, (
                f"beat entry {name!r} routes to {queue!r}, expected {expected_queue!r}"
            )

    def test_every_market_is_covered_for_each_prefix(self):
        schedule = celery_app.conf.beat_schedule or {}
        for prefix in MARKET_SCOPED_PREFIXES:
            present = {
                name[len(prefix):].upper()
                for name in schedule
                if name.startswith(prefix)
            }
            for m in SUPPORTED_MARKETS:
                assert m in present, (
                    f"No beat entry for market {m!r} with prefix {prefix!r}. "
                    f"Fan-out gap: got {sorted(present)}"
                )

    def test_no_market_entry_routes_to_shared_queue(self):
        for name, entry, _ in _market_entries():
            queue = (entry.get("options") or {}).get("queue")
            assert queue != SHARED_DATA_FETCH_QUEUE, (
                f"Market-scoped entry {name!r} incorrectly routes to shared queue"
            )

    def test_weekly_universe_refresh_uses_market_appropriate_task(self):
        schedule = celery_app.conf.beat_schedule or {}
        assert schedule["weekly-universe-refresh-us"]["task"] == (
            "app.tasks.universe_tasks.refresh_stock_universe"
        )
        for market in (m.lower() for m in SUPPORTED_MARKETS if m != "US"):
            assert schedule[f"weekly-universe-refresh-{market}"]["task"] == (
                "app.tasks.universe_tasks.refresh_official_market_universe"
            )
