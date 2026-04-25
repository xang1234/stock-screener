"""Beat schedule per-market fan-out (bead StockScreenClaude-asia.9.1).

Verifies that every market-scoped beat entry carries both a market kwarg AND
an explicit queue option, and that the queue matches the market. Without this
check it's easy for a new beat entry to silently land on the shared queue.
"""
from __future__ import annotations

import pytest

from app.celery_app import celery_app
from app.tasks.market_queues import (
    SUPPORTED_MARKETS,
    data_fetch_queue_for_market,
    market_jobs_queue_for_market,
)


# Beat entry name prefixes that MUST be fanned out per market onto the
# external-fetch lane.
EXTERNAL_FETCH_PREFIXES = (
    "daily-smart-refresh-",
    "weekly-full-refresh-",
    "weekly-fundamental-refresh-",
    "weekly-universe-refresh-",
)

# Per-market compute/write tasks run on market_jobs_<market>, not data_fetch.
MARKET_JOB_PREFIXES = (
    "daily-feature-snapshot-",
)


def _market_entries(prefixes):
    schedule = celery_app.conf.beat_schedule or {}
    for name, entry in schedule.items():
        for prefix in prefixes:
            if name.startswith(prefix):
                suffix = name[len(prefix):]  # e.g. "us"
                yield name, entry, suffix.upper()


class TestBeatScheduleFanout:
    def test_each_external_fetch_entry_has_market_kwarg(self):
        for name, entry, expected_market in _market_entries(EXTERNAL_FETCH_PREFIXES):
            kwargs = entry.get("kwargs", {})
            assert kwargs.get("market") == expected_market, (
                f"beat entry {name!r} missing/mismatched market kwarg "
                f"(got {kwargs.get('market')!r}, expected {expected_market!r})"
            )

    def test_each_external_fetch_entry_has_explicit_market_queue(self):
        for name, entry, expected_market in _market_entries(EXTERNAL_FETCH_PREFIXES):
            opts = entry.get("options") or {}
            queue = opts.get("queue")
            expected_queue = data_fetch_queue_for_market(expected_market)
            assert queue == expected_queue, (
                f"beat entry {name!r} routes to {queue!r}, expected {expected_queue!r}"
            )

    def test_every_market_is_covered_for_external_fetch_prefixes(self):
        schedule = celery_app.conf.beat_schedule or {}
        for prefix in EXTERNAL_FETCH_PREFIXES:
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

    def test_feature_snapshots_run_on_market_job_queue(self):
        for name, entry, expected_market in _market_entries(MARKET_JOB_PREFIXES):
            queue = (entry.get("options") or {}).get("queue")
            expected_queue = market_jobs_queue_for_market(expected_market)
            assert queue == expected_queue, (
                f"Market job entry {name!r} routes to {queue!r}, expected {expected_queue!r}"
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

    def test_breadth_and_group_rankings_fan_out_to_every_market(self):
        """Breadth + industry-group rankings now run daily for every
        SUPPORTED_MARKETS entry, each with its own market-jobs queue and the
        correct market kwarg. Previously US-only.
        """
        schedule = celery_app.conf.beat_schedule or {}
        for market in SUPPORTED_MARKETS:
            m_lower = market.lower()
            breadth_key = f"daily-breadth-calculation-{m_lower}"
            groups_key = f"daily-group-ranking-calculation-{m_lower}"
            assert breadth_key in schedule, f"missing {breadth_key}"
            assert groups_key in schedule, f"missing {groups_key}"
            assert schedule[breadth_key]["kwargs"]["market"] == market
            assert schedule[groups_key]["kwargs"]["market"] == market
            expected_queue = market_jobs_queue_for_market(market)
            assert schedule[breadth_key]["options"]["queue"] == expected_queue
            assert schedule[groups_key]["options"]["queue"] == expected_queue
