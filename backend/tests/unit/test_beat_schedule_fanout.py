"""Beat schedule per-market fan-out (bead StockScreenClaude-asia.9.1).

Verifies that every market-scoped beat entry carries both a market kwarg AND
an explicit queue option, and that the queue matches the market. Without this
check it's easy for a new beat entry to silently land on the shared queue.
"""
from __future__ import annotations

from app.celery_app import _build_cache_warmup_beat_schedule, celery_app
from app.config import settings
from app.tasks.market_queues import (
    SHARED_DATA_FETCH_QUEUE,
    SUPPORTED_MARKETS,
    data_fetch_queue_for_market,
    market_jobs_queue_for_market,
)

# Beat entry name prefixes that MUST be fanned out per market onto the
# external-fetch lane.
EXTERNAL_FETCH_PREFIXES = (
    "weekly-full-refresh-",
    "weekly-fundamental-refresh-",
    "weekly-universe-refresh-",
)

# Per-market compute/write tasks run on market_jobs_<market>, not data_fetch.
MARKET_JOB_PREFIXES = (
    "daily-market-pipeline-",
)


def _market_entries(prefixes):
    schedule = celery_app.conf.beat_schedule or {}
    for name, entry in schedule.items():
        for prefix in prefixes:
            if name.startswith(prefix):
                suffix = name[len(prefix):]  # e.g. "us"
                yield name, entry, suffix.upper()


def _scheduled_market_entries(schedule=None):
    schedule = schedule if schedule is not None else (celery_app.conf.beat_schedule or {})
    for name, entry in schedule.items():
        market = (entry.get("kwargs") or {}).get("market")
        if isinstance(market, str) and market.strip().upper() in SUPPORTED_MARKETS:
            yield name, entry, market.strip().upper()


def _deployment_enabled_markets():
    return tuple(settings.enabled_markets_list)


class TestBeatScheduleFanout:
    def test_development_model_work_uses_shared_data_fetch_queue(self):
        entry = celery_app.conf.beat_schedule["theme-development-preparation"]
        assert entry["options"]["queue"] == SHARED_DATA_FETCH_QUEUE

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

    def test_market_scoped_entries_are_limited_to_deployment_enabled_markets(self):
        enabled = set(_deployment_enabled_markets())
        disabled_entries = [
            name
            for name, _entry, market in _scheduled_market_entries()
            if market not in enabled
        ]

        assert disabled_entries == []

    def test_builder_limits_market_scoped_entries_to_explicit_enabled_subset(self):
        schedule = _build_cache_warmup_beat_schedule(["US"])

        market_scoped_entries = list(_scheduled_market_entries(schedule))
        assert market_scoped_entries
        assert [
            name
            for name, _entry, market in market_scoped_entries
            if market != "US"
        ] == []

        for prefix in (*EXTERNAL_FETCH_PREFIXES, *MARKET_JOB_PREFIXES):
            assert f"{prefix}us" in schedule
            for market in (m.lower() for m in SUPPORTED_MARKETS if m != "US"):
                assert f"{prefix}{market}" not in schedule

    def test_deployment_enabled_markets_are_covered_for_external_fetch_prefixes(self):
        schedule = celery_app.conf.beat_schedule or {}
        for prefix in EXTERNAL_FETCH_PREFIXES:
            present = {
                name[len(prefix):].upper()
                for name in schedule
                if name.startswith(prefix)
            }
            for m in _deployment_enabled_markets():
                assert m in present, (
                    f"No beat entry for market {m!r} with prefix {prefix!r}. "
                    f"Fan-out gap: got {sorted(present)}"
                )

    def test_daily_market_pipelines_run_on_market_job_queue(self):
        for name, entry, expected_market in _market_entries(MARKET_JOB_PREFIXES):
            queue = (entry.get("options") or {}).get("queue")
            expected_queue = market_jobs_queue_for_market(expected_market)
            assert queue == expected_queue, (
                f"Market job entry {name!r} routes to {queue!r}, expected {expected_queue!r}"
            )
            assert entry["task"] == "app.tasks.daily_market_pipeline_tasks.queue_daily_market_pipeline"
            assert entry["kwargs"]["market"] == expected_market

    def test_weekly_universe_refresh_uses_market_appropriate_task(self):
        schedule = celery_app.conf.beat_schedule or {}
        if "US" in _deployment_enabled_markets():
            assert schedule["weekly-universe-refresh-us"]["task"] == (
                "app.tasks.universe_tasks.refresh_stock_universe"
            )
        else:
            assert "weekly-universe-refresh-us" not in schedule
        for market in (m.lower() for m in _deployment_enabled_markets() if m != "US"):
            assert schedule[f"weekly-universe-refresh-{market}"]["task"] == (
                "app.tasks.universe_tasks.refresh_official_market_universe"
            )

    def test_independent_daily_refresh_compute_and_snapshot_entries_are_removed(self):
        schedule = celery_app.conf.beat_schedule or {}
        for market in SUPPORTED_MARKETS:
            m_lower = market.lower()
            assert f"daily-smart-refresh-{m_lower}" not in schedule
            assert f"daily-breadth-calculation-{m_lower}" not in schedule
            assert f"daily-group-ranking-calculation-{m_lower}" not in schedule
            assert f"daily-feature-snapshot-{m_lower}" not in schedule
            expected_presence = market in _deployment_enabled_markets()
            assert (f"daily-market-pipeline-{m_lower}" in schedule) is expected_presence
