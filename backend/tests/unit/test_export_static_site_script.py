"""Tests for the static-site export CLI bootstrap behavior."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime
from types import SimpleNamespace

import app.scripts.export_static_site as export_script
import app.tasks.breadth_tasks as breadth_tasks
import app.tasks.cache_tasks as cache_tasks
import app.tasks.fundamentals_tasks as fundamentals_tasks
import app.tasks.group_rank_tasks as group_rank_tasks
import app.tasks.universe_tasks as universe_tasks
from app.interfaces.tasks import feature_store_tasks


def test_run_daily_refresh_bootstraps_universe_before_other_tasks(monkeypatch):
    calls: list[str] = []

    def make_task(name: str):
        return SimpleNamespace(
            run=lambda **kwargs: calls.append(name) or {"task": name, "kwargs": kwargs},
        )

    monkeypatch.setattr(universe_tasks, "refresh_stock_universe", make_task("universe_refresh"))
    monkeypatch.setattr(cache_tasks, "smart_refresh_cache", make_task("cache_refresh"))
    monkeypatch.setattr(fundamentals_tasks, "refresh_all_fundamentals", make_task("fundamentals_refresh"))
    monkeypatch.setattr(breadth_tasks, "calculate_daily_breadth_with_gapfill", make_task("breadth_refresh"))
    monkeypatch.setattr(group_rank_tasks, "calculate_daily_group_rankings", make_task("groups_refresh"))
    monkeypatch.setattr(feature_store_tasks, "build_daily_snapshot", make_task("feature_snapshot"))
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_us_trading_date",
        lambda: date(2026, 4, 2),
    )

    results, warnings = export_script._run_daily_refresh()  # noqa: SLF001 - intentional unit test coverage

    assert warnings == []
    assert calls == [
        "universe_refresh",
        "cache_refresh",
        "fundamentals_refresh",
        "breadth_refresh",
        "groups_refresh",
        "feature_snapshot",
    ]
    assert results["universe_refresh"]["task"] == "universe_refresh"
    assert results["cache_refresh"]["kwargs"] == {"mode": "full"}
    assert results["feature_snapshot"]["kwargs"] == {
        "as_of_date_str": "2026-04-02",
        "static_daily_mode": True,
    }


def test_run_daily_refresh_can_hydrate_imported_snapshot_without_live_fundamentals(monkeypatch):
    calls: list[str] = []

    def make_task(name: str):
        return SimpleNamespace(
            run=lambda **kwargs: calls.append(name) or {"task": name, "kwargs": kwargs},
        )

    @contextmanager
    def fake_session():
        yield "db-session"

    hydrate_calls: list[tuple[object, bool]] = []

    monkeypatch.setattr(export_script, "SessionLocal", fake_session)
    monkeypatch.setattr(cache_tasks, "smart_refresh_cache", make_task("cache_refresh"))
    monkeypatch.setattr(fundamentals_tasks, "refresh_all_fundamentals", make_task("fundamentals_refresh"))
    monkeypatch.setattr(breadth_tasks, "calculate_daily_breadth_with_gapfill", make_task("breadth_refresh"))
    monkeypatch.setattr(group_rank_tasks, "calculate_daily_group_rankings", make_task("groups_refresh"))
    monkeypatch.setattr(feature_store_tasks, "build_daily_snapshot", make_task("feature_snapshot"))
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_us_trading_date",
        lambda: date(2026, 4, 2),
    )
    monkeypatch.setattr(
        export_script.provider_snapshot_service,
        "hydrate_published_snapshot",
        lambda db, allow_yahoo_hydration=False: hydrate_calls.append((db, allow_yahoo_hydration))
        or {"task": "fundamentals_hydrate"},
    )

    results, warnings = export_script._run_daily_refresh(  # noqa: SLF001 - intentional unit test coverage
        skip_universe_refresh=True,
        skip_fundamentals_refresh=True,
        hydrate_published_snapshot=True,
    )

    assert warnings == []
    assert calls == [
        "cache_refresh",
        "breadth_refresh",
        "groups_refresh",
        "feature_snapshot",
    ]
    assert hydrate_calls == [("db-session", False)]
    assert "universe_refresh" not in results
    assert "fundamentals_refresh" not in results
    assert results["fundamentals_hydrate"]["task"] == "fundamentals_hydrate"


def test_run_daily_refresh_bypasses_smart_refresh_time_window_for_static_exports(monkeypatch):
    calls: list[tuple[str, bool]] = []
    state = {"time_window_bypassed": False}
    events: list[str] = []

    @contextmanager
    def fake_time_window_bypass():
        events.append("enter")
        state["time_window_bypassed"] = True
        try:
            yield
        finally:
            state["time_window_bypassed"] = False
            events.append("exit")

    def make_task(name: str):
        def run(**kwargs):
            calls.append((name, state["time_window_bypassed"]))
            return {"task": name, "kwargs": kwargs}

        return SimpleNamespace(run=run)

    monkeypatch.setattr(cache_tasks, "allow_smart_refresh_time_window_bypass", fake_time_window_bypass)
    monkeypatch.setattr(universe_tasks, "refresh_stock_universe", make_task("universe_refresh"))
    monkeypatch.setattr(cache_tasks, "smart_refresh_cache", make_task("cache_refresh"))
    monkeypatch.setattr(fundamentals_tasks, "refresh_all_fundamentals", make_task("fundamentals_refresh"))
    monkeypatch.setattr(breadth_tasks, "calculate_daily_breadth_with_gapfill", make_task("breadth_refresh"))
    monkeypatch.setattr(group_rank_tasks, "calculate_daily_group_rankings", make_task("groups_refresh"))
    monkeypatch.setattr(feature_store_tasks, "build_daily_snapshot", make_task("feature_snapshot"))
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_us_trading_date",
        lambda: date(2026, 4, 2),
    )

    export_script._run_daily_refresh()  # noqa: SLF001 - intentional unit test coverage

    assert events == ["enter", "exit"]
    assert calls[0] == ("universe_refresh", False)
    assert calls[1] == ("cache_refresh", True)


def test_run_daily_refresh_disables_serialized_lock_during_export(monkeypatch):
    calls: list[tuple[str, bool]] = []
    state = {"lock_disabled": False}
    events: list[str] = []

    @contextmanager
    def fake_disable_lock():
        events.append("enter")
        state["lock_disabled"] = True
        try:
            yield
        finally:
            state["lock_disabled"] = False
            events.append("exit")

    def make_task(name: str):
        def run(**kwargs):
            calls.append((name, state["lock_disabled"]))
            return {"task": name, "kwargs": kwargs}

        return SimpleNamespace(run=run)

    monkeypatch.setattr(export_script, "disable_serialized_data_fetch_lock", fake_disable_lock)
    monkeypatch.setattr(universe_tasks, "refresh_stock_universe", make_task("universe_refresh"))
    monkeypatch.setattr(cache_tasks, "smart_refresh_cache", make_task("cache_refresh"))
    monkeypatch.setattr(fundamentals_tasks, "refresh_all_fundamentals", make_task("fundamentals_refresh"))
    monkeypatch.setattr(breadth_tasks, "calculate_daily_breadth_with_gapfill", make_task("breadth_refresh"))
    monkeypatch.setattr(group_rank_tasks, "calculate_daily_group_rankings", make_task("groups_refresh"))
    monkeypatch.setattr(feature_store_tasks, "build_daily_snapshot", make_task("feature_snapshot"))
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_us_trading_date",
        lambda: date(2026, 4, 2),
    )

    export_script._run_daily_refresh()  # noqa: SLF001 - intentional unit test coverage

    assert events == ["enter", "exit"]
    assert all(lock_disabled for _, lock_disabled in calls)


def test_run_daily_refresh_uses_static_daily_mode_and_group_rank_bypass(monkeypatch):
    calls: list[tuple[str, dict, bool]] = []
    state = {"group_bypass": False}

    @contextmanager
    def fake_group_bypass():
        state["group_bypass"] = True
        try:
            yield
        finally:
            state["group_bypass"] = False

    def make_task(name: str):
        def run(**kwargs):
            calls.append((name, kwargs, state["group_bypass"]))
            return {"task": name, "kwargs": kwargs}

        return SimpleNamespace(run=run)

    monkeypatch.setattr(group_rank_tasks, "allow_same_day_group_rank_warmup_bypass", fake_group_bypass)
    monkeypatch.setattr(universe_tasks, "refresh_stock_universe", make_task("universe_refresh"))
    monkeypatch.setattr(cache_tasks, "smart_refresh_cache", make_task("cache_refresh"))
    monkeypatch.setattr(fundamentals_tasks, "refresh_all_fundamentals", make_task("fundamentals_refresh"))
    monkeypatch.setattr(breadth_tasks, "calculate_daily_breadth_with_gapfill", make_task("breadth_refresh"))
    monkeypatch.setattr(group_rank_tasks, "calculate_daily_group_rankings", make_task("groups_refresh"))
    monkeypatch.setattr(feature_store_tasks, "build_daily_snapshot", make_task("feature_snapshot"))
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_us_trading_date",
        lambda: date(2026, 4, 2),
    )

    export_script._run_daily_refresh()  # noqa: SLF001 - intentional unit test coverage

    groups_call = next(call for call in calls if call[0] == "groups_refresh")
    feature_call = next(call for call in calls if call[0] == "feature_snapshot")
    assert groups_call[2] is True
    assert feature_call[1] == {
        "as_of_date_str": "2026-04-02",
        "static_daily_mode": True,
    }


def test_resolve_latest_completed_us_trading_date_uses_last_market_close(monkeypatch):
    monkeypatch.setattr(
        export_script,
        "get_last_market_close",
        lambda: datetime(2026, 4, 2, 16, 0),
    )

    assert export_script._resolve_latest_completed_us_trading_date() == datetime(2026, 4, 2, 16, 0).date()
