"""Tests for the static-site export CLI bootstrap behavior."""

from __future__ import annotations

from contextlib import contextmanager
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

    results, warnings = export_script._run_daily_refresh(  # noqa: SLF001 - intentional unit test coverage
        refresh_themes_best_effort=False,
    )

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
        export_script.provider_snapshot_service,
        "hydrate_published_snapshot",
        lambda db, allow_yahoo_hydration=False: hydrate_calls.append((db, allow_yahoo_hydration))
        or {"task": "fundamentals_hydrate"},
    )

    results, warnings = export_script._run_daily_refresh(  # noqa: SLF001 - intentional unit test coverage
        refresh_themes_best_effort=False,
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

    export_script._run_daily_refresh(refresh_themes_best_effort=False)  # noqa: SLF001 - intentional unit test coverage

    assert events == ["enter", "exit"]
    assert all(lock_disabled for _, lock_disabled in calls)
