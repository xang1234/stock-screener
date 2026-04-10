"""Tests for the static-site export CLI bootstrap behavior."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime
from types import SimpleNamespace

import app.scripts.export_static_site as export_script
import app.tasks.breadth_tasks as breadth_tasks
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
    monkeypatch.setattr(fundamentals_tasks, "refresh_all_fundamentals", make_task("fundamentals_refresh"))
    monkeypatch.setattr(breadth_tasks, "calculate_daily_breadth", make_task("breadth_refresh"))
    monkeypatch.setattr(group_rank_tasks, "calculate_daily_group_rankings", make_task("groups_refresh"))
    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(
            run=lambda **kwargs: calls.append("feature_snapshot")
            or {"run_id": 77, "kwargs": kwargs}
        ),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date: calls.append("price_refresh") or {"task": "price_refresh", "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_ensure_breadth_history",
        lambda *, as_of_date: calls.append("breadth_history_refresh") or {"task": "breadth_history_refresh", "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_ensure_group_rank_history",
        lambda *, as_of_date: calls.append("groups_history_refresh") or {"task": "groups_history_refresh", "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_us_trading_date",
        lambda: date(2026, 4, 2),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(
        feature_store_tasks,
        "_enrich_feature_run_with_ibd_metadata",
        lambda *, feature_run_id, ranking_date: calls.append("feature_metadata_refresh")
        or {"run_id": feature_run_id, "ranking_date": ranking_date.isoformat(), "updated_rows": 2},
    )

    results, warnings = export_script._run_daily_refresh()  # noqa: SLF001 - intentional unit test coverage

    assert warnings == []
    assert calls == [
        "universe_refresh",
        "fundamentals_refresh",
        "price_refresh",
        "feature_snapshot",
        "breadth_refresh",
        "breadth_history_refresh",
        "groups_history_refresh",
        "groups_refresh",
        "feature_metadata_refresh",
    ]
    assert results["universe_refresh"]["task"] == "universe_refresh"
    assert results["ibd_seed_refresh"]["loaded"] == 10105
    assert results["feature_snapshot"]["kwargs"] == {
        "as_of_date_str": "2026-04-02",
        "static_daily_mode": True,
    }
    assert results["breadth_refresh"]["kwargs"] == {
        "calculation_date": "2026-04-02",
        "force_cache_only": True,
    }
    assert results["groups_refresh"]["kwargs"] == {
        "calculation_date": "2026-04-02",
        "force_cache_only": True,
    }
    assert results["feature_metadata_refresh"]["run_id"] == 77


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
    monkeypatch.setattr(fundamentals_tasks, "refresh_all_fundamentals", make_task("fundamentals_refresh"))
    monkeypatch.setattr(breadth_tasks, "calculate_daily_breadth", make_task("breadth_refresh"))
    monkeypatch.setattr(group_rank_tasks, "calculate_daily_group_rankings", make_task("groups_refresh"))
    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(
            run=lambda **kwargs: calls.append("feature_snapshot")
            or {"run_id": 77, "kwargs": kwargs}
        ),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date: calls.append("price_refresh") or {"task": "price_refresh", "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_ensure_breadth_history",
        lambda *, as_of_date: calls.append("breadth_history_refresh") or {"task": "breadth_history_refresh", "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_ensure_group_rank_history",
        lambda *, as_of_date: calls.append("groups_history_refresh") or {"task": "groups_history_refresh", "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_us_trading_date",
        lambda: date(2026, 4, 2),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(
        export_script.provider_snapshot_service,
        "hydrate_published_snapshot",
        lambda db, allow_yahoo_hydration=False: hydrate_calls.append((db, allow_yahoo_hydration))
        or {"task": "fundamentals_hydrate"},
    )
    monkeypatch.setattr(
        feature_store_tasks,
        "_enrich_feature_run_with_ibd_metadata",
        lambda *, feature_run_id, ranking_date: calls.append("feature_metadata_refresh")
        or {"run_id": feature_run_id, "ranking_date": ranking_date.isoformat(), "updated_rows": 2},
    )

    results, warnings = export_script._run_daily_refresh(  # noqa: SLF001 - intentional unit test coverage
        skip_universe_refresh=True,
        skip_fundamentals_refresh=True,
        hydrate_published_snapshot=True,
    )

    assert warnings == []
    assert calls == [
        "price_refresh",
        "feature_snapshot",
        "breadth_refresh",
        "breadth_history_refresh",
        "groups_history_refresh",
        "groups_refresh",
        "feature_metadata_refresh",
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
    monkeypatch.setattr(fundamentals_tasks, "refresh_all_fundamentals", make_task("fundamentals_refresh"))
    monkeypatch.setattr(breadth_tasks, "calculate_daily_breadth", make_task("breadth_refresh"))
    monkeypatch.setattr(group_rank_tasks, "calculate_daily_group_rankings", make_task("groups_refresh"))
    monkeypatch.setattr(feature_store_tasks, "build_daily_snapshot", make_task("feature_snapshot"))
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date: {"task": "price_refresh", "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_ensure_breadth_history",
        lambda *, as_of_date: {"task": "breadth_history_refresh", "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_ensure_group_rank_history",
        lambda *, as_of_date: {"task": "groups_history_refresh", "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_us_trading_date",
        lambda: date(2026, 4, 2),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(
        feature_store_tasks,
        "_enrich_feature_run_with_ibd_metadata",
        lambda **_kwargs: {"run_id": 1, "ranking_date": "2026-04-02", "updated_rows": 0},
    )

    export_script._run_daily_refresh()  # noqa: SLF001 - intentional unit test coverage

    assert events == ["enter", "exit"]
    assert all(lock_disabled for _, lock_disabled in calls)


def test_run_daily_refresh_uses_static_daily_mode_and_group_rank_bypass(monkeypatch):
    calls: list[tuple[str, dict, bool, bool]] = []
    state = {"group_bypass": False, "breadth_bypass": False}

    @contextmanager
    def fake_group_bypass():
        state["group_bypass"] = True
        try:
            yield
        finally:
            state["group_bypass"] = False

    @contextmanager
    def fake_breadth_bypass():
        state["breadth_bypass"] = True
        try:
            yield
        finally:
            state["breadth_bypass"] = False

    def make_task(name: str):
        def run(**kwargs):
            calls.append((name, kwargs, state["group_bypass"], state["breadth_bypass"]))
            return {"task": name, "kwargs": kwargs}

        return SimpleNamespace(run=run)

    monkeypatch.setattr(group_rank_tasks, "allow_same_day_group_rank_warmup_bypass", fake_group_bypass)
    monkeypatch.setattr(breadth_tasks, "allow_same_day_breadth_warmup_bypass", fake_breadth_bypass)
    monkeypatch.setattr(universe_tasks, "refresh_stock_universe", make_task("universe_refresh"))
    monkeypatch.setattr(fundamentals_tasks, "refresh_all_fundamentals", make_task("fundamentals_refresh"))
    monkeypatch.setattr(breadth_tasks, "calculate_daily_breadth", make_task("breadth_refresh"))
    monkeypatch.setattr(group_rank_tasks, "calculate_daily_group_rankings", make_task("groups_refresh"))
    monkeypatch.setattr(feature_store_tasks, "build_daily_snapshot", make_task("feature_snapshot"))
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date: {"task": "price_refresh", "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_ensure_breadth_history",
        lambda *, as_of_date: {"task": "breadth_history_refresh", "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_ensure_group_rank_history",
        lambda *, as_of_date: {"task": "groups_history_refresh", "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_us_trading_date",
        lambda: date(2026, 4, 2),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(
        feature_store_tasks,
        "_enrich_feature_run_with_ibd_metadata",
        lambda **_kwargs: {"run_id": 1, "ranking_date": "2026-04-02", "updated_rows": 0},
    )

    export_script._run_daily_refresh()  # noqa: SLF001 - intentional unit test coverage

    groups_call = next(call for call in calls if call[0] == "groups_refresh")
    breadth_call = next(call for call in calls if call[0] == "breadth_refresh")
    feature_call = next(call for call in calls if call[0] == "feature_snapshot")
    assert groups_call[2] is True
    assert breadth_call[3] is True
    assert feature_call[1] == {
        "as_of_date_str": "2026-04-02",
        "static_daily_mode": True,
    }
    assert groups_call[1] == {
        "calculation_date": "2026-04-02",
        "force_cache_only": True,
    }
    assert breadth_call[1] == {
        "calculation_date": "2026-04-02",
        "force_cache_only": True,
    }


def test_resolve_latest_completed_us_trading_date_uses_last_market_close(monkeypatch):
    monkeypatch.setattr(
        export_script,
        "get_last_market_close",
        lambda: datetime(2026, 4, 2, 16, 0),
    )

    assert export_script._resolve_latest_completed_us_trading_date() == datetime(2026, 4, 2, 16, 0).date()
