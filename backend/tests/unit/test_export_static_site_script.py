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
        "_resolve_latest_completed_us_trading_date",
        lambda: date(2026, 4, 2),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(
        export_script,
        "_upsert_feature_run_pointer",
        lambda *, pointer_key, run_id: calls.append(f"pointer:{pointer_key}:{run_id}"),
    )

    results, warnings = export_script._run_daily_refresh()  # noqa: SLF001 - intentional unit test coverage

    expected_markets = list(export_script.STATIC_EXPORT_MARKETS)

    assert warnings == []
    assert calls == [
        "universe_refresh",
        "fundamentals_refresh",
        "price_refresh",
        *(["feature_snapshot"] * len(expected_markets)),
        "pointer:latest_published:77",
    ]
    assert results["universe_refresh"]["task"] == "universe_refresh"
    assert results["ibd_seed_refresh"]["loaded"] == 10105
    assert set(results["feature_snapshots"]) == set(expected_markets)
    for market in expected_markets:
        assert results["feature_snapshots"][market]["kwargs"] == {
            "as_of_date_str": "2026-04-02",
            "static_daily_mode": True,
            "universe_name": f"market:{market.lower()}",
            "market": market,
            "publish_pointer_key": f"latest_published_market:{market}",
        }
    assert results["default_market_pointer"] == {
        "market": "US",
        "pointer_key": "latest_published",
        "run_id": 77,
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
    monkeypatch.setattr(fundamentals_tasks, "refresh_all_fundamentals", make_task("fundamentals_refresh"))
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
        "_resolve_latest_completed_us_trading_date",
        lambda: date(2026, 4, 2),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(
        export_script,
        "get_provider_snapshot_service",
        lambda: SimpleNamespace(
            hydrate_all_published_snapshots=lambda db, allow_yahoo_hydration=False: hydrate_calls.append((db, allow_yahoo_hydration))
            or {"task": "fundamentals_hydrate"},
        ),
    )
    monkeypatch.setattr(
        export_script,
        "_upsert_feature_run_pointer",
        lambda *, pointer_key, run_id: calls.append(f"pointer:{pointer_key}:{run_id}"),
    )

    results, warnings = export_script._run_daily_refresh(  # noqa: SLF001 - intentional unit test coverage
        skip_universe_refresh=True,
        skip_fundamentals_refresh=True,
        build_mode=export_script.STATIC_BUILD_MODE_FULL,
        hydrate_published_snapshot=True,
    )

    assert warnings == []
    assert calls == [
        "price_refresh",
        *(["feature_snapshot"] * len(export_script.STATIC_EXPORT_MARKETS)),
        "pointer:latest_published:77",
    ]
    assert hydrate_calls == [("db-session", False)]
    assert "universe_refresh" not in results
    assert "fundamentals_refresh" not in results
    assert results["fundamentals_hydrate"]["task"] == "fundamentals_hydrate"


def test_run_daily_refresh_price_delta_mode_skips_snapshot_hydration(monkeypatch):
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
        "_resolve_latest_completed_us_trading_date",
        lambda: date(2026, 4, 2),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(
        export_script,
        "get_provider_snapshot_service",
        lambda: SimpleNamespace(
            hydrate_all_published_snapshots=lambda db, allow_yahoo_hydration=False: hydrate_calls.append((db, allow_yahoo_hydration))
            or {"task": "fundamentals_hydrate"},
        ),
    )
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)

    results, warnings = export_script._run_daily_refresh(  # noqa: SLF001 - intentional unit test coverage
        skip_universe_refresh=True,
        skip_fundamentals_refresh=True,
        build_mode=export_script.STATIC_BUILD_MODE_PRICE_DELTA,
        hydrate_published_snapshot=True,
    )

    assert warnings == []
    assert hydrate_calls == []
    assert "fundamentals_hydrate" not in results


def test_run_daily_refresh_warns_when_default_market_run_id_is_missing(monkeypatch):
    calls: list[str] = []

    def build_snapshot(**kwargs):
        calls.append(kwargs["market"])
        if kwargs["market"] == export_script.STATIC_DEFAULT_MARKET:
            return {"status": "completed"}
        return {"run_id": 77, "kwargs": kwargs}

    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(run=build_snapshot),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date: {"task": "price_refresh", "as_of_date": as_of_date.isoformat()},
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
    monkeypatch.setattr(universe_tasks, "refresh_stock_universe", SimpleNamespace(run=lambda: {"task": "universe_refresh"}))
    monkeypatch.setattr(
        fundamentals_tasks,
        "refresh_all_fundamentals",
        SimpleNamespace(run=lambda: {"task": "fundamentals_refresh"}),
    )
    monkeypatch.setattr(
        export_script,
        "_upsert_feature_run_pointer",
        lambda **_kwargs: calls.append("pointer"),
    )

    results, warnings = export_script._run_daily_refresh()  # noqa: SLF001 - intentional unit test coverage

    assert calls == list(export_script.STATIC_EXPORT_MARKETS)
    assert "default_market_pointer" not in results
    assert warnings == ["No US feature snapshot produced a run id; 'latest_published' was not updated."]


def test_run_daily_refresh_does_not_repoint_default_pointer_for_unpublished_us_run(monkeypatch):
    pointer_calls: list[dict] = []

    def build_snapshot(**kwargs):
        if kwargs["market"] == export_script.STATIC_DEFAULT_MARKET:
            return {"status": "failed", "run_id": 91}
        return {"status": "published", "run_id": 77, "kwargs": kwargs}

    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(run=build_snapshot),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date: {"task": "price_refresh", "as_of_date": as_of_date.isoformat()},
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
    monkeypatch.setattr(universe_tasks, "refresh_stock_universe", SimpleNamespace(run=lambda: {"task": "universe_refresh"}))
    monkeypatch.setattr(
        fundamentals_tasks,
        "refresh_all_fundamentals",
        SimpleNamespace(run=lambda: {"task": "fundamentals_refresh"}),
    )
    monkeypatch.setattr(
        export_script,
        "_upsert_feature_run_pointer",
        lambda **kwargs: pointer_calls.append(kwargs),
    )

    results, warnings = export_script._run_daily_refresh()  # noqa: SLF001 - intentional unit test coverage

    assert pointer_calls == []
    assert "default_market_pointer" not in results
    assert warnings == ["US feature snapshot returned status 'failed'; 'latest_published' was not updated."]


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
    monkeypatch.setattr(feature_store_tasks, "build_daily_snapshot", make_task("feature_snapshot"))
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date: {"task": "price_refresh", "as_of_date": as_of_date.isoformat()},
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
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)

    export_script._run_daily_refresh()  # noqa: SLF001 - intentional unit test coverage

    assert events == ["enter", "exit"]
    assert all(lock_disabled for _, lock_disabled in calls)


def test_run_daily_refresh_uses_static_daily_mode_and_group_rank_bypass(monkeypatch):
    calls: list[tuple[str, dict]] = []

    def make_task(name: str):
        def run(**kwargs):
            calls.append((name, kwargs))
            return {"task": name, "kwargs": kwargs}

        return SimpleNamespace(run=run)

    monkeypatch.setattr(universe_tasks, "refresh_stock_universe", make_task("universe_refresh"))
    monkeypatch.setattr(fundamentals_tasks, "refresh_all_fundamentals", make_task("fundamentals_refresh"))
    monkeypatch.setattr(feature_store_tasks, "build_daily_snapshot", make_task("feature_snapshot"))
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date: {"task": "price_refresh", "as_of_date": as_of_date.isoformat()},
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
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)

    export_script._run_daily_refresh()  # noqa: SLF001 - intentional unit test coverage

    feature_calls = [call for call in calls if call[0] == "feature_snapshot"]
    assert len(feature_calls) == len(export_script.STATIC_EXPORT_MARKETS)
    assert feature_calls[0][1] == {
        "as_of_date_str": "2026-04-02",
        "static_daily_mode": True,
        "universe_name": "market:us",
        "market": "US",
        "publish_pointer_key": "latest_published_market:US",
    }


def test_resolve_latest_completed_us_trading_date_uses_last_market_close(monkeypatch):
    monkeypatch.setattr(
        export_script,
        "get_last_market_close",
        lambda: datetime(2026, 4, 2, 16, 0),
    )

    assert export_script._resolve_latest_completed_us_trading_date() == datetime(2026, 4, 2, 16, 0).date()
