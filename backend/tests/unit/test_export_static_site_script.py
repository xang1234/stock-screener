"""Tests for the static-site export CLI bootstrap behavior."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys
from unittest.mock import MagicMock

import pytest

import app.scripts.export_static_site as export_script
import app.tasks.fundamentals_tasks as fundamentals_tasks
import app.tasks.universe_tasks as universe_tasks
from app.domain.markets import market_registry
from app.interfaces.tasks import feature_store_tasks


def test_static_export_markets_match_market_registry():
    assert export_script.STATIC_EXPORT_MARKETS == market_registry.supported_market_codes()


def test_ensure_group_rank_history_uses_market_calendar_for_non_us_market(monkeypatch):
    query = MagicMock()
    query.filter.return_value = query
    query.distinct.return_value = query
    query.all.return_value = []

    db = MagicMock()
    db.query.return_value = query

    @contextmanager
    def fake_session():
        yield db

    calendar_calls: list[tuple[str, date]] = []
    hk_trading_dates = {date(2026, 4, 3), date(2026, 4, 7)}

    def is_trading_day(market: str, day: date) -> bool:
        calendar_calls.append((market, day))
        return day in hk_trading_dates

    fill_calls: list[dict] = []

    def fill_gaps_optimized(db_arg, missing_dates, *, market):
        fill_calls.append(
            {
                "db": db_arg,
                "missing_dates": list(missing_dates),
                "market": market,
            }
        )
        return {"processed": len(missing_dates), "errors": 0}

    monkeypatch.setattr(export_script, "SessionLocal", fake_session)
    monkeypatch.setattr(
        export_script,
        "get_market_calendar_service",
        lambda: SimpleNamespace(is_trading_day=is_trading_day),
    )
    monkeypatch.setattr(
        export_script,
        "get_group_rank_service",
        lambda: SimpleNamespace(fill_gaps_optimized=fill_gaps_optimized),
    )

    result = export_script._ensure_group_rank_history(  # noqa: SLF001 - intentional unit test coverage
        as_of_date=date(2026, 4, 7),
        market="hk",
    )

    assert {market for market, _day in calendar_calls} == {"HK"}
    assert calendar_calls[0] == ("HK", date(2025, 12, 28))
    assert calendar_calls[-1] == ("HK", date(2026, 4, 7))
    assert fill_calls == [
        {
            "db": db,
            "missing_dates": [date(2026, 4, 3), date(2026, 4, 7)],
            "market": "HK",
        }
    ]
    assert result["status"] == "completed"
    assert result["market"] == "HK"
    assert result["missing_dates"] == 2


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
        lambda *, as_of_date, market=None: calls.append("price_refresh") or {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
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
        *(["price_refresh"] * len(expected_markets)),
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
            "ignore_runtime_market_gate": True,
        }
    assert set(results["price_refresh"]) == set(expected_markets)
    assert results["default_market_pointer"] == {
        "market": "US",
        "pointer_key": "latest_published",
        "run_id": 77,
    }


def test_run_daily_refresh_uses_resolved_tracked_ibd_csv_path(monkeypatch, tmp_path):
    calls: list[str] = []
    resolved_csv = tmp_path / "data" / "IBD_industry_group.csv"
    resolved_csv.parent.mkdir(parents=True, exist_ok=True)
    resolved_csv.write_text("AAPL,Software\n", encoding="utf-8")

    def make_task(name: str):
        return SimpleNamespace(
            run=lambda **kwargs: calls.append(name) or {"task": name, "kwargs": kwargs},
        )

    load_calls: list[tuple[object, Path | None]] = []

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
        lambda *, as_of_date, market=None: calls.append("price_refresh") or {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
    )
    monkeypatch.setattr(export_script, "_tracked_ibd_csv_path", lambda: resolved_csv)
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: load_calls.append((db, csv_path)) or 10105,
    )
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)

    results, warnings = export_script._run_daily_refresh()  # noqa: SLF001 - intentional unit test coverage

    assert warnings == []
    assert load_calls
    assert load_calls[0][1] == resolved_csv
    assert results["ibd_seed_refresh"] == {
        "csv_path": str(resolved_csv),
        "loaded": 10105,
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
        lambda *, as_of_date, market=None: calls.append("price_refresh") or {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
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
        *(["price_refresh"] * len(export_script.STATIC_EXPORT_MARKETS)),
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
        lambda *, as_of_date, market=None: calls.append("price_refresh") or {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
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
        lambda *, as_of_date, market=None: {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
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
        lambda *, as_of_date, market=None: {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
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
    calls: list[tuple[str, bool, bool]] = []
    state = {"fetch_lock_disabled": False, "workload_disabled": False}
    events: list[str] = []

    def make_disable(name: str, state_key: str):
        @contextmanager
        def _ctx():
            events.append(f"enter:{name}")
            state[state_key] = True
            try:
                yield
            finally:
                state[state_key] = False
                events.append(f"exit:{name}")

        return _ctx

    def make_task(name: str):
        def run(**kwargs):
            calls.append((name, state["fetch_lock_disabled"], state["workload_disabled"]))
            return {"task": name, "kwargs": kwargs}

        return SimpleNamespace(run=run)

    monkeypatch.setattr(
        export_script,
        "disable_serialized_data_fetch_lock",
        make_disable("fetch", "fetch_lock_disabled"),
    )
    monkeypatch.setattr(
        export_script,
        "disable_serialized_market_workload",
        make_disable("workload", "workload_disabled"),
    )
    monkeypatch.setattr(universe_tasks, "refresh_stock_universe", make_task("universe_refresh"))
    monkeypatch.setattr(fundamentals_tasks, "refresh_all_fundamentals", make_task("fundamentals_refresh"))
    monkeypatch.setattr(feature_store_tasks, "build_daily_snapshot", make_task("feature_snapshot"))
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)

    export_script._run_daily_refresh()  # noqa: SLF001 - intentional unit test coverage

    assert events == ["enter:fetch", "enter:workload", "exit:workload", "exit:fetch"]
    assert all(fetch_disabled and workload_disabled for _, fetch_disabled, workload_disabled in calls)


def test_run_daily_refresh_limits_work_to_selected_market(monkeypatch):
    calls: list[tuple[str, dict]] = []

    def make_task(name: str):
        def run(**kwargs):
            calls.append((name, kwargs))
            return {"task": name, "kwargs": kwargs, "run_id": 77}

        return SimpleNamespace(run=run)

    monkeypatch.setattr(universe_tasks, "refresh_stock_universe", make_task("universe_refresh"))
    monkeypatch.setattr(fundamentals_tasks, "refresh_all_fundamentals", make_task("fundamentals_refresh"))
    monkeypatch.setattr(feature_store_tasks, "build_daily_snapshot", make_task("feature_snapshot"))
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)

    results, warnings = export_script._run_daily_refresh(  # noqa: SLF001 - intentional unit test coverage
        market="HK",
    )

    assert warnings == []
    assert results["price_refresh"]["market"] == "HK"
    assert set(results["feature_snapshots"]) == {"HK"}
    assert calls == [
        ("universe_refresh", {"market": "HK"}),
        ("fundamentals_refresh", {"market": "HK"}),
        (
            "feature_snapshot",
            {
                "as_of_date_str": "2026-04-02",
                "static_daily_mode": True,
                "universe_name": "market:hk",
                "market": "HK",
                "publish_pointer_key": "latest_published_market:HK",
                "ignore_runtime_market_gate": True,
            },
        ),
    ]


def test_run_daily_refresh_uses_per_market_trading_date_for_in(monkeypatch):
    """Regression: ``build_daily_snapshot`` for IN must receive IN's latest
    completed trading date, not NYSE's. Before the per-market calendar fix,
    running ``--market IN`` on a day NSE was closed but NYSE was open caused
    the snapshot to be silently skipped with ``reason='not_trading_day'``."""

    snapshot_calls: list[dict] = []

    def build_snapshot_run(**kwargs):
        snapshot_calls.append(kwargs)
        return {"status": "published", "run_id": 91}

    # Simulate the failure scenario: NYSE traded on Apr 2 (US holiday for
    # IN), so IN's latest completed session is Apr 1. The previous bug used
    # Apr 2 for IN's snapshot, hitting the not_trading_day guard.
    market_dates = {"US": date(2026, 4, 2), "IN": date(2026, 4, 1)}

    monkeypatch.setattr(
        universe_tasks,
        "refresh_stock_universe",
        SimpleNamespace(run=lambda **_kwargs: {"task": "universe_refresh"}),
    )
    monkeypatch.setattr(
        fundamentals_tasks,
        "refresh_all_fundamentals",
        SimpleNamespace(run=lambda **_kwargs: {"task": "fundamentals_refresh"}),
    )
    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(run=build_snapshot_run),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: {
            "task": "price_refresh",
            "market": market,
            "as_of_date": as_of_date.isoformat(),
        },
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda selected_market: market_dates[selected_market],
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)

    results, _warnings = export_script._run_daily_refresh(market="IN")  # noqa: SLF001

    assert results["price_refresh"]["as_of_date"] == "2026-04-01"
    assert snapshot_calls == [
        {
            "as_of_date_str": "2026-04-01",
            "static_daily_mode": True,
            "universe_name": "market:in",
            "market": "IN",
            "publish_pointer_key": "latest_published_market:IN",
            "ignore_runtime_market_gate": True,
        }
    ]


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
        lambda *, as_of_date, market=None: {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
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
        "ignore_runtime_market_gate": True,
    }


def test_run_daily_refresh_reenriches_ibd_metadata_after_group_rank_backfill(monkeypatch):
    """Group ranks for ``as_of_date`` are backfilled by ``_ensure_group_rank_history``
    only after ``build_daily_snapshot`` has already run its inner enrichment.
    The driver must re-run ``_enrich_feature_run_with_ibd_metadata`` so the
    static export reads up-to-date ``ibd_group_rank`` values from
    ``details_json``."""

    events: list[str] = []

    def make_task(name: str):
        def run(**kwargs):
            events.append(name)
            return {"task": name, "kwargs": kwargs, "run_id": 77, "status": "published"}

        return SimpleNamespace(run=run)

    enrich_calls: list[dict] = []

    def fake_enrich(*, feature_run_id, ranking_date, **_kwargs):
        events.append(f"enrich:{feature_run_id}")
        enrich_calls.append({"feature_run_id": feature_run_id, "ranking_date": ranking_date})
        return {"run_id": feature_run_id, "updated_rows": 3, "missing_rank_rows": 0}

    group_rank_calls: list[dict] = []

    def fake_ensure_group_rank_history(*, as_of_date, market):
        events.append(f"group_rank:{market}")
        group_rank_calls.append({"as_of_date": as_of_date, "market": market})
        return {"status": "completed", "market": market, "missing_dates": 1}

    monkeypatch.setattr(universe_tasks, "refresh_stock_universe", make_task("universe_refresh"))
    monkeypatch.setattr(fundamentals_tasks, "refresh_all_fundamentals", make_task("fundamentals_refresh"))
    monkeypatch.setattr(feature_store_tasks, "build_daily_snapshot", make_task("feature_snapshot"))
    monkeypatch.setattr(
        feature_store_tasks,
        "_enrich_feature_run_with_ibd_metadata",
        fake_enrich,
    )
    monkeypatch.setattr(export_script, "_ensure_group_rank_history", fake_ensure_group_rank_history)
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)

    results, warnings = export_script._run_daily_refresh(market="US")  # noqa: SLF001 - intentional unit test coverage

    assert warnings == []
    # Order matters: the re-enrich step must run *after* the group-rank
    # backfill (otherwise it would re-read the same empty IBDGroupRank rows
    # build_daily_snapshot already saw).
    assert events == [
        "universe_refresh",
        "fundamentals_refresh",
        "feature_snapshot",
        "group_rank:US",
        "enrich:77",
    ]
    assert group_rank_calls == [{"as_of_date": date(2026, 4, 2), "market": "US"}]
    assert enrich_calls == [{"feature_run_id": 77, "ranking_date": date(2026, 4, 2)}]
    assert results["ibd_metadata_refresh"]["US"] == {
        "run_id": 77,
        "updated_rows": 3,
        "missing_rank_rows": 0,
    }


def test_run_daily_refresh_skips_reenrich_when_group_rank_backfill_errored(monkeypatch):
    """If ``_ensure_group_rank_history`` fails, the IBDGroupRank table is
    still missing ``as_of_date`` rows. Re-enriching in that state would
    overwrite previously valid ``ibd_group_rank`` values with ``None``
    (most damaging when ``build_daily_snapshot`` returned ``already_published``
    and the existing run carries ranks from an earlier successful refresh).
    The driver must skip re-enrich when the backfill did not succeed."""

    enrich_calls: list[dict] = []

    def fake_enrich(**kwargs):
        enrich_calls.append(kwargs)
        return {"updated_rows": 99}

    monkeypatch.setattr(universe_tasks, "refresh_stock_universe", SimpleNamespace(run=lambda **_kwargs: {"task": "universe_refresh"}))
    monkeypatch.setattr(fundamentals_tasks, "refresh_all_fundamentals", SimpleNamespace(run=lambda **_kwargs: {"task": "fundamentals_refresh"}))
    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(
            run=lambda **kwargs: {"status": "published", "run_id": 77, "kwargs": kwargs}
        ),
    )
    monkeypatch.setattr(
        feature_store_tasks,
        "_enrich_feature_run_with_ibd_metadata",
        fake_enrich,
    )
    monkeypatch.setattr(
        export_script,
        "_ensure_group_rank_history",
        lambda *, as_of_date, market: {
            "status": "errored",
            "market": market,
            "error": "Failed to fetch SPY benchmark data",
        },
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: {"task": "price_refresh"},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
    )
    monkeypatch.setattr(
        export_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path=None: 10105,
    )
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)

    results, _warnings = export_script._run_daily_refresh(market="US")  # noqa: SLF001 - intentional unit test coverage

    assert enrich_calls == []
    assert results["ibd_metadata_refresh"]["US"] == {
        "status": "skipped",
        "market": "US",
        "reason": "group_rank_backfill_errored",
    }


def test_run_daily_refresh_skips_reenrich_when_snapshot_not_ready(monkeypatch):
    enrich_calls: list[dict] = []

    def fake_enrich(**kwargs):
        enrich_calls.append(kwargs)
        return {"updated_rows": 0}

    def build_snapshot(**kwargs):
        if kwargs["market"] == "US":
            return {"status": "failed", "run_id": 91}
        return {"status": "published", "run_id": 77, "kwargs": kwargs}

    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(run=build_snapshot),
    )
    monkeypatch.setattr(
        feature_store_tasks,
        "_enrich_feature_run_with_ibd_metadata",
        fake_enrich,
    )
    monkeypatch.setattr(
        export_script,
        "_ensure_group_rank_history",
        lambda **_kwargs: {"status": "completed"},
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: {"task": "price_refresh"},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
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
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)

    results, _warnings = export_script._run_daily_refresh()  # noqa: SLF001 - intentional unit test coverage

    # US returned status "failed" → not snapshot_ready → re-enrich must skip it
    assert results["ibd_metadata_refresh"]["US"]["status"] == "skipped"
    assert results["ibd_metadata_refresh"]["US"]["reason"] == "snapshot_not_ready"
    # The other markets returned no status but a run_id, so they ARE re-enriched.
    other_markets = [m for m in export_script.STATIC_EXPORT_MARKETS if m != "US"]
    assert all(
        results["ibd_metadata_refresh"][m]["updated_rows"] == 0
        for m in other_markets
    )
    assert {call["feature_run_id"] for call in enrich_calls} == {77}


def test_run_daily_refresh_warns_when_non_default_market_snapshot_is_not_publish_ready(monkeypatch):
    def build_snapshot(**kwargs):
        if kwargs["market"] == "HK":
            return {
                "status": "skipped",
                "reason": "market HK is disabled in local runtime preferences",
                "market": "HK",
            }
        return {"status": "published", "run_id": 77, "kwargs": kwargs}

    monkeypatch.setattr(
        feature_store_tasks,
        "build_daily_snapshot",
        SimpleNamespace(run=build_snapshot),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda *, as_of_date, market=None: {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 4, 2),
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
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)

    _results, warnings = export_script._run_daily_refresh()  # noqa: SLF001 - intentional unit test coverage

    assert "Static export market HK snapshot returned status 'skipped' (market HK is disabled in local runtime preferences)." in warnings


def test_resolve_latest_completed_trading_date_uses_market_calendar(monkeypatch):
    """Per-market resolution: each market hits its own calendar instead of
    NYSE's. Regression test for the IN-skipped-on-US-trading-day bug."""

    calls: list[str] = []
    market_dates = {
        "US": date(2026, 4, 2),
        "IN": date(2026, 4, 3),
        "HK": date(2026, 4, 1),
    }

    def last_completed_trading_day(market: str) -> date:
        calls.append(market)
        return market_dates[market]

    monkeypatch.setattr(
        export_script,
        "get_market_calendar_service",
        lambda: SimpleNamespace(last_completed_trading_day=last_completed_trading_day),
    )

    assert export_script._resolve_latest_completed_trading_date("IN") == date(2026, 4, 3)
    assert export_script._resolve_latest_completed_trading_date("US") == date(2026, 4, 2)
    assert export_script._resolve_latest_completed_trading_date("HK") == date(2026, 4, 1)
    assert calls == ["IN", "US", "HK"]


def test_main_rejects_market_in_combine_mode(monkeypatch, tmp_path):
    combine_calls: list[tuple[object, object, bool]] = []

    monkeypatch.setattr(
        export_script.StaticSiteExportService,
        "combine_market_artifacts",
        lambda artifacts_dir, output_dir, *, clean=True: combine_calls.append((artifacts_dir, output_dir, clean)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_static_site.py",
            "--output-dir",
            str(tmp_path / "out"),
            "--combine-artifacts-dir",
            str(tmp_path / "artifacts"),
            "--market",
            "HK",
        ],
    )

    with pytest.raises(SystemExit, match="--combine-artifacts-dir cannot be used together with --market"):
        export_script.main()

    assert combine_calls == []


def test_main_rejects_fallback_artifacts_without_combine_mode(monkeypatch, tmp_path):
    combine_calls: list[tuple[object, object, object, bool]] = []

    monkeypatch.setattr(
        export_script.StaticSiteExportService,
        "combine_market_artifacts",
        lambda artifacts_dir, output_dir, *, fallback_artifacts_dir=None, clean=True: combine_calls.append(
            (artifacts_dir, output_dir, fallback_artifacts_dir, clean)
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_static_site.py",
            "--output-dir",
            str(tmp_path / "out"),
            "--fallback-artifacts-dir",
            str(tmp_path / "fallback"),
        ],
    )

    with pytest.raises(SystemExit, match="--fallback-artifacts-dir requires --combine-artifacts-dir"):
        export_script.main()

    assert combine_calls == []


def test_main_passes_fallback_artifacts_dir_to_combine(monkeypatch, tmp_path):
    combine_calls: list[tuple[object, object, object, bool]] = []
    output_dir = tmp_path / "out"
    artifacts_dir = tmp_path / "artifacts"
    fallback_dir = tmp_path / "fallback"

    monkeypatch.setattr(
        export_script.StaticSiteExportService,
        "combine_market_artifacts",
        lambda artifacts_dir, output_dir, *, fallback_artifacts_dir=None, clean=True: combine_calls.append(
            (artifacts_dir, output_dir, fallback_artifacts_dir, clean)
        )
        or SimpleNamespace(
            output_dir=output_dir,
            generated_at="2026-04-05T22:00:00Z",
            as_of_date="2026-04-05",
            warnings=(),
            manifest={},
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_static_site.py",
            "--output-dir",
            str(output_dir),
            "--combine-artifacts-dir",
            str(artifacts_dir),
            "--fallback-artifacts-dir",
            str(fallback_dir),
            "--no-clean",
        ],
    )

    assert export_script.main() == 0

    assert combine_calls == [(artifacts_dir, output_dir, fallback_dir, False)]


def test_main_returns_skip_code_for_market_not_trading_day(monkeypatch, tmp_path, capsys):
    export_calls: list[object] = []
    output_dir = tmp_path / "out"

    monkeypatch.setattr(export_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(
        export_script,
        "_run_daily_refresh",
        lambda **_kwargs: (
            {
                "feature_snapshots": {
                    "TW": {
                        "status": "skipped",
                        "reason": "not_trading_day",
                        "market": "TW",
                        "as_of_date": "2026-05-01",
                    }
                }
            },
            ["Static export market TW snapshot returned status 'skipped' (not_trading_day)."],
        ),
    )

    class ExportShouldNotRun:
        def __init__(self, *_args, **_kwargs):
            export_calls.append("constructed")

        def export(self, *_args, **_kwargs):
            raise AssertionError("market export should not run for not_trading_day")

    monkeypatch.setattr(export_script, "StaticSiteExportService", ExportShouldNotRun)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_static_site.py",
            "--output-dir",
            str(output_dir),
            "--refresh-daily",
            "--market",
            "TW",
        ],
    )

    assert export_script.main() == 78

    captured = capsys.readouterr()
    assert "Static site export skipped for market TW because it is not a trading day." in captured.out
    assert export_calls == []
    assert not (output_dir / "diagnostics" / "tw" / "snapshot-failure.json").exists()


def test_write_market_diagnostics_records_quarantined_snapshot(tmp_path):
    path = export_script._write_market_diagnostics(  # noqa: SLF001 - intentional unit test coverage
        tmp_path / "out",
        "in",
        {
            "status": "quarantined",
            "reason": "data_quality_gate",
            "run_id": 91,
            "existing_run_id": 77,
            "failed_symbols": ["RELIANCE.NS", "TCS.NS"],
            "row_count": 4312,
            "warnings": ["price rows missing"],
            "failure_diagnostics": {"failed_symbol_count": 2},
            "ignored": "not exported",
        },
    )

    assert path == tmp_path / "out" / "diagnostics" / "in" / "snapshot-failure.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "market": "IN",
        "status": "quarantined",
        "reason": "data_quality_gate",
        "run_id": 91,
        "existing_run_id": 77,
        "failed_symbols": ["RELIANCE.NS", "TCS.NS"],
        "row_count": 4312,
        "warnings": ["price rows missing"],
        "failure_diagnostics": {"failed_symbol_count": 2},
    }


def test_main_returns_no_current_artifact_code_for_quarantined_selected_market(
    monkeypatch,
    tmp_path,
    capsys,
):
    output_dir = tmp_path / "out"

    monkeypatch.setattr(export_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(
        export_script,
        "_run_daily_refresh",
        lambda **_kwargs: (
            {
                "feature_snapshots": {
                    "IN": {
                        "status": "quarantined",
                        "reason": "data_quality_gate",
                        "market": "IN",
                        "run_id": 91,
                        "failed_symbols": ["TCS.NS"],
                        "row_count": 4312,
                        "failure_diagnostics": {"failed_symbol_count": 1},
                    }
                }
            },
            [],
        ),
    )

    class ExportRaisesNoCurrentArtifact:
        def __init__(self, *_args, **_kwargs):
            pass

        def export(self, output_dir_arg, *_args, **_kwargs):
            shutil.rmtree(output_dir_arg, ignore_errors=True)
            raise export_script.NoPublishedStaticMarketArtifact(
                "No current IN artifact",
                markets=("IN",),
            )

    monkeypatch.setattr(export_script, "StaticSiteExportService", ExportRaisesNoCurrentArtifact)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_static_site.py",
            "--output-dir",
            str(output_dir),
            "--refresh-daily",
            "--market",
            "IN",
        ],
    )

    assert export_script.main() == 79

    diagnostics_path = output_dir / "diagnostics" / "in" / "snapshot-failure.json"
    assert diagnostics_path.exists()
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert payload["market"] == "IN"
    assert payload["status"] == "quarantined"
    assert payload["failure_diagnostics"] == {"failed_symbol_count": 1}
    captured = capsys.readouterr()
    assert "fallback" in captured.out
    assert "skipped" in captured.out


def test_main_reraises_unrelated_runtime_errors_for_non_ready_selected_market(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(export_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(
        export_script,
        "_run_daily_refresh",
        lambda **_kwargs: (
            {
                "feature_snapshots": {
                    "IN": {
                        "status": "quarantined",
                        "market": "IN",
                        "failure_diagnostics": {"failed_symbol_count": 1},
                    }
                }
            },
            [],
        ),
    )

    class ExportRaisesUnrelatedRuntimeError:
        def __init__(self, *_args, **_kwargs):
            pass

        def export(self, *_args, **_kwargs):
            raise RuntimeError("database connection dropped")

    monkeypatch.setattr(export_script, "StaticSiteExportService", ExportRaisesUnrelatedRuntimeError)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_static_site.py",
            "--output-dir",
            str(tmp_path / "out"),
            "--refresh-daily",
            "--market",
            "IN",
        ],
    )

    with pytest.raises(RuntimeError, match="database connection dropped"):
        export_script.main()
