"""Tests for the static-site export CLI bootstrap behavior."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime
from types import SimpleNamespace
import sys

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.scripts.export_static_site as export_script
import app.tasks.breadth_tasks as breadth_tasks
import app.tasks.fundamentals_tasks as fundamentals_tasks
import app.tasks.group_rank_tasks as group_rank_tasks
import app.tasks.universe_tasks as universe_tasks
from app.database import Base
from app.interfaces.tasks import feature_store_tasks
from app.models.stock import StockPrice
from app.models.stock_universe import StockUniverse


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
            "ignore_runtime_market_gate": True,
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
        lambda *, as_of_date, market=None: calls.append("price_refresh") or {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
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
        lambda *, as_of_date, market=None: calls.append("price_refresh") or {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
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
        lambda *, as_of_date, market=None: {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
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
        lambda *, as_of_date, market=None: {"task": "price_refresh", "market": market, "as_of_date": as_of_date.isoformat()},
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
        "_resolve_latest_completed_us_trading_date",
        lambda: date(2026, 4, 2),
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
        "ignore_runtime_market_gate": True,
    }


def test_refresh_static_daily_prices_filters_to_selected_market(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[StockUniverse.__table__, StockPrice.__table__])
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)

    with session_factory() as db:
        db.add_all(
            [
                StockUniverse(symbol="0700.HK", market="HK", is_active=True, market_cap=100.0),
                StockUniverse(symbol="9988.HK", market="HK", is_active=True, market_cap=90.0),
                StockUniverse(symbol="AAPL", market="US", is_active=True, market_cap=120.0),
                StockUniverse(symbol="BAD-W", market="HK", is_active=True, market_cap=80.0),
            ]
        )
        db.add(
            StockPrice(
                symbol="0700.HK",
                date=date(2026, 4, 1),
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                volume=1000,
            )
        )
        db.add(
            StockPrice(
                symbol="AAPL",
                date=date(2026, 4, 1),
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                volume=1000,
            )
        )
        db.commit()

    fetch_calls: list[dict] = []
    stored_batches: list[dict] = []

    class _FakeFetcher:
        def fetch_prices_in_batches(self, symbols, period="2y", start_batch_size=None, market=None):
            fetch_calls.append(
                {
                    "symbols": list(symbols),
                    "period": period,
                    "market": market,
                }
            )
            return {
                symbol: {"price_data": SimpleNamespace(empty=False), "has_error": False}
                for symbol in symbols
            }

    monkeypatch.setattr(export_script, "SessionLocal", session_factory)
    monkeypatch.setattr(export_script, "BulkDataFetcher", lambda: _FakeFetcher())
    monkeypatch.setattr(
        export_script,
        "get_price_cache",
        lambda: SimpleNamespace(
            store_batch_in_cache=lambda payload, also_store_db=True: stored_batches.append(
                {"symbols": sorted(payload.keys()), "also_store_db": also_store_db}
            )
        ),
    )

    result = export_script._refresh_static_daily_prices(  # noqa: SLF001 - intentional unit test coverage
        as_of_date=date(2026, 4, 2),
        market="HK",
    )

    assert result["market"] == "HK"
    assert result["total_active_symbols"] == 3
    assert result["supported_symbols"] == 2
    assert result["skipped_unsupported_symbols"] == 1
    assert fetch_calls == [{"symbols": ["0700.HK"], "period": "7d", "market": "HK"}]
    assert stored_batches == [{"symbols": ["0700.HK"], "also_store_db": True}]


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
    monkeypatch.setattr(export_script, "_upsert_feature_run_pointer", lambda **_kwargs: None)

    _results, warnings = export_script._run_daily_refresh()  # noqa: SLF001 - intentional unit test coverage

    assert "Static export market HK snapshot returned status 'skipped' (market HK is disabled in local runtime preferences)." in warnings


def test_resolve_latest_completed_us_trading_date_uses_last_market_close(monkeypatch):
    monkeypatch.setattr(
        export_script,
        "get_last_market_close",
        lambda: datetime(2026, 4, 2, 16, 0),
    )

    assert export_script._resolve_latest_completed_us_trading_date() == datetime(2026, 4, 2, 16, 0).date()


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
