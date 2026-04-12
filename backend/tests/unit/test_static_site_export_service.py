"""Tests for the static-site export service."""

from __future__ import annotations

from datetime import date, datetime
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.services.static_site_export_service as export_module
from app.database import Base
from app.infra.db.models.feature_store import FeatureRun, FeatureRunPointer
from app.models.stock import StockPrice
from app.services.static_site_export_service import (
    STATIC_SITE_SCHEMA_VERSION,
    StaticSiteSectionUnavailableError,
    StaticSiteExportService,
)


@pytest.fixture
def service_and_session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        engine,
        tables=[FeatureRun.__table__, FeatureRunPointer.__table__, StockPrice.__table__],
    )
    session_factory = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )
    try:
        yield StaticSiteExportService(session_factory), session_factory
    finally:
        engine.dispose()


def _insert_runs(session_factory, *runs: FeatureRun, pointer_run_id: int | None = None) -> None:
    with session_factory() as db:
        db.add_all(runs)
        if pointer_run_id is not None:
            db.add(FeatureRunPointer(key="latest_published", run_id=pointer_run_id))
        db.commit()


def test_get_latest_published_run_prefers_latest_published_pointer(service_and_session_factory):
    service, session_factory = service_and_session_factory
    _insert_runs(
        session_factory,
        FeatureRun(
            id=1,
            as_of_date=date(2026, 3, 28),
            run_type="daily_snapshot",
            status="published",
            published_at=datetime(2026, 3, 28, 21, 30, 0),
        ),
        FeatureRun(
            id=2,
            as_of_date=date(2026, 3, 29),
            run_type="daily_snapshot",
            status="published",
            published_at=datetime(2026, 3, 29, 21, 30, 0),
        ),
        pointer_run_id=1,
    )

    with session_factory() as db:
        run = service._get_latest_published_run(db)  # noqa: SLF001 - intentional unit test coverage

    assert run is not None
    assert run.id == 1


def test_get_latest_published_run_falls_back_to_latest_published_when_pointer_is_not_published(
    service_and_session_factory,
):
    service, session_factory = service_and_session_factory
    _insert_runs(
        session_factory,
        FeatureRun(
            id=1,
            as_of_date=date(2026, 3, 28),
            run_type="daily_snapshot",
            status="published",
            published_at=datetime(2026, 3, 28, 21, 30, 0),
        ),
        FeatureRun(
            id=2,
            as_of_date=date(2026, 3, 29),
            run_type="daily_snapshot",
            status="published",
            published_at=datetime(2026, 3, 29, 21, 30, 0),
        ),
        FeatureRun(
            id=3,
            as_of_date=date(2026, 3, 30),
            run_type="daily_snapshot",
            status="completed",
            published_at=None,
        ),
        pointer_run_id=3,
    )

    with session_factory() as db:
        run = service._get_latest_published_run(db)  # noqa: SLF001 - intentional unit test coverage

    assert run is not None
    assert run.id == 2


def test_export_writes_serializable_manifest_and_page_bundles(
    service_and_session_factory,
    monkeypatch,
    tmp_path,
):
    service, session_factory = service_and_session_factory
    _insert_runs(
        session_factory,
        FeatureRun(
            id=7,
            as_of_date=date(2026, 3, 31),
            run_type="daily_snapshot",
            status="published",
            published_at=datetime(2026, 3, 31, 21, 30, 0),
        ),
        pointer_run_id=7,
    )

    scan_manifest = {
        "schema_version": "static-scan-v1",
        "generated_at": "2026-03-31T22:00:00Z",
        "as_of_date": "2026-03-31",
        "run_id": 7,
        "rows_total": 2,
        "default_filtered_rows_total": 1,
        "chunks": [{"path": "scan/chunks/chunk-0001.json", "count": 2}],
        "preview_rows": [{"symbol": "NVDA", "composite_score": 97.5}],
    }
    breadth_payload = {
        "schema_version": STATIC_SITE_SCHEMA_VERSION,
        "generated_at": "2026-03-31T22:00:00Z",
        "available": True,
        "payload": {"current": {"date": "2026-03-31"}},
    }
    groups_payload = {
        "schema_version": STATIC_SITE_SCHEMA_VERSION,
        "generated_at": "2026-03-31T22:00:00Z",
        "available": True,
        "payload": {
            "rankings": {
                "date": "2026-03-31",
                "rankings": [{"industry_group": "Semiconductors", "rank": 1}],
            },
        },
    }
    home_payload = {
        "schema_version": STATIC_SITE_SCHEMA_VERSION,
        "generated_at": "2026-03-31T22:00:00Z",
        "as_of_date": "2026-03-31",
        "freshness": {"scan_run_id": 7},
        "key_markets": [],
        "scan_summary": {"top_results": [{"symbol": "NVDA"}]},
        "top_groups": [{"industry_group": "Semiconductors", "rank": 1}],
    }
    chart_manifest = {
        "path": "charts/index.json",
        "limit": 200,
        "symbols_total": 1,
        "available": True,
        "skipped_symbols": [],
    }

    monkeypatch.setattr(service, "_load_scan_export_source", lambda *_args, **_kwargs: ([], SimpleNamespace()))
    monkeypatch.setattr(service, "_export_scan_bundle", lambda **_kwargs: (scan_manifest, []))
    monkeypatch.setattr(service, "_export_chart_bundle", lambda **_kwargs: chart_manifest)
    monkeypatch.setattr(service, "_build_breadth_payload", lambda **_kwargs: breadth_payload)
    monkeypatch.setattr(service, "_build_groups_payload", lambda **_kwargs: groups_payload)
    monkeypatch.setattr(service, "_build_home_payload", lambda **_kwargs: home_payload)

    output_dir = tmp_path / "static-data"
    result = service.export(output_dir)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    scan = json.loads((output_dir / "scan" / "manifest.json").read_text(encoding="utf-8"))
    breadth = json.loads((output_dir / "breadth.json").read_text(encoding="utf-8"))
    groups = json.loads((output_dir / "groups.json").read_text(encoding="utf-8"))
    home = json.loads((output_dir / "home.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == STATIC_SITE_SCHEMA_VERSION
    assert manifest["features"]["charts"] is True
    assert manifest["pages"]["scan"]["path"] == "scan/manifest.json"
    assert manifest["assets"]["charts"]["path"] == "charts/index.json"
    assert "themes" not in manifest["features"]
    assert "themes" not in manifest["pages"]
    assert manifest["warnings"] == []
    assert scan["charts"]["path"] == "charts/index.json"
    assert breadth["payload"]["current"]["date"] == "2026-03-31"
    assert groups["payload"]["rankings"]["rankings"][0]["industry_group"] == "Semiconductors"
    assert not (output_dir / "themes").exists()
    assert result.manifest == manifest
    assert result.warnings == ()


def test_export_scan_bundle_chunks_large_result_sets(service_and_session_factory, monkeypatch, tmp_path):
    service, session_factory = service_and_session_factory
    _insert_runs(
        session_factory,
        FeatureRun(
            id=11,
            as_of_date=date(2026, 3, 31),
            run_type="daily_snapshot",
            status="published",
            published_at=datetime(2026, 3, 31, 21, 30, 0),
        ),
        pointer_run_id=11,
    )

    rows = [SimpleNamespace(index=index) for index in range(5)]
    monkeypatch.setattr(export_module, "SCAN_CHUNK_SIZE", 3)
    monkeypatch.setattr(
        export_module.SqlFeatureStoreRepository,
        "query_all_as_scan_results",
        lambda self, *_args, **_kwargs: rows,
    )
    monkeypatch.setattr(
        export_module.SqlFeatureStoreRepository,
        "get_filter_options_for_run",
        lambda self, _run_id: SimpleNamespace(
            ibd_industries=("Semiconductors",),
            gics_sectors=("Technology",),
            ratings=("Strong Buy", "Buy"),
        ),
    )
    monkeypatch.setattr(
        service,
        "_serialize_scan_row",
        lambda row: {
            "symbol": f"SYM{row.index}",
            "composite_score": 100 - row.index,
            "volume": row.volume,
        },
    )
    rows = [
        SimpleNamespace(index=0, volume=150_000_000),
        SimpleNamespace(index=1, volume=90_000_000),
        SimpleNamespace(index=2, volume=120_000_000),
        SimpleNamespace(index=3, volume=80_000_000),
        SimpleNamespace(index=4, volume=300_000_000),
    ]
    monkeypatch.setattr(
        export_module.SqlFeatureStoreRepository,
        "query_all_as_scan_results",
        lambda self, *_args, **_kwargs: rows,
    )

    with session_factory() as db:
        run = db.get(FeatureRun, 11)
        manifest, _serialized = service._export_scan_bundle(  # noqa: SLF001 - intentional unit test coverage
            db=db,
            output_dir=tmp_path,
            generated_at="2026-03-31T22:00:00Z",
            run=run,
        )

    first_chunk = json.loads((tmp_path / "scan" / "chunks" / "chunk-0001.json").read_text(encoding="utf-8"))
    second_chunk = json.loads((tmp_path / "scan" / "chunks" / "chunk-0002.json").read_text(encoding="utf-8"))

    assert manifest["chunk_size"] == 3
    assert manifest["rows_total"] == 5
    assert manifest["default_filters"] == {"minVolume": 100_000_000}
    assert manifest["default_filtered_rows_total"] == 3
    assert [row["symbol"] for row in manifest["initial_rows"]] == ["SYM0", "SYM2", "SYM4"]
    assert [row["symbol"] for row in manifest["preview_rows"]] == ["SYM0", "SYM2", "SYM4"]
    assert [chunk["count"] for chunk in manifest["chunks"]] == [3, 2]
    assert [row["symbol"] for row in first_chunk["rows"]] == ["SYM0", "SYM1", "SYM2"]
    assert [row["symbol"] for row in second_chunk["rows"]] == ["SYM3", "SYM4"]


def test_export_marks_optional_sections_unavailable_without_aborting(
    service_and_session_factory,
    monkeypatch,
    tmp_path,
):
    service, session_factory = service_and_session_factory
    _insert_runs(
        session_factory,
        FeatureRun(
            id=12,
            as_of_date=date(2026, 4, 2),
            run_type="daily_snapshot",
            status="published",
            published_at=datetime(2026, 4, 2, 21, 30, 0),
        ),
        pointer_run_id=12,
    )

    scan_manifest = {
        "schema_version": "static-scan-v1",
        "generated_at": "2026-04-02T22:00:00Z",
        "as_of_date": "2026-04-02",
        "run_id": 12,
        "rows_total": 2,
        "default_filtered_rows_total": 1,
        "chunks": [{"path": "scan/chunks/chunk-0001.json", "count": 2}],
        "preview_rows": [{"symbol": "NVDA", "composite_score": 97.5}],
    }
    chart_manifest = {
        "path": "charts/index.json",
        "limit": 200,
        "symbols_total": 1,
        "available": True,
        "skipped_symbols": [],
    }

    monkeypatch.setattr(service, "_load_scan_export_source", lambda *_args, **_kwargs: ([], SimpleNamespace()))
    monkeypatch.setattr(service, "_export_scan_bundle", lambda **_kwargs: (scan_manifest, []))
    monkeypatch.setattr(service, "_export_chart_bundle", lambda **_kwargs: chart_manifest)
    monkeypatch.setattr(
        service,
        "_build_breadth_payload",
        lambda **_kwargs: (_ for _ in ()).throw(
            StaticSiteSectionUnavailableError(
                section="breadth",
                reason="No breadth snapshot is available for static-site export date 2026-04-02 (latest snapshot date: none).",
            )
        ),
    )
    monkeypatch.setattr(
        service,
        "_build_groups_payload",
        lambda **_kwargs: (_ for _ in ()).throw(
            StaticSiteSectionUnavailableError(
                section="groups",
                reason="No group rankings are available for static-site export date 2026-04-02.",
            )
        ),
    )

    output_dir = tmp_path / "static-data"
    result = service.export(output_dir)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    breadth = json.loads((output_dir / "breadth.json").read_text(encoding="utf-8"))
    groups = json.loads((output_dir / "groups.json").read_text(encoding="utf-8"))
    home = json.loads((output_dir / "home.json").read_text(encoding="utf-8"))

    assert result.manifest == manifest
    assert manifest["features"]["breadth"] is False
    assert manifest["features"]["groups"] is False
    assert len(manifest["warnings"]) == 2
    assert breadth["available"] is False
    assert breadth["expected_as_of_date"] == "2026-04-02"
    assert "No breadth snapshot is available" in breadth["message"]
    assert breadth["payload"] == {}
    assert groups["available"] is False
    assert groups["expected_as_of_date"] == "2026-04-02"
    assert "No group rankings are available" in groups["message"]
    assert groups["payload"] == {}
    assert home["freshness"]["breadth_latest_date"] is None
    assert home["freshness"]["groups_latest_date"] is None
    assert home["top_groups"] == []


def test_build_breadth_payload_requires_target_date(service_and_session_factory, monkeypatch):
    service, _session_factory = service_and_session_factory

    monkeypatch.setattr(
        service._ui_snapshot_service,
        "publish_breadth_bootstrap",
        lambda: SimpleNamespace(
            to_dict=lambda: {
                "published_at": "2026-04-02T22:00:00Z",
                "source_revision": "breadth:2026-04-02",
                "payload": {"current": {"date": "2026-04-01"}},
            }
        ),
    )

    with pytest.raises(StaticSiteSectionUnavailableError, match="No breadth snapshot is available"):
        service._build_breadth_payload(  # noqa: SLF001 - intentional unit coverage
            generated_at="2026-04-02T22:00:00Z",
            expected_as_of_date=date(2026, 4, 2),
        )


def test_build_groups_payload_requires_target_date(service_and_session_factory, monkeypatch):
    service, session_factory = service_and_session_factory

    rankings_calls: list[tuple[int, date | None]] = []
    movers_calls: list[tuple[str, int, date | None]] = []
    fake_service = SimpleNamespace(
        get_current_rankings=lambda db, limit=197, calculation_date=None: rankings_calls.append((limit, calculation_date)) or [],
        get_rank_movers=lambda db, period="1w", limit=10, calculation_date=None: movers_calls.append((period, limit, calculation_date)) or {"period": period, "gainers": [], "losers": []},
    )
    monkeypatch.setattr(export_module, "get_group_rank_service", lambda: fake_service)

    with session_factory() as db, pytest.raises(StaticSiteSectionUnavailableError, match="No group rankings are available"):
        service._build_groups_payload(  # noqa: SLF001 - intentional unit coverage
            db=db,
            generated_at="2026-04-02T22:00:00Z",
            expected_as_of_date=date(2026, 4, 2),
        )

    assert rankings_calls == [(197, date(2026, 4, 2))]
    assert movers_calls == []


def test_export_chart_bundle_writes_top_ranked_payloads_with_sidebar_metadata(
    service_and_session_factory,
    tmp_path,
):
    service, session_factory = service_and_session_factory
    _insert_runs(
        session_factory,
        FeatureRun(
            id=17,
            as_of_date=date(2026, 4, 2),
            run_type="daily_snapshot",
            status="published",
            published_at=datetime(2026, 4, 2, 21, 30, 0),
        ),
        pointer_run_id=17,
    )

    def make_price_frame(closes: list[float]) -> pd.DataFrame:
        dates = pd.date_range("2026-03-28", periods=len(closes), freq="D")
        return pd.DataFrame(
            {
                "Open": closes,
                "High": [close + 1 for close in closes],
                "Low": [close - 1 for close in closes],
                "Close": closes,
                "Volume": [1000000 + index for index, _ in enumerate(closes)],
            },
            index=dates,
        )

    service._price_cache = SimpleNamespace(
        get_many_cached_only=lambda symbols, period="2y": {
            "NVDA": make_price_frame([101.0, 102.5, 103.0]),
            "MSFT": make_price_frame([201.0, 202.0, 204.0]),
        }
    )
    service._fundamentals_cache = SimpleNamespace(
        get_many_cached_only=lambda symbols: {
            "NVDA": {"symbol": "NVDA", "description": "AI chip leader", "pe_ratio": 45.2},
            "MSFT": {"symbol": "MSFT", "description": "Enterprise software", "pe_ratio": 31.4},
        }
    )

    rows = [
        SimpleNamespace(
            symbol="NVDA",
            composite_score=97.5,
            rating="Strong Buy",
            current_price=103.0,
            screeners_run=["minervini", "setup_engine"],
            extended_fields={
                "company_name": "NVIDIA Corporation",
                "ibd_industry_group": "Semiconductors",
                "ibd_group_rank": 1,
                "gics_sector": "Technology",
                "gics_industry": "Semiconductors",
                "adr_percent": 3.8,
                "eps_rating": 94,
                "rs_rating": 96.0,
                "rs_rating_1m": 95.0,
                "rs_rating_3m": 94.0,
                "rs_rating_12m": 97.0,
                "rs_trend": 1,
                "stage": 2,
                "market_cap": 3000000000000,
                "ma_alignment": True,
                "passes_template": True,
                "eps_growth_qq": 62.0,
                "sales_growth_qq": 44.0,
            },
        ),
        SimpleNamespace(
            symbol="MSFT",
            composite_score=92.1,
            rating="Buy",
            current_price=204.0,
            screeners_run=["minervini"],
            extended_fields={
                "company_name": "Microsoft Corporation",
                "ibd_industry_group": "Software",
                "ibd_group_rank": 7,
                "gics_sector": "Technology",
                "gics_industry": "Software",
                "adr_percent": 2.1,
                "eps_rating": 88,
                "rs_rating": 91.0,
                "rs_rating_1m": 90.0,
                "rs_rating_3m": 89.0,
                "rs_rating_12m": 93.0,
                "rs_trend": 1,
                "stage": 2,
                "market_cap": 2800000000000,
                "ma_alignment": True,
                "passes_template": True,
                "eps_growth_qq": 18.0,
                "sales_growth_qq": 13.0,
            },
        ),
    ]

    with session_factory() as db:
        run = db.get(FeatureRun, 17)
        manifest = service._export_chart_bundle(  # noqa: SLF001 - intentional unit coverage
            output_dir=tmp_path,
            generated_at="2026-04-02T22:00:00Z",
            run=run,
            rows=rows,
        )

    index_payload = json.loads((tmp_path / "charts" / "index.json").read_text(encoding="utf-8"))
    nvda_payload = json.loads((tmp_path / "charts" / "NVDA.json").read_text(encoding="utf-8"))

    assert manifest["available"] is True
    assert manifest["symbols_total"] == 2
    assert index_payload["symbols"][0]["symbol"] == "NVDA"
    assert index_payload["symbols"][0]["path"] == "charts/NVDA.json"
    assert nvda_payload["stock_data"]["company_name"] == "NVIDIA Corporation"
    assert nvda_payload["stock_data"]["ibd_group_rank"] == 1
    assert nvda_payload["fundamentals"]["description"] == "AI chip leader"
    assert nvda_payload["bars"][-1]["close"] == 103.0


def test_export_chart_bundle_backfills_past_skipped_symbols_to_fill_limit(
    service_and_session_factory,
    monkeypatch,
    tmp_path,
):
    service, session_factory = service_and_session_factory
    _insert_runs(
        session_factory,
        FeatureRun(
            id=18,
            as_of_date=date(2026, 4, 2),
            run_type="daily_snapshot",
            status="published",
            published_at=datetime(2026, 4, 2, 21, 30, 0),
        ),
        pointer_run_id=18,
    )

    monkeypatch.setattr(export_module, "STATIC_CHART_LIMIT", 2)
    monkeypatch.setattr(export_module, "STATIC_CHART_LOOKUP_BATCH_SIZE", 2)

    def make_price_frame(close: float) -> pd.DataFrame:
        dates = pd.date_range("2026-03-28", periods=2, freq="D")
        return pd.DataFrame(
            {
                "Open": [close - 1, close],
                "High": [close, close + 1],
                "Low": [close - 2, close - 1],
                "Close": [close - 0.5, close],
                "Volume": [1000000, 1100000],
            },
            index=dates,
        )

    service._price_cache = SimpleNamespace(
        get_many_cached_only=lambda symbols, period="2y": {
            "NVDA": None,
            "MSFT": make_price_frame(204.0),
            "AAPL": make_price_frame(175.0),
        }
    )
    service._fundamentals_cache = SimpleNamespace(
        get_many_cached_only=lambda symbols: {symbol: {"symbol": symbol} for symbol in symbols}
    )

    rows = [
        SimpleNamespace(symbol="NVDA", composite_score=99.0, rating="Strong Buy", current_price=0.0, screeners_run=[], extended_fields={}),
        SimpleNamespace(symbol="MSFT", composite_score=98.0, rating="Buy", current_price=0.0, screeners_run=[], extended_fields={}),
        SimpleNamespace(symbol="AAPL", composite_score=97.0, rating="Buy", current_price=0.0, screeners_run=[], extended_fields={}),
    ]

    with session_factory() as db:
        run = db.get(FeatureRun, 18)
        manifest = service._export_chart_bundle(  # noqa: SLF001 - intentional unit coverage
            output_dir=tmp_path,
            generated_at="2026-04-02T22:00:00Z",
            run=run,
            rows=rows,
        )

    index_payload = json.loads((tmp_path / "charts" / "index.json").read_text(encoding="utf-8"))

    assert manifest["symbols_total"] == 2
    assert [entry["symbol"] for entry in index_payload["symbols"]] == ["MSFT", "AAPL"]
    assert [entry["rank"] for entry in index_payload["symbols"]] == [2, 3]
    assert index_payload["skipped_symbols"] == ["NVDA"]


def test_build_key_markets_skips_change_when_latest_close_is_null(service_and_session_factory):
    service, session_factory = service_and_session_factory

    with session_factory() as db:
        db.add_all(
            [
                StockPrice(symbol="SPY", date=date(2026, 3, 30), close=500.0),
                StockPrice(symbol="SPY", date=date(2026, 3, 31), close=None),
            ]
        )
        db.commit()

        markets = service._build_key_markets(db)  # noqa: SLF001 - intentional unit test coverage

    spy = next(item for item in markets if item["symbol"] == "SPY")
    assert spy["latest_close"] is None
    assert spy["change_1d"] is None
