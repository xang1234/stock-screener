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
    STATIC_MARKET_METADATA_FILENAME,
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


def _insert_runs(
    session_factory,
    *runs: FeatureRun,
    pointer_run_id: int | None = None,
    pointer_key: str = "latest_published",
) -> None:
    with session_factory() as db:
        db.add_all(runs)
        if pointer_run_id is not None:
            db.add(FeatureRunPointer(key=pointer_key, run_id=pointer_run_id))
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


def test_get_latest_published_run_ignores_market_pointer_for_wrong_market(
    service_and_session_factory,
):
    service, session_factory = service_and_session_factory
    _insert_runs(
        session_factory,
        FeatureRun(
            id=4,
            as_of_date=date(2026, 3, 30),
            run_type="daily_snapshot",
            status="published",
            published_at=datetime(2026, 3, 30, 21, 30, 0),
            config_json={"universe": {"market": "US"}},
        ),
        FeatureRun(
            id=5,
            as_of_date=date(2026, 3, 31),
            run_type="daily_snapshot",
            status="published",
            published_at=datetime(2026, 3, 31, 21, 30, 0),
            config_json={"universe": {"market": "HK"}},
        ),
        pointer_run_id=5,
        pointer_key="latest_published_market:US",
    )

    with session_factory() as db:
        run = service._get_latest_published_run(  # noqa: SLF001 - intentional unit test coverage
            db,
            market="US",
        )

    assert run is not None
    assert run.id == 4


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
            config_json={"universe": {"market": "US"}},
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
    scan = json.loads((output_dir / "markets" / "us" / "scan" / "manifest.json").read_text(encoding="utf-8"))
    breadth = json.loads((output_dir / "markets" / "us" / "breadth.json").read_text(encoding="utf-8"))
    groups = json.loads((output_dir / "markets" / "us" / "groups.json").read_text(encoding="utf-8"))
    home = json.loads((output_dir / "markets" / "us" / "home.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == STATIC_SITE_SCHEMA_VERSION
    assert manifest["default_market"] == "US"
    assert manifest["supported_markets"] == ["US"]
    assert manifest["features"]["charts"] is True
    assert manifest["pages"]["scan"]["path"] == "markets/us/scan/manifest.json"
    assert manifest["assets"]["charts"]["path"] == "charts/index.json"
    assert manifest["markets"]["US"]["pages"]["scan"]["path"] == "markets/us/scan/manifest.json"
    assert manifest["markets"]["US"]["assets"]["charts"]["path"] == "charts/index.json"
    assert "themes" not in manifest["features"]
    assert "themes" not in manifest["pages"]
    assert manifest["warnings"] == []
    assert scan["charts"]["path"] == "charts/index.json"
    assert breadth["payload"]["current"]["date"] == "2026-03-31"
    assert groups["payload"]["rankings"]["rankings"][0]["industry_group"] == "Semiconductors"
    assert not (output_dir / "themes").exists()
    assert result.manifest == manifest
    assert result.warnings == ()


def test_export_writes_india_market_bundle_and_root_manifest(
    service_and_session_factory,
    monkeypatch,
    tmp_path,
):
    service, session_factory = service_and_session_factory
    _insert_runs(
        session_factory,
        FeatureRun(
            id=70,
            as_of_date=date(2026, 4, 4),
            run_type="daily_snapshot",
            status="published",
            published_at=datetime(2026, 4, 4, 21, 30, 0),
            config_json={"universe": {"market": "IN"}},
        ),
        pointer_run_id=70,
        pointer_key="latest_published_market:IN",
    )

    scan_manifest = {
        "schema_version": "static-scan-v1",
        "generated_at": "2026-04-04T22:00:00Z",
        "as_of_date": "2026-04-04",
        "run_id": 70,
        "rows_total": 2,
        "default_filtered_rows_total": 1,
        "chunks": [{"path": "markets/in/scan/chunks/chunk-0001.json", "count": 2}],
        "preview_rows": [{"symbol": "RELIANCE.NS", "composite_score": 97.5}],
    }
    breadth_payload = {
        "schema_version": STATIC_SITE_SCHEMA_VERSION,
        "generated_at": "2026-04-04T22:00:00Z",
        "available": True,
        "payload": {"current": {"date": "2026-04-04"}},
    }
    groups_payload = {
        "schema_version": STATIC_SITE_SCHEMA_VERSION,
        "generated_at": "2026-04-04T22:00:00Z",
        "available": True,
        "payload": {"rankings": {"date": "2026-04-04", "rankings": []}},
    }
    home_payload = {
        "schema_version": STATIC_SITE_SCHEMA_VERSION,
        "generated_at": "2026-04-04T22:00:00Z",
        "as_of_date": "2026-04-04",
        "freshness": {"scan_run_id": 70},
        "key_markets": [{"symbol": "^NSEI", "display_name": "Nifty 50"}],
        "scan_summary": {"top_results": [{"symbol": "RELIANCE.NS"}]},
        "top_groups": [],
    }
    chart_manifest = {
        "path": "markets/in/charts/index.json",
        "limit": 200,
        "symbols_total": 2,
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
    result = service.export(output_dir, markets=("IN",))

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (output_dir / "markets" / "in" / STATIC_MARKET_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    scan = json.loads((output_dir / "markets" / "in" / "scan" / "manifest.json").read_text(encoding="utf-8"))
    breadth = json.loads((output_dir / "markets" / "in" / "breadth.json").read_text(encoding="utf-8"))
    groups = json.loads((output_dir / "markets" / "in" / "groups.json").read_text(encoding="utf-8"))
    home = json.loads((output_dir / "markets" / "in" / "home.json").read_text(encoding="utf-8"))

    assert result.manifest == manifest
    assert manifest["supported_markets"] == ["IN"]
    assert manifest["markets"]["IN"]["display_name"] == "India"
    assert manifest["markets"]["IN"]["pages"]["scan"]["path"] == "markets/in/scan/manifest.json"
    assert manifest["markets"]["IN"]["pages"]["home"]["path"] == "markets/in/home.json"
    assert manifest["markets"]["IN"]["assets"]["charts"]["path"] == "markets/in/charts/index.json"
    assert metadata["market"] == "IN"
    assert metadata["entry"]["display_name"] == "India"
    assert scan["preview_rows"][0]["symbol"] == "RELIANCE.NS"
    assert breadth["payload"]["current"]["date"] == "2026-04-04"
    assert groups["payload"]["rankings"]["date"] == "2026-04-04"
    assert home["key_markets"][0]["symbol"] == "^NSEI"


def test_export_can_write_single_market_artifact_without_root_manifest(
    service_and_session_factory,
    monkeypatch,
    tmp_path,
):
    service, session_factory = service_and_session_factory
    _insert_runs(
        session_factory,
        FeatureRun(
            id=8,
            as_of_date=date(2026, 3, 31),
            run_type="daily_snapshot",
            status="published",
            published_at=datetime(2026, 3, 31, 21, 30, 0),
            config_json={"universe": {"market": "HK"}},
        ),
        pointer_run_id=8,
        pointer_key="latest_published_market:HK",
    )

    scan_manifest = {
        "schema_version": "static-scan-v1",
        "generated_at": "2026-03-31T22:00:00Z",
        "as_of_date": "2026-03-31",
        "run_id": 8,
        "rows_total": 1,
        "default_filtered_rows_total": 1,
        "chunks": [{"path": "markets/hk/scan/chunks/chunk-0001.json", "count": 1}],
        "preview_rows": [{"symbol": "0700.HK", "composite_score": 97.5}],
    }
    chart_manifest = {
        "path": "markets/hk/charts/index.json",
        "limit": 200,
        "symbols_total": 1,
        "available": True,
        "skipped_symbols": [],
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
        "payload": {"rankings": {"date": "2026-03-31", "rankings": []}},
    }
    home_payload = {
        "schema_version": STATIC_SITE_SCHEMA_VERSION,
        "generated_at": "2026-03-31T22:00:00Z",
        "as_of_date": "2026-03-31",
        "freshness": {"scan_run_id": 8},
        "key_markets": [],
        "scan_summary": {"top_results": [{"symbol": "0700.HK"}]},
        "top_groups": [],
    }

    monkeypatch.setattr(service, "_load_scan_export_source", lambda *_args, **_kwargs: ([], SimpleNamespace()))
    monkeypatch.setattr(service, "_export_scan_bundle", lambda **_kwargs: (scan_manifest, []))
    monkeypatch.setattr(service, "_export_chart_bundle", lambda **_kwargs: chart_manifest)
    monkeypatch.setattr(service, "_build_breadth_payload", lambda **_kwargs: breadth_payload)
    monkeypatch.setattr(service, "_build_groups_payload", lambda **_kwargs: groups_payload)
    monkeypatch.setattr(service, "_build_home_payload", lambda **_kwargs: home_payload)

    output_dir = tmp_path / "market-artifact"
    result = service.export(output_dir, markets=("HK",), write_manifest=False)

    assert not (output_dir / "manifest.json").exists()
    metadata = json.loads(
        (output_dir / "markets" / "hk" / STATIC_MARKET_METADATA_FILENAME).read_text(encoding="utf-8")
    )

    assert result.manifest["supported_markets"] == ["HK"]
    assert metadata["market"] == "HK"
    assert metadata["entry"]["pages"]["scan"]["path"] == "markets/hk/scan/manifest.json"
    assert metadata["entry"]["assets"]["charts"]["path"] == "markets/hk/charts/index.json"
    assert metadata["warnings"] == []


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


def test_export_scan_bundle_prioritizes_full_rows_before_ipo_weighted_and_listing_only(
    service_and_session_factory,
    monkeypatch,
    tmp_path,
):
    service, session_factory = service_and_session_factory
    _insert_runs(
        session_factory,
        FeatureRun(
            id=12,
            as_of_date=date(2026, 3, 31),
            run_type="daily_snapshot",
            status="published",
            published_at=datetime(2026, 3, 31, 21, 30, 0),
        ),
        pointer_run_id=12,
    )

    rows = [
        SimpleNamespace(symbol="IPO95", scan_mode="ipo_weighted", composite_score=95.0, volume=150_000_000),
        SimpleNamespace(symbol="FULL80", scan_mode="full", composite_score=80.0, volume=150_000_000),
        SimpleNamespace(symbol="NEW1", scan_mode="listing_only", composite_score=None, volume=None),
        SimpleNamespace(symbol="FULL70", scan_mode="full", composite_score=70.0, volume=150_000_000),
    ]
    monkeypatch.setattr(
        export_module.SqlFeatureStoreRepository,
        "query_all_as_scan_results",
        lambda self, *_args, **_kwargs: rows,
    )
    monkeypatch.setattr(
        export_module.SqlFeatureStoreRepository,
        "get_filter_options_for_run",
        lambda self, _run_id: SimpleNamespace(
            ibd_industries=(),
            gics_sectors=(),
            ratings=("Strong Buy", "Buy", "Insufficient Data"),
        ),
    )
    monkeypatch.setattr(
        service,
        "_serialize_scan_row",
        lambda row: {
            "symbol": row.symbol,
            "scan_mode": row.scan_mode,
            "data_status": "complete" if row.scan_mode == "full" else "insufficient_history",
            "composite_score": row.composite_score,
            "volume": row.volume,
        },
    )

    with session_factory() as db:
        run = db.get(FeatureRun, 12)
        manifest, _serialized = service._export_scan_bundle(
            db=db,
            output_dir=tmp_path,
            generated_at="2026-03-31T22:00:00Z",
            run=run,
        )

    assert [row["symbol"] for row in manifest["initial_rows"]] == ["FULL80", "FULL70", "IPO95"]
    chunk = json.loads((tmp_path / "scan" / "chunks" / "chunk-0001.json").read_text(encoding="utf-8"))
    assert [row["symbol"] for row in chunk["rows"]] == ["FULL80", "FULL70", "IPO95", "NEW1"]


def test_static_scan_mode_sort_priority_matches_frontend_unknown_mode_fallback(
    service_and_session_factory,
):
    service, _session_factory = service_and_session_factory

    assert service._static_scan_mode_sort_priority({"scan_mode": None}) == 0  # noqa: SLF001
    assert service._static_scan_mode_sort_priority({"scan_mode": "full"}) == 0  # noqa: SLF001
    assert service._static_scan_mode_sort_priority({"scan_mode": "ipo_weighted"}) == 1  # noqa: SLF001
    assert service._static_scan_mode_sort_priority({"scan_mode": "listing_only"}) == 2  # noqa: SLF001
    assert service._static_scan_mode_sort_priority({"scan_mode": "mystery_mode"}) == 3  # noqa: SLF001


def test_serialize_scan_row_preserves_young_ipo_partial_metrics(service_and_session_factory):
    service, _session_factory = service_and_session_factory
    row = SimpleNamespace(
        symbol="NEWIPO",
        company_name=None,
        composite_score=None,
        rating="Insufficient Data",
        current_price=10.0,
        screeners_run=[],
        extended_fields={
            "data_status": "insufficient_history",
            "is_scannable": False,
            "scan_mode": "listing_only",
            "history_bars": 45,
            "price_sparkline_data": [1.0, 1.01, 1.02],
            "price_trend": 1,
            "price_change_1d": 2.5,
            "rs_sparkline_data": [1.0, 0.99, 1.03],
            "rs_trend": 1,
            "adr_percent": 10.0,
            "rs_rating_1m": 50.0,
            "rs_rating": None,
            "rs_rating_3m": None,
            "rs_rating_12m": None,
        },
    )

    payload = service._serialize_scan_row(row)  # noqa: SLF001 - intentional unit coverage

    assert payload["composite_score"] is None
    assert payload["scan_mode"] == "listing_only"
    assert payload["price_sparkline_data"] == [1.0, 1.01, 1.02]
    assert payload["rs_sparkline_data"] == [1.0, 0.99, 1.03]
    assert payload["price_change_1d"] == 2.5
    assert payload["adr_percent"] == 10.0
    assert payload["rs_rating_1m"] == 50.0
    assert payload["rs_rating"] is None


def test_combine_market_artifacts_builds_manifest_from_subset(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    us_dir = artifacts_dir / "job-us" / "markets" / "us"
    hk_dir = artifacts_dir / "job-hk" / "markets" / "hk"
    us_dir.mkdir(parents=True)
    hk_dir.mkdir(parents=True)

    (us_dir / "scan").mkdir()
    (hk_dir / "scan").mkdir()
    (us_dir / "scan" / "manifest.json").write_text('{"ok": true}\n', encoding="utf-8")
    (hk_dir / "scan" / "manifest.json").write_text('{"ok": true}\n', encoding="utf-8")

    us_entry = {
        "market": "US",
        "display_name": "United States",
        "as_of_date": "2026-04-04",
        "features": {"scan": True, "breadth": True, "groups": True, "charts": True},
        "pages": {"home": {"path": "markets/us/home.json"}, "scan": {"path": "markets/us/scan/manifest.json"}},
        "assets": {"charts": {"path": "markets/us/charts/index.json", "limit": 200, "symbols_total": 1}},
        "freshness": {"scan_run_id": 11},
    }
    hk_entry = {
        "market": "HK",
        "display_name": "Hong Kong",
        "as_of_date": "2026-04-03",
        "features": {"scan": True, "breadth": False, "groups": False, "charts": False},
        "pages": {"home": {"path": "markets/hk/home.json"}, "scan": {"path": "markets/hk/scan/manifest.json"}},
        "assets": {"charts": {"path": "markets/hk/charts/index.json", "limit": 200, "symbols_total": 0}},
        "freshness": {"scan_run_id": 12},
    }

    (us_dir / STATIC_MARKET_METADATA_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": STATIC_SITE_SCHEMA_VERSION,
                "generated_at": "2026-04-04T22:00:00Z",
                "market": "US",
                "entry": us_entry,
                "warnings": ["US local warning"],
            }
        ),
        encoding="utf-8",
    )
    (hk_dir / STATIC_MARKET_METADATA_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": STATIC_SITE_SCHEMA_VERSION,
                "generated_at": "2026-04-04T22:00:00Z",
                "market": "HK",
                "entry": hk_entry,
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "combined"
    result = StaticSiteExportService.combine_market_artifacts(artifacts_dir, output_dir)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert result.manifest == manifest
    assert manifest["default_market"] == "US"
    assert manifest["supported_markets"] == ["US", "HK"]
    assert manifest["markets"]["US"]["pages"]["scan"]["path"] == "markets/us/scan/manifest.json"
    assert manifest["markets"]["HK"]["pages"]["scan"]["path"] == "markets/hk/scan/manifest.json"
    assert (output_dir / "markets" / "us" / "scan" / "manifest.json").exists()
    assert (output_dir / "markets" / "hk" / "scan" / "manifest.json").exists()
    assert "US local warning" in manifest["warnings"]
    assert any("JP" in warning for warning in manifest["warnings"])
    assert any("TW" in warning for warning in manifest["warnings"])


def test_build_manifest_orders_india_between_hk_and_jp():
    market_entries = {
        "US": {
            "market": "US",
            "display_name": "United States",
            "as_of_date": "2026-04-04",
            "features": {"scan": True, "breadth": True, "groups": True, "charts": True},
            "pages": {"scan": {"path": "markets/us/scan/manifest.json"}},
            "assets": {"charts": {"path": "markets/us/charts/index.json"}},
        },
        "IN": {
            "market": "IN",
            "display_name": "India",
            "as_of_date": "2026-04-04",
            "features": {"scan": True, "breadth": True, "groups": False, "charts": True},
            "pages": {"scan": {"path": "markets/in/scan/manifest.json"}},
            "assets": {"charts": {"path": "markets/in/charts/index.json"}},
        },
        "HK": {
            "market": "HK",
            "display_name": "Hong Kong",
            "as_of_date": "2026-04-04",
            "features": {"scan": True, "breadth": False, "groups": False, "charts": False},
            "pages": {"scan": {"path": "markets/hk/scan/manifest.json"}},
            "assets": {"charts": {"path": "markets/hk/charts/index.json"}},
        },
    }

    manifest = StaticSiteExportService._build_manifest(
        market_entries=market_entries,
        generated_at="2026-04-04T22:00:00Z",
        warnings=[],
    )

    assert manifest["supported_markets"] == ["US", "HK", "IN"]
    assert list(manifest["markets"]) == ["US", "HK", "IN"]


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
            config_json={"universe": {"market": "US"}},
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
    breadth = json.loads((output_dir / "markets" / "us" / "breadth.json").read_text(encoding="utf-8"))
    groups = json.loads((output_dir / "markets" / "us" / "groups.json").read_text(encoding="utf-8"))
    home = json.loads((output_dir / "markets" / "us" / "home.json").read_text(encoding="utf-8"))

    assert result.manifest == manifest
    assert manifest["features"]["breadth"] is False
    assert manifest["features"]["groups"] is False
    assert manifest["markets"]["US"]["features"]["breadth"] is False
    assert manifest["markets"]["US"]["features"]["groups"] is False
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
        lambda market="US": SimpleNamespace(
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


def test_build_breadth_payload_serialized_rows_emit_market_benchmark_overlay(
    service_and_session_factory,
    monkeypatch,
):
    service, _session_factory = service_and_session_factory
    idx = pd.date_range("2026-03-01", "2026-04-24", freq="D")

    def history_frame(start_price: float) -> pd.DataFrame:
        closes = [start_price + index for index in range(len(idx))]
        return pd.DataFrame(
            {
                "Open": closes,
                "High": [value + 1 for value in closes],
                "Low": [value - 1 for value in closes],
                "Close": closes,
                "Volume": [1000] * len(idx),
            },
            index=idx,
        )

    monkeypatch.setattr(
        service,
        "_get_market_benchmark_history",
        lambda market, *, period: ("^HSI", history_frame(18000.0)),
    )
    monkeypatch.setattr(
        service,
        "_get_cached_price_histories",
        lambda symbols, *, period: {symbol: history_frame(100.0) for symbol in symbols},
    )

    payload = service._build_breadth_payload(  # noqa: SLF001 - intentional unit coverage
        generated_at="2026-04-24T22:00:00Z",
        expected_as_of_date=date(2026, 4, 24),
        market="HK",
        serialized_rows=[{"symbol": "0700.HK"}],
    )

    breadth_payload = payload["payload"]
    assert breadth_payload["current"]["market"] == "HK"
    assert breadth_payload["summary"]["market"] == "HK"
    assert breadth_payload["benchmark_symbol"] == "^HSI"
    assert breadth_payload["benchmark_overlay"] == breadth_payload["spy_overlay"]
    assert breadth_payload["benchmark_overlay"][-1]["date"] == "2026-04-24"


def test_export_rejects_legacy_unscoped_run_for_market_bundle(
    service_and_session_factory,
    tmp_path,
):
    service, session_factory = service_and_session_factory
    _insert_runs(
        session_factory,
        FeatureRun(
            id=21,
            as_of_date=date(2026, 4, 3),
            run_type="daily_snapshot",
            status="published",
            published_at=datetime(2026, 4, 3, 21, 30, 0),
        ),
        pointer_run_id=21,
    )

    with pytest.raises(RuntimeError, match="No market-scoped published feature runs"):
        service.export(tmp_path / "static-data")


def test_serialize_history_bars_clamps_to_end_date_and_skips_nan_rows(service_and_session_factory):
    service, _session_factory = service_and_session_factory
    frame = pd.DataFrame(
        {
            "Open": [100.0, 101.0, float("nan"), 104.0],
            "High": [101.0, 102.0, 104.0, 105.0],
            "Low": [99.0, 100.0, 102.0, 103.0],
            "Close": [100.5, 101.5, 103.5, 104.5],
            "Volume": [1000, 1100, 1200, 1300],
        },
        index=pd.to_datetime(["2026-03-01", "2026-04-01", "2026-04-02", "2026-04-03"]),
    )

    payload = service._serialize_history_bars(  # noqa: SLF001 - intentional unit test coverage
        frame,
        period_days=10,
        end_date=date(2026, 4, 2),
    )

    assert payload == [
        {
            "date": "2026-04-01",
            "open": 101.0,
            "high": 102.0,
            "low": 100.0,
            "close": 101.5,
            "volume": 1100,
        },
    ]


def test_get_market_run_series_normalizes_market_to_uppercase(service_and_session_factory):
    service, session_factory = service_and_session_factory
    run_us_latest = FeatureRun(
        id=31,
        as_of_date=date(2026, 4, 3),
        run_type="daily_snapshot",
        status="published",
        published_at=datetime(2026, 4, 3, 21, 30, 0),
        config_json={"universe": {"market": "US"}},
    )
    run_us_previous = FeatureRun(
        id=30,
        as_of_date=date(2026, 4, 2),
        run_type="daily_snapshot",
        status="published",
        published_at=datetime(2026, 4, 2, 21, 30, 0),
        config_json={"universe": {"market": "US"}},
    )
    run_hk = FeatureRun(
        id=29,
        as_of_date=date(2026, 4, 1),
        run_type="daily_snapshot",
        status="published",
        published_at=datetime(2026, 4, 1, 21, 30, 0),
        config_json={"universe": {"market": "HK"}},
    )
    _insert_runs(session_factory, run_us_latest, run_us_previous, run_hk)

    with session_factory() as db:
        market_runs = service._get_market_run_series(  # noqa: SLF001 - intentional unit test coverage
            db=db,
            market="us",
            latest_run=run_us_latest,
        )

    assert [run.id for run in market_runs] == [31, 30]


def test_get_market_run_series_deduplicates_same_day_reruns(service_and_session_factory):
    service, session_factory = service_and_session_factory
    latest_us_run = FeatureRun(
        id=41,
        as_of_date=date(2026, 4, 3),
        run_type="daily_snapshot",
        status="published",
        published_at=datetime(2026, 4, 3, 22, 30, 0),
        config_json={"universe": {"market": "US"}},
    )
    rerun_same_day = FeatureRun(
        id=40,
        as_of_date=date(2026, 4, 3),
        run_type="daily_snapshot",
        status="published",
        published_at=datetime(2026, 4, 3, 21, 30, 0),
        config_json={"universe": {"market": "US"}},
    )
    previous_day = FeatureRun(
        id=39,
        as_of_date=date(2026, 4, 2),
        run_type="daily_snapshot",
        status="published",
        published_at=datetime(2026, 4, 2, 21, 30, 0),
        config_json={"universe": {"market": "US"}},
    )
    _insert_runs(session_factory, latest_us_run, rerun_same_day, previous_day)

    with session_factory() as db:
        market_runs = service._get_market_run_series(  # noqa: SLF001 - intentional unit test coverage
            db=db,
            market="US",
            latest_run=latest_us_run,
        )

    assert [run.id for run in market_runs] == [41, 39]


def test_compute_breadth_metrics_uses_full_history_for_shifted_ranges(service_and_session_factory):
    service, _session_factory = service_and_session_factory
    all_dates = pd.date_range("2025-10-01", periods=200, freq="D")
    canonical_dates = [ts.date() for ts in all_dates[-120:]]
    closes = [100.0 * (1.02 ** index) for index, _ in enumerate(all_dates)]
    price_data = {
        "AAA": pd.DataFrame({"Close": closes}, index=all_dates),
    }

    metrics = service._compute_breadth_metrics_by_date(  # noqa: SLF001 - intentional unit test coverage
        canonical_dates,
        price_data,
    )

    oldest_history_date = canonical_dates[30]
    assert metrics[oldest_history_date]["stocks_up_25pct_quarter"] == 1
    assert metrics[oldest_history_date]["stocks_up_25pct_month"] == 1


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


def test_export_chart_bundle_uses_sorted_scan_order_for_primary_chart_selection(
    service_and_session_factory,
    monkeypatch,
    tmp_path,
):
    service, session_factory = service_and_session_factory
    _insert_runs(
        session_factory,
        FeatureRun(
            id=21,
            as_of_date=date(2026, 4, 2),
            run_type="daily_snapshot",
            status="published",
            published_at=datetime(2026, 4, 2, 21, 30, 0),
        ),
        pointer_run_id=21,
    )

    monkeypatch.setattr(export_module, "STATIC_CHART_LIMIT", 1)
    monkeypatch.setattr(export_module, "STATIC_CHART_LOOKUP_BATCH_SIZE", 2)

    service._price_cache = SimpleNamespace(
        get_many_cached_only=lambda symbols, period="2y": {
            symbol: _make_chart_price_frame(100.0 + index)
            for index, symbol in enumerate(symbols)
        }
    )
    service._fundamentals_cache = SimpleNamespace(
        get_many_cached_only=lambda symbols: {s: {"symbol": s} for s in symbols}
    )

    rows = [
        SimpleNamespace(
            symbol="LOW",
            composite_score=10.0,
            rating="Watch",
            current_price=90.0,
            screeners_run=[],
            extended_fields={},
        ),
        SimpleNamespace(
            symbol="HIGH",
            composite_score=99.0,
            rating="Strong Buy",
            current_price=120.0,
            screeners_run=[],
            extended_fields={},
        ),
    ]
    serialized_rows = [
        {"symbol": "HIGH", "composite_score": 99.0, "scan_mode": "full"},
        {"symbol": "LOW", "composite_score": 10.0, "scan_mode": "full"},
    ]

    with session_factory() as db:
        run = db.get(FeatureRun, 21)
        manifest = service._export_chart_bundle(  # noqa: SLF001 - intentional unit coverage
            output_dir=tmp_path,
            generated_at="2026-04-02T22:00:00Z",
            run=run,
            rows=rows,
            serialized_rows=serialized_rows,
        )

    index_payload = json.loads((tmp_path / "charts" / "index.json").read_text(encoding="utf-8"))

    assert manifest["symbols_total"] == 1
    assert [entry["symbol"] for entry in index_payload["symbols"]] == ["HIGH"]
    assert [entry["rank"] for entry in index_payload["symbols"]] == [1]


def _make_chart_price_frame(close: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range("2026-03-28", periods=2, freq="D")
    return pd.DataFrame(
        {
            "Open": [close - 1, close],
            "High": [close, close + 1],
            "Low": [close - 2, close - 1],
            "Close": [close - 0.5, close],
            "Volume": [1_000_000, 1_100_000],
        },
        index=dates,
    )


def test_export_chart_bundle_expands_coverage_for_preset_screens(
    service_and_session_factory,
    monkeypatch,
    tmp_path,
):
    """Pass 2 should export charts for preset top-N matches that fall
    outside the composite-score top-N covered by Pass 1, so selective or
    orthogonally-ranked presets (e.g. 97 Club) get full chart coverage.
    """
    service, session_factory = service_and_session_factory
    _insert_runs(
        session_factory,
        FeatureRun(
            id=19,
            as_of_date=date(2026, 4, 2),
            run_type="daily_snapshot",
            status="published",
            published_at=datetime(2026, 4, 2, 21, 30, 0),
        ),
        pointer_run_id=19,
    )

    # Tight Pass 1 limit forces the GAIN* rows to rely on Pass 2 expansion.
    monkeypatch.setattr(export_module, "STATIC_CHART_LIMIT", 1)
    monkeypatch.setattr(export_module, "STATIC_CHART_LOOKUP_BATCH_SIZE", 5)
    monkeypatch.setattr(export_module, "STATIC_CHART_PRESET_TOP_N", 5)

    service._price_cache = SimpleNamespace(
        get_many_cached_only=lambda symbols, period="2y": {
            symbol: _make_chart_price_frame(100.0 + i)
            for i, symbol in enumerate(symbols)
        }
    )
    service._fundamentals_cache = SimpleNamespace(
        get_many_cached_only=lambda symbols: {s: {"symbol": s} for s in symbols}
    )

    # NVDA has the highest composite score but a small perfDay (won't match
    # the 4% Gainers preset). GAIN* rows invert that — they rely on Pass 2.
    rows = [
        SimpleNamespace(
            symbol="NVDA",
            composite_score=99.0,
            rating="Strong Buy",
            current_price=100.0,
            screeners_run=[],
            extended_fields={"price_change_1d": 2.0},
        ),
        SimpleNamespace(
            symbol="GAIN1",
            composite_score=50.0,
            rating="Buy",
            current_price=101.0,
            screeners_run=[],
            extended_fields={"price_change_1d": 12.0},
        ),
        SimpleNamespace(
            symbol="GAIN2",
            composite_score=49.0,
            rating="Buy",
            current_price=102.0,
            screeners_run=[],
            extended_fields={"price_change_1d": 9.0},
        ),
        SimpleNamespace(
            symbol="GAIN3",
            composite_score=48.0,
            rating="Hold",
            current_price=103.0,
            screeners_run=[],
            extended_fields={"price_change_1d": 7.0},
        ),
    ]

    serialized_rows = [
        {"symbol": "NVDA", "composite_score": 99.0, "price_change_1d": 2.0},
        {"symbol": "GAIN1", "composite_score": 50.0, "price_change_1d": 12.0},
        {"symbol": "GAIN2", "composite_score": 49.0, "price_change_1d": 9.0},
        {"symbol": "GAIN3", "composite_score": 48.0, "price_change_1d": 7.0},
    ]

    with session_factory() as db:
        run = db.get(FeatureRun, 19)
        manifest = service._export_chart_bundle(  # noqa: SLF001 - intentional unit coverage
            output_dir=tmp_path,
            generated_at="2026-04-02T22:00:00Z",
            run=run,
            rows=rows,
            serialized_rows=serialized_rows,
        )

    index_payload = json.loads((tmp_path / "charts" / "index.json").read_text(encoding="utf-8"))
    symbols_in_order = [entry["symbol"] for entry in index_payload["symbols"]]

    assert symbols_in_order[0] == "NVDA"
    assert index_payload["symbols"][0]["rank"] == 1
    assert set(symbols_in_order[1:]) == {"GAIN1", "GAIN2", "GAIN3"}
    for entry in index_payload["symbols"][1:]:
        assert entry["rank"] is None

    assert manifest["symbols_total"] == 4
    assert manifest["available"] is True
    gain1_payload = json.loads((tmp_path / "charts" / "GAIN1.json").read_text(encoding="utf-8"))
    assert gain1_payload["rank"] is None
    assert gain1_payload["symbol"] == "GAIN1"


def test_export_chart_bundle_skips_preset_symbols_without_cached_prices(
    service_and_session_factory,
    monkeypatch,
    tmp_path,
):
    """Symbols skipped in Pass 1 for lack of cached prices should not be
    re-attempted in Pass 2 (they're already tracked in skipped_symbols).
    """
    service, session_factory = service_and_session_factory
    _insert_runs(
        session_factory,
        FeatureRun(
            id=20,
            as_of_date=date(2026, 4, 2),
            run_type="daily_snapshot",
            status="published",
            published_at=datetime(2026, 4, 2, 21, 30, 0),
        ),
        pointer_run_id=20,
    )

    monkeypatch.setattr(export_module, "STATIC_CHART_LIMIT", 5)
    monkeypatch.setattr(export_module, "STATIC_CHART_LOOKUP_BATCH_SIZE", 5)
    monkeypatch.setattr(export_module, "STATIC_CHART_PRESET_TOP_N", 5)

    # NOCACHE has no cached prices but matches the 4% Gainers preset —
    # Pass 2 must honor the skipped_symbols exclusion and not re-attempt.
    service._price_cache = SimpleNamespace(
        get_many_cached_only=lambda symbols, period="2y": {
            "NVDA": _make_chart_price_frame(),
            "NOCACHE": None,
        }
    )
    service._fundamentals_cache = SimpleNamespace(
        get_many_cached_only=lambda symbols: {s: {"symbol": s} for s in symbols}
    )

    rows = [
        SimpleNamespace(
            symbol="NVDA",
            composite_score=99.0,
            rating="Strong Buy",
            current_price=100.0,
            screeners_run=[],
            extended_fields={"price_change_1d": 1.0},
        ),
        SimpleNamespace(
            symbol="NOCACHE",
            composite_score=10.0,
            rating="Hold",
            current_price=50.0,
            screeners_run=[],
            extended_fields={"price_change_1d": 15.0},
        ),
    ]
    serialized_rows = [
        {"symbol": "NVDA", "composite_score": 99.0, "price_change_1d": 1.0},
        {"symbol": "NOCACHE", "composite_score": 10.0, "price_change_1d": 15.0},
    ]

    with session_factory() as db:
        run = db.get(FeatureRun, 20)
        manifest = service._export_chart_bundle(  # noqa: SLF001 - intentional unit coverage
            output_dir=tmp_path,
            generated_at="2026-04-02T22:00:00Z",
            run=run,
            rows=rows,
            serialized_rows=serialized_rows,
        )

    index_payload = json.loads((tmp_path / "charts" / "index.json").read_text(encoding="utf-8"))
    exported = [e["symbol"] for e in index_payload["symbols"]]

    assert exported == ["NVDA"]
    assert index_payload["skipped_symbols"] == ["NOCACHE"]
    assert manifest["symbols_total"] == 1


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


def test_build_key_markets_includes_india_defaults(service_and_session_factory, monkeypatch):
    service, _session_factory = service_and_session_factory
    history = [
        {"date": "2026-03-30", "close": 100.0},
        {"date": "2026-03-31", "close": 101.0},
    ]
    monkeypatch.setattr(service, "_get_symbol_price_history", lambda symbol, period="6mo": [symbol, period])
    monkeypatch.setattr(service, "_serialize_close_history", lambda history_payload, days=30: history)

    markets = service._build_key_markets("IN")  # noqa: SLF001 - intentional unit test coverage

    assert [item["symbol"] for item in markets] == ["^NSEI", "NIFTYBEES.NS", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    assert all(item["currency"] == "INR" for item in markets)
