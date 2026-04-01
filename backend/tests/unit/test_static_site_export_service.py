"""Tests for the static-site export service."""

from __future__ import annotations

from datetime import date, datetime
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.services.static_site_export_service as export_module
from app.database import Base
from app.infra.db.models.feature_store import FeatureRun, FeatureRunPointer
from app.models.stock import StockPrice
from app.services.static_site_export_service import (
    STATIC_SITE_SCHEMA_VERSION,
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
    themes_index = {
        "schema_version": STATIC_SITE_SCHEMA_VERSION,
        "generated_at": "2026-03-31T22:00:00Z",
        "available": True,
        "variants": {
            "technical:flat": {
                "available": True,
                "path": "themes/technical-flat.json",
                "preview_rankings": [{"theme": "AI Infrastructure", "rank": 1}],
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
        "top_themes": [{"theme": "AI Infrastructure", "rank": 1}],
    }

    monkeypatch.setattr(service, "_export_scan_bundle", lambda **_kwargs: scan_manifest)
    monkeypatch.setattr(service, "_build_breadth_payload", lambda **_kwargs: breadth_payload)
    monkeypatch.setattr(service, "_build_groups_payload", lambda **_kwargs: groups_payload)
    monkeypatch.setattr(service, "_build_home_payload", lambda **_kwargs: home_payload)

    def _build_themes_payloads(**kwargs):
        kwargs["warnings"].append("Themes export failed for fundamental:flat: upstream unavailable")
        return themes_index

    monkeypatch.setattr(service, "_build_themes_payloads", _build_themes_payloads)

    output_dir = tmp_path / "static-data"
    result = service.export(output_dir)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    breadth = json.loads((output_dir / "breadth.json").read_text(encoding="utf-8"))
    groups = json.loads((output_dir / "groups.json").read_text(encoding="utf-8"))
    home = json.loads((output_dir / "home.json").read_text(encoding="utf-8"))
    themes = json.loads((output_dir / "themes" / "index.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == STATIC_SITE_SCHEMA_VERSION
    assert manifest["pages"]["scan"]["path"] == "scan/manifest.json"
    assert manifest["warnings"] == ["Themes export failed for fundamental:flat: upstream unavailable"]
    assert breadth["payload"]["current"]["date"] == "2026-03-31"
    assert groups["payload"]["rankings"]["rankings"][0]["industry_group"] == "Semiconductors"
    assert home["top_themes"][0]["theme"] == "AI Infrastructure"
    assert themes["variants"]["technical:flat"]["path"] == "themes/technical-flat.json"
    assert result.manifest == manifest
    assert result.warnings == ("Themes export failed for fundamental:flat: upstream unavailable",)


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
        lambda row: {"symbol": f"SYM{row.index}", "composite_score": 100 - row.index},
    )

    with session_factory() as db:
        run = db.get(FeatureRun, 11)
        manifest = service._export_scan_bundle(  # noqa: SLF001 - intentional unit test coverage
            db=db,
            output_dir=tmp_path,
            generated_at="2026-03-31T22:00:00Z",
            run=run,
        )

    first_chunk = json.loads((tmp_path / "scan" / "chunks" / "chunk-0001.json").read_text(encoding="utf-8"))
    second_chunk = json.loads((tmp_path / "scan" / "chunks" / "chunk-0002.json").read_text(encoding="utf-8"))

    assert manifest["chunk_size"] == 3
    assert manifest["rows_total"] == 5
    assert [chunk["count"] for chunk in manifest["chunks"]] == [3, 2]
    assert [row["symbol"] for row in first_chunk["rows"]] == ["SYM0", "SYM1", "SYM2"]
    assert [row["symbol"] for row in second_chunk["rows"]] == ["SYM3", "SYM4"]


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
