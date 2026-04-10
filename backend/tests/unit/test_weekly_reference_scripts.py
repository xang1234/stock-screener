from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import app.scripts.build_weekly_reference_bundle as build_script
import app.scripts.import_weekly_reference_bundle as import_script
import app.scripts.load_ibd_industry_groups as load_ibd_script


@contextmanager
def _fake_session():
    yield "db-session"


def test_build_weekly_reference_bundle_runs_publish_hydrate_and_export(monkeypatch, tmp_path, capsys):
    published_at = datetime(2026, 4, 4, 12, 10, 0)
    monkeypatch.setattr(build_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(build_script, "SessionLocal", _fake_session)
    stock_universe_service = SimpleNamespace(
        populate_universe=lambda db: {"added": 10},
    )
    provider_snapshot_service = SimpleNamespace(
        create_snapshot_run=lambda db, run_mode, publish, **kwargs: (
            kwargs["progress_callback"](
                {
                    "stage": "snapshot_fetch_complete",
                    "completed_fetches": 1,
                    "total_fetches": 12,
                    "percent_complete": 8.3,
                    "exchange": "NYSE",
                    "category": "overview",
                    "rows": 20,
                }
            )
            or {"published": True, "source_revision": "fundamentals_v1:20260404121000"}
        ),
        hydrate_published_snapshot=lambda db, allow_yahoo_hydration=True, **kwargs: (
            kwargs["progress_callback"](
                {
                    "stage": "hydrate_start",
                    "total_symbols": 10,
                    "total_chunks": 1,
                    "chunk_size": 200,
                }
            )
            or kwargs["progress_callback"](
                {
                    "stage": "hydrate_chunk_complete",
                    "chunk_index": 1,
                    "total_chunks": 1,
                    "processed_symbols": 10,
                    "total_symbols": 10,
                    "percent_complete": 100.0,
                    "live_price_symbols": 9,
                    "cached_only_symbols": 1,
                    "yahoo_hydrated": 3,
                    "missing_prices": 0,
                    "missing_yahoo": 1,
                    "skipped_yahoo_price_symbols": 1,
                    "skipped_yahoo_field_symbols": 1,
                }
            )
            or {"hydrated": 10}
        ),
        get_published_run=lambda db: type(
            "Run",
            (),
            {
                "published_at": published_at,
                "created_at": published_at,
                "source_revision": "fundamentals_v1:20260404121000",
            },
        )(),
    )
    monkeypatch.setattr(build_script, "get_stock_universe_service", lambda: stock_universe_service)
    monkeypatch.setattr(build_script, "get_provider_snapshot_service", lambda: provider_snapshot_service)

    export_calls: list[tuple[Path, str, Path]] = []

    def fake_export(db, *, output_path, bundle_asset_name, latest_manifest_path):
        export_calls.append((output_path, bundle_asset_name, latest_manifest_path))
        latest_manifest_path.write_text(json.dumps({"bundle_asset_name": bundle_asset_name}), encoding="utf-8")
        output_path.write_bytes(b"bundle")
        return {"bundle_path": str(output_path)}

    provider_snapshot_service.export_weekly_reference_bundle = fake_export

    monkeypatch.setattr(
        "sys.argv",
        ["build_weekly_reference_bundle", "--output-dir", str(tmp_path)],
    )

    assert build_script.main() == 0
    assert export_calls[0][0] == tmp_path / "weekly-reference-20260404-fundamentals_v1-20260404121000.json.gz"
    assert export_calls[0][1] == "weekly-reference-20260404-fundamentals_v1-20260404121000.json.gz"
    assert export_calls[0][2] == tmp_path / build_script.ProviderSnapshotService.WEEKLY_REFERENCE_LATEST_MANIFEST_NAME
    stdout = capsys.readouterr().out
    assert "Starting stock universe refresh from Finviz..." in stdout
    assert "[snapshot] 1/12 (8.3%) NYSE overview rows=20" in stdout
    assert "[hydrate] starting 10 symbols in 1 chunks (chunk_size=200)" in stdout
    assert "[hydrate] chunk 1/1 processed 10/10 (100.0%)" in stdout
    assert "Weekly reference bundle complete:" in stdout


def test_import_weekly_reference_bundle_script_calls_service(monkeypatch, tmp_path, capsys):
    bundle_path = tmp_path / "weekly-reference.json.gz"
    bundle_path.write_bytes(b"bundle")
    monkeypatch.setattr(import_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(import_script, "SessionLocal", _fake_session)
    import_calls: list[Path] = []
    provider_snapshot_service = SimpleNamespace(
        import_weekly_reference_bundle=lambda db, input_path: import_calls.append(input_path) or {"rows": 10},
    )
    monkeypatch.setattr(import_script, "get_provider_snapshot_service", lambda: provider_snapshot_service)
    monkeypatch.setattr(
        "sys.argv",
        ["import_weekly_reference_bundle", "--input", str(bundle_path)],
    )

    assert import_script.main() == 0
    assert import_calls == [bundle_path]
    assert "Weekly reference import complete:" in capsys.readouterr().out


def test_load_ibd_industry_groups_script_uses_csv_path(monkeypatch, tmp_path, capsys):
    csv_path = tmp_path / "IBD_industry_group.csv"
    csv_path.write_text("AAPL,Software\n", encoding="utf-8")
    monkeypatch.setattr(load_ibd_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(load_ibd_script, "SessionLocal", _fake_session)
    load_calls: list[str] = []
    monkeypatch.setattr(
        load_ibd_script.IBDIndustryService,
        "load_from_csv",
        lambda db, csv_path: load_calls.append(csv_path) or 1,
    )
    monkeypatch.setattr(
        "sys.argv",
        ["load_ibd_industry_groups", "--csv", str(csv_path)],
    )

    assert load_ibd_script.main() == 0
    assert load_calls == [str(csv_path)]
    assert "IBD industry group load complete:" in capsys.readouterr().out
