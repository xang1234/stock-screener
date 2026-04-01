from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

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
    monkeypatch.setattr(
        build_script.stock_universe_service,
        "populate_universe",
        lambda db: {"added": 10},
    )
    monkeypatch.setattr(
        build_script.provider_snapshot_service,
        "create_snapshot_run",
        lambda db, run_mode, publish: {"published": True, "source_revision": "fundamentals_v1:20260404121000"},
    )
    monkeypatch.setattr(
        build_script.provider_snapshot_service,
        "hydrate_published_snapshot",
        lambda db, allow_yahoo_hydration=True: {"hydrated": 10},
    )
    monkeypatch.setattr(
        build_script.provider_snapshot_service,
        "get_published_run",
        lambda db: type(
            "Run",
            (),
            {
                "published_at": published_at,
                "created_at": published_at,
                "source_revision": "fundamentals_v1:20260404121000",
            },
        )(),
    )

    export_calls: list[tuple[Path, str, Path]] = []

    def fake_export(db, *, output_path, bundle_asset_name, latest_manifest_path):
        export_calls.append((output_path, bundle_asset_name, latest_manifest_path))
        latest_manifest_path.write_text(json.dumps({"bundle_asset_name": bundle_asset_name}), encoding="utf-8")
        output_path.write_bytes(b"bundle")
        return {"bundle_path": str(output_path)}

    monkeypatch.setattr(
        build_script.provider_snapshot_service,
        "export_weekly_reference_bundle",
        fake_export,
    )

    monkeypatch.setattr(
        "sys.argv",
        ["build_weekly_reference_bundle", "--output-dir", str(tmp_path)],
    )

    assert build_script.main() == 0
    assert export_calls[0][0] == tmp_path / "weekly-reference-20260404-fundamentals_v1-20260404121000.json.gz"
    assert export_calls[0][1] == "weekly-reference-20260404-fundamentals_v1-20260404121000.json.gz"
    assert export_calls[0][2] == tmp_path / build_script.provider_snapshot_service.WEEKLY_REFERENCE_LATEST_MANIFEST_NAME
    assert "Weekly reference bundle complete:" in capsys.readouterr().out


def test_import_weekly_reference_bundle_script_calls_service(monkeypatch, tmp_path, capsys):
    bundle_path = tmp_path / "weekly-reference.json.gz"
    bundle_path.write_bytes(b"bundle")
    monkeypatch.setattr(import_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(import_script, "SessionLocal", _fake_session)
    import_calls: list[Path] = []
    monkeypatch.setattr(
        import_script.provider_snapshot_service,
        "import_weekly_reference_bundle",
        lambda db, input_path: import_calls.append(input_path) or {"rows": 10},
    )
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
