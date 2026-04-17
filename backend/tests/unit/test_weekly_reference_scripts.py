from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import app.scripts.build_weekly_reference_bundle as build_script
import app.scripts.import_weekly_reference_bundle as import_script
import app.scripts.load_ibd_industry_groups as load_ibd_script


@contextmanager
def _fake_session(db="db-session"):
    yield db


def test_build_weekly_reference_bundle_requires_market(monkeypatch, tmp_path):
    monkeypatch.setattr(build_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(
        "sys.argv",
        ["build_weekly_reference_bundle", "--output-dir", str(tmp_path)],
    )

    with pytest.raises(SystemExit):
        build_script.main()


def test_build_weekly_reference_bundle_runs_us_publish_and_export(monkeypatch, tmp_path, capsys):
    published_at = datetime(2026, 4, 4, 12, 10, 0)
    monkeypatch.setattr(build_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(build_script, "SessionLocal", _fake_session)
    stock_universe_service = SimpleNamespace(
        populate_universe=lambda db: {"added": 10},
    )
    provider_snapshot_service = SimpleNamespace(
        create_snapshot_run=lambda db, run_mode, publish, snapshot_key, market, **kwargs: (
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
            or {
                "published": True,
                "source_revision": "fundamentals_v1_us:20260404121000",
                "snapshot_key": snapshot_key,
                "market": market,
                "coverage": {
                    "snapshot_symbols": 20,
                    "active_symbols": 20,
                },
                "coverage_thresholds": {
                    "market": market,
                    "active_coverage": 1.0,
                    "min_active_coverage": 0.98,
                    "missing_ratio": 0.0,
                    "max_missing_ratio": 0.005,
                },
            }
        ),
        get_published_run=lambda db, snapshot_key: type(
            "Run",
            (),
            {
                "published_at": published_at,
                "created_at": published_at,
                "source_revision": "fundamentals_v1_us:20260404121000",
            },
        )(),
    )
    monkeypatch.setattr(build_script, "get_stock_universe_service", lambda: stock_universe_service)
    monkeypatch.setattr(build_script, "get_provider_snapshot_service", lambda: provider_snapshot_service)

    export_calls: list[dict[str, object]] = []

    def fake_export(db, **kwargs):
        export_calls.append(kwargs)
        kwargs["latest_manifest_path"].write_text(
            json.dumps({"bundle_asset_name": kwargs["bundle_asset_name"]}),
            encoding="utf-8",
        )
        kwargs["output_path"].write_bytes(b"bundle")
        return {"bundle_path": str(kwargs["output_path"])}

    provider_snapshot_service.export_weekly_reference_bundle = fake_export

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_weekly_reference_bundle",
            "--market",
            "US",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert build_script.main() == 0
    assert export_calls[0]["output_path"] == (
        tmp_path / "weekly-reference-us-20260404-fundamentals_v1_us-20260404121000.json.gz"
    )
    assert export_calls[0]["bundle_asset_name"] == (
        "weekly-reference-us-20260404-fundamentals_v1_us-20260404121000.json.gz"
    )
    assert export_calls[0]["latest_manifest_path"] == tmp_path / "weekly-reference-latest-us.json"
    assert export_calls[0]["snapshot_key"] == build_script.ProviderSnapshotService.snapshot_key_for_market("US")
    assert export_calls[0]["market"] == "US"
    stdout = capsys.readouterr().out
    assert "Starting stock universe refresh from Finviz..." in stdout
    assert "[snapshot] 1/12 (8.3%) NYSE overview rows=20" in stdout
    assert "[publish] market=US coverage=100.00% (min=98.00%) missing_ratio=0.00% (max=0.50%)" in stdout
    assert "Weekly reference bundle complete for US:" in stdout


def test_build_weekly_reference_bundle_writes_summary_when_us_publish_is_blocked(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(build_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(build_script, "SessionLocal", _fake_session)
    summary_path = tmp_path / "github-step-summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))
    stock_universe_service = SimpleNamespace(
        populate_universe=lambda db: {"added": 10},
    )
    provider_snapshot_service = SimpleNamespace(
        create_snapshot_run=lambda db, run_mode, publish, snapshot_key, market, **kwargs: {
            "published": False,
            "warnings": ["Active snapshot coverage 80.00% below minimum 98.00%"],
            "source_revision": "fundamentals_v1_us:20260404121000",
            "snapshot_key": snapshot_key,
            "market": market,
            "coverage": {
                "snapshot_symbols": 8,
                "active_symbols": 10,
            },
            "coverage_thresholds": {
                "market": market,
                "active_coverage": 0.8,
                "min_active_coverage": 0.98,
                "missing_ratio": 0.2,
                "max_missing_ratio": 0.005,
            },
        },
    )
    monkeypatch.setattr(build_script, "get_stock_universe_service", lambda: stock_universe_service)
    monkeypatch.setattr(build_script, "get_provider_snapshot_service", lambda: provider_snapshot_service)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_weekly_reference_bundle",
            "--market",
            "US",
            "--output-dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(RuntimeError, match="Weekly fundamentals snapshot did not publish"):
        build_script.main()

    summary_text = summary_path.read_text(encoding="utf-8")
    assert "## Weekly Reference Bundle: US" in summary_text
    assert "| Active coverage | 80.00% |" in summary_text
    assert "| Minimum coverage | 98.00% |" in summary_text
    assert "| Bundle rows exported | 0 |" in summary_text


def test_build_weekly_reference_bundle_runs_hk_official_path(monkeypatch, tmp_path, capsys):
    published_at = datetime(2026, 4, 4, 12, 10, 0)
    active_rows = [
        SimpleNamespace(
            symbol="0700.HK",
            market="HK",
            exchange="XHKG",
            name="Tencent",
            sector="Technology",
            industry="Internet Content & Information",
            market_cap=456.0,
        )
    ]
    fake_query = MagicMock()
    fake_query.filter.return_value.order_by.return_value.all.return_value = active_rows
    fake_db = MagicMock()
    fake_db.query.return_value = fake_query

    monkeypatch.setattr(build_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(build_script, "SessionLocal", lambda: _fake_session(fake_db))
    summary_path = tmp_path / "github-step-summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))

    fetch_calls: list[str] = []
    official_service = SimpleNamespace(
        fetch_market_snapshot=lambda market: fetch_calls.append(market)
        or SimpleNamespace(
            market=market,
            source_name="hkex_official",
            snapshot_id="hkex-listofsecurities-2026-04-04",
            snapshot_as_of="2026-04-04",
            source_metadata={"source_urls": ["https://example.com"]},
            rows=(
                {
                    "symbol": "0700.HK",
                    "name": "Tencent",
                    "exchange": "XHKG",
                    "sector": "",
                    "industry": "",
                    "market_cap": None,
                },
            ),
        )
    )
    monkeypatch.setattr(build_script, "OfficialMarketUniverseSourceService", lambda: official_service)

    stock_universe_service = SimpleNamespace(
        ingest_hk_snapshot_rows=lambda db, **kwargs: {"added": 1, "updated": 0, "deactivated": 0},
    )
    monkeypatch.setattr(build_script, "get_stock_universe_service", lambda: stock_universe_service)

    hybrid_calls: list[dict[str, object]] = []
    hybrid_service = SimpleNamespace(
        fetch_fundamentals_batch=lambda symbols, **kwargs: hybrid_calls.append(
            {"symbols": symbols, **kwargs}
        )
        or {"0700.HK": {"market_cap": 456.0, "sector": "Technology"}},
        store_all_caches=lambda *args, **kwargs: {
            "fundamentals_stored": 1,
            "persisted_symbols": 1,
            "failed_persistence_symbols": 0,
            "failed": 0,
            "provider_error_counts": {"yahoo_quote_not_found": 2},
        },
    )
    monkeypatch.setattr(build_script, "get_hybrid_fundamentals_service", lambda: hybrid_service)
    monkeypatch.setattr(
        build_script,
        "get_fundamentals_cache",
        lambda: SimpleNamespace(get_many=lambda symbols: {"0700.HK": {"market_cap": 456.0, "sector": "Technology"}}),
    )

    published_rows: list[dict[str, object]] = []
    export_calls: list[dict[str, object]] = []
    provider_snapshot_service = SimpleNamespace(
        build_market_snapshot_row=lambda **kwargs: {
            "symbol": kwargs["symbol"],
            "exchange": kwargs["exchange"],
            "row_hash": "row-hash",
            "normalized_payload": kwargs["normalized_payload"],
            "raw_payload": kwargs["raw_payload"],
        },
        publish_market_snapshot_run=lambda db, **kwargs: published_rows.append(kwargs)
        or {
            "published": True,
            "source_revision": "fundamentals_v1_hk:20260404121000",
            "snapshot_key": kwargs["snapshot_key"],
            "coverage": {
                "snapshot_symbols": 1,
                "active_symbols": 1,
            },
            "coverage_thresholds": {
                "market": "HK",
                "active_coverage": 1.0,
                "min_active_coverage": 0.70,
                "missing_ratio": 0.0,
                "max_missing_ratio": 0.30,
            },
        },
        get_published_run=lambda db, snapshot_key: type(
            "Run",
            (),
            {
                "published_at": published_at,
                "created_at": published_at,
                "source_revision": "fundamentals_v1_hk:20260404121000",
            },
        )(),
        export_weekly_reference_bundle=lambda db, **kwargs: export_calls.append(kwargs)
        or {"bundle_path": str(kwargs["output_path"])},
    )
    monkeypatch.setattr(build_script, "get_provider_snapshot_service", lambda: provider_snapshot_service)

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_weekly_reference_bundle",
            "--market",
            "HK",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert build_script.main() == 0
    assert fetch_calls == ["HK"]
    assert hybrid_calls[0]["include_finviz"] is False
    assert hybrid_calls[0]["market_by_symbol"] == {"0700.HK": "HK"}
    assert published_rows[0]["snapshot_key"] == build_script.ProviderSnapshotService.snapshot_key_for_market("HK")
    assert published_rows[0]["market"] == "HK"
    assert export_calls[0]["output_path"] == (
        tmp_path / "weekly-reference-hk-20260404-fundamentals_v1_hk-20260404121000.json.gz"
    )
    assert export_calls[0]["latest_manifest_path"] == tmp_path / "weekly-reference-latest-hk.json"
    assert export_calls[0]["market"] == "HK"
    stdout = capsys.readouterr().out
    assert "Starting official universe refresh for HK..." in stdout
    assert "Starting hybrid fundamentals refresh for HK..." in stdout
    assert "Fundamentals refresh complete:" in stdout
    assert "[publish] market=HK coverage=100.00% (min=70.00%) missing_ratio=0.00% (max=30.00%)" in stdout
    assert "Weekly reference bundle complete for HK:" in stdout
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "## Weekly Reference Bundle: HK" in summary_text
    assert "| Coverage gate market | HK |" in summary_text
    assert "| Minimum coverage | 70.00% |" in summary_text
    assert "| Failed persistence symbols | 0 |" in summary_text
    assert "| `yahoo_quote_not_found` | 2 |" in summary_text


def test_import_weekly_reference_bundle_script_calls_service(monkeypatch, tmp_path, capsys):
    bundle_path = tmp_path / "weekly-reference.json.gz"
    bundle_path.write_bytes(b"bundle")
    monkeypatch.setattr(import_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(import_script, "SessionLocal", _fake_session)
    import_calls: list[tuple[Path, bool, str]] = []
    provider_snapshot_service = SimpleNamespace(
        import_weekly_reference_bundle=lambda db, input_path, hydrate_cache=True, hydrate_mode="static": (
            import_calls.append((input_path, hydrate_cache, hydrate_mode)) or {"rows": 10}
        ),
    )
    monkeypatch.setattr(import_script, "get_provider_snapshot_service", lambda: provider_snapshot_service)
    monkeypatch.setattr(
        "sys.argv",
        ["import_weekly_reference_bundle", "--input", str(bundle_path)],
    )

    assert import_script.main() == 0
    assert import_calls == [(bundle_path, True, "static")]
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
