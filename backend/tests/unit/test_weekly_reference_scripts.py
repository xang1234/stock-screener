from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

import app.scripts.build_weekly_reference_bundle as build_script
import app.scripts.import_weekly_reference_bundle as import_script
import app.scripts.load_ibd_industry_groups as load_ibd_script
from app.models.provider_snapshot import ProviderSnapshotRow
from app.models.stock_universe import StockUniverse


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
    universe_calls: list[tuple] = []
    stock_universe_service = SimpleNamespace(
        seed_reconciliation_baseline_from_active_rows=lambda db, **kwargs: universe_calls.append(
            ("baseline", kwargs)
        ) or {"snapshot_id": kwargs["snapshot_id"], "baseline_rows": 20},
        populate_universe=lambda db: universe_calls.append(("populate",)) or {"added": 10},
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
        hydrate_published_snapshot=lambda db, snapshot_key, progress_callback=None: {
            "hydrated": 20,
            "yahoo_hydrated": 18,
            "missing_prices": 0,
            "missing_yahoo": 0,
        },
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
    # The imported seed universe becomes the Finviz reconciliation baseline
    # before the refresh, so symbols Finviz dropped can be deactivated.
    assert [call[0] for call in universe_calls] == ["baseline", "populate"]
    assert universe_calls[0][1]["market"] == "US"
    assert universe_calls[0][1]["source_name"] == "finviz"
    assert universe_calls[0][1]["snapshot_id"].startswith("weekly-reference-seed:")
    assert "Seeded Finviz reconciliation baseline: 20 active rows" in stdout
    assert "[snapshot] 1/12 (8.3%) NYSE overview rows=20" in stdout
    assert "[publish] market=US coverage=100.00% (min=98.00%) missing_ratio=0.00% (max=0.50%)" in stdout
    assert "Starting Yahoo hydration for US published snapshot..." in stdout
    assert "Hydration complete:" in stdout
    # Hydration must run before export so market_cap from stock_fundamentals
    # is available at merge-time in export_weekly_reference_bundle.
    hydrate_idx = stdout.index("Starting Yahoo hydration for US published snapshot...")
    export_idx = stdout.index("Weekly reference bundle complete for US:")
    assert hydrate_idx < export_idx
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
        seed_reconciliation_baseline_from_active_rows=lambda db, **kwargs: None,
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


def test_build_weekly_reference_bundle_us_publishes_seeded_cache_fallback(
    monkeypatch,
    tmp_path,
):
    # The cache backfill needs the prior seed within the max age (#520).
    published_at = datetime.utcnow() - timedelta(days=1)
    active_rows = [
        SimpleNamespace(
            symbol="AAPL",
            market="US",
            exchange="NASDAQ",
            name="Apple Inc.",
            sector="Technology",
            industry="Consumer Electronics",
            market_cap=3_000_000_000_000,
            currency="USD",
            timezone="America/New_York",
            local_code="AAPL",
        ),
        SimpleNamespace(
            symbol="MSFT",
            market="US",
            exchange="NASDAQ",
            name="Microsoft Corp.",
            sector="Technology",
            industry="Software",
            market_cap=3_100_000_000_000,
            currency="USD",
            timezone="America/New_York",
            local_code="MSFT",
        ),
    ]
    blocked_rows = [
        SimpleNamespace(
            symbol="AAPL",
            exchange="NASDAQ",
            row_hash="hash-aapl",
            normalized_payload_json=json.dumps(
                {"symbol": "AAPL", "exchange": "NASDAQ", "market": "US"}
            ),
            raw_payload_json=json.dumps({"overview": {"Ticker": "AAPL"}}),
        )
    ]

    fake_db = MagicMock()

    def fake_query(model):
        query = MagicMock()
        if model is StockUniverse:
            query.filter.return_value.order_by.return_value.all.return_value = active_rows
        elif model is ProviderSnapshotRow:
            query.filter.return_value.all.return_value = blocked_rows
        return query

    fake_db.query.side_effect = fake_query

    monkeypatch.setattr(build_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(build_script, "SessionLocal", lambda: _fake_session(fake_db))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(
        build_script,
        "get_fundamentals_cache",
        lambda: SimpleNamespace(
            get_many=lambda symbols: {
                "MSFT": {
                    "symbol": "MSFT",
                    "company_name": "Microsoft Corp.",
                    "market_cap": 3_100_000_000_000,
                }
            }
        ),
    )
    monkeypatch.setattr(
        build_script,
        "get_stock_universe_service",
        lambda: SimpleNamespace(
            seed_reconciliation_baseline_from_active_rows=lambda db, **kwargs: None,
            populate_universe=lambda db: {"added": 0, "updated": 2},
        ),
    )

    publish_calls: list[dict[str, object]] = []
    export_calls: list[dict[str, object]] = []
    provider_snapshot_service = SimpleNamespace(
        create_snapshot_run=lambda db, run_mode, publish, snapshot_key, market, **kwargs: {
            "run_id": 42,
            "published": False,
            "warnings": ["Missing active symbol ratio 50.00% above maximum 0.50%"],
            "source_revision": "fundamentals_v1_us:20260404121000",
            "snapshot_key": snapshot_key,
            "market": market,
            "coverage": {
                "snapshot_symbols": 1,
                "active_symbols": 2,
                "covered_active_symbols": 1,
                "missing_active_symbols": 1,
            },
            "coverage_thresholds": {
                "market": "US",
                "active_coverage": 0.5,
                "min_active_coverage": 0.98,
                "missing_ratio": 0.5,
                "max_missing_ratio": 0.005,
            },
        },
        build_market_snapshot_row=lambda **kwargs: {
            "symbol": kwargs["symbol"],
            "exchange": kwargs["exchange"],
            "row_hash": f"hash-{kwargs['symbol'].lower()}",
            "normalized_payload": kwargs["normalized_payload"],
            "raw_payload": kwargs["raw_payload"],
        },
        publish_market_snapshot_run=lambda db, **kwargs: publish_calls.append(kwargs)
        or {
            "published": True,
            "source_revision": "fundamentals_v1_us:20260404121100-seeded-fallback",
            "snapshot_key": kwargs["snapshot_key"],
            "market": kwargs["market"],
            "coverage": dict(kwargs["coverage_stats"]),
            "warnings": list(kwargs["warnings"]),
            "coverage_thresholds": {
                "market": "US",
                "active_coverage": 1.0,
                "min_active_coverage": 0.98,
                "missing_ratio": 0.0,
                "max_missing_ratio": 0.005,
            },
        },
        get_published_run=lambda db, snapshot_key: type(
            "Run",
            (),
            {
                "published_at": published_at,
                "created_at": published_at,
                "source_revision": "fundamentals_v1_us:20260404121100-seeded-fallback",
                "coverage_stats_json": None,
            },
        )(),
        hydrate_published_snapshot=lambda db, snapshot_key, progress_callback=None: {
            "hydrated": 2,
        },
        export_weekly_reference_bundle=lambda db, **kwargs: export_calls.append(kwargs)
        or {"bundle_path": str(kwargs["output_path"])},
    )
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

    assert build_script.main() == 0

    publish_kwargs = publish_calls[0]
    assert [row["symbol"] for row in publish_kwargs["rows"]] == ["AAPL", "MSFT"]
    assert publish_kwargs["coverage_stats"]["backfilled_active_symbols"] == 1
    assert publish_kwargs["coverage_stats"]["missing_active_symbols"] == 0
    assert any("Backfilled 1 US active symbols" in w for w in publish_kwargs["warnings"])
    assert export_calls, "Bundle export should run after seeded fallback publish"


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
        seed_reconciliation_baseline_from_active_rows=lambda db, **kwargs: None,
        ingest_hk_snapshot_rows=lambda db, **kwargs: {"added": 1, "updated": 0, "deactivated": 0},
    )
    monkeypatch.setattr(build_script, "get_stock_universe_service", lambda: stock_universe_service)

    hybrid_calls: list[dict[str, object]] = []

    def fake_fetch_fundamentals_batch(symbols, **kwargs):
        hybrid_calls.append({"symbols": symbols, **kwargs})
        kwargs["progress_callback"](1, len(symbols))
        return {"0700.HK": {"market_cap": 456.0, "sector": "Technology"}}

    hybrid_service = SimpleNamespace(
        yfinance_delay_per_ticker=1.5,
        fetch_fundamentals_batch=fake_fetch_fundamentals_batch,
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
    assert callable(hybrid_calls[0]["progress_callback"])
    assert hybrid_service.yfinance_delay_per_ticker == build_script._WEEKLY_NON_US_YFINANCE_DELAY_PER_TICKER
    assert published_rows[0]["snapshot_key"] == build_script.ProviderSnapshotService.snapshot_key_for_market("HK")
    assert published_rows[0]["market"] == "HK"
    assert export_calls[0]["output_path"] == (
        tmp_path / "weekly-reference-hk-20260404-fundamentals_v1_hk-20260404121000.json.gz"
    )
    assert export_calls[0]["latest_manifest_path"] == tmp_path / "weekly-reference-latest-hk.json"
    assert export_calls[0]["market"] == "HK"
    stdout = capsys.readouterr().out
    assert "Starting official universe refresh for HK..." in stdout
    assert (
        "Using weekly non-US yfinance per-ticker delay "
        f"{build_script._WEEKLY_NON_US_YFINANCE_DELAY_PER_TICKER:.2f}s for HK"
    ) in stdout
    assert "Starting hybrid fundamentals refresh for HK..." in stdout
    assert "[fundamentals] HK 1/1 (100.0%)" in stdout
    assert "Fundamentals refresh complete:" in stdout
    assert "[publish] market=HK coverage=100.00% (min=70.00%) missing_ratio=0.00% (max=30.00%)" in stdout
    assert "Weekly reference bundle complete for HK:" in stdout
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "## Weekly Reference Bundle: HK" in summary_text
    assert "| Coverage gate market | HK |" in summary_text
    assert "| Minimum coverage | 70.00% |" in summary_text
    assert "| Failed persistence symbols | 0 |" in summary_text
    assert "| `yahoo_quote_not_found` | 2 |" in summary_text


def test_build_weekly_reference_bundle_runs_de_official_path(monkeypatch, tmp_path, capsys):
    """Mirror of the HK official-path test for the new DE fetcher.

    Verifies the build script (1) fetches the DE snapshot via the official
    source service, (2) routes the snapshot to ``ingest_de_snapshot_rows``
    rather than raising on ``Unsupported official weekly reference market``,
    and (3) publishes/exports the resulting bundle.
    """
    published_at = datetime(2026, 5, 9, 12, 10, 0)
    active_rows = [
        SimpleNamespace(
            symbol="SAP.DE",
            market="DE",
            exchange="XETR",
            name="SAP SE",
            sector="Technology",
            industry="Application Software",
            market_cap=200.0,
        )
    ]
    fake_query = MagicMock()
    fake_query.filter.return_value.order_by.return_value.all.return_value = active_rows
    fake_db = MagicMock()
    fake_db.query.return_value = fake_query

    monkeypatch.setattr(build_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(build_script, "SessionLocal", lambda: _fake_session(fake_db))

    fetch_calls: list[str] = []
    official_service = SimpleNamespace(
        fetch_market_snapshot=lambda market: fetch_calls.append(market)
        or SimpleNamespace(
            market=market,
            source_name="dbg_official",
            snapshot_id="dbg-equity-2026-05-09",
            snapshot_as_of="2026-05-09",
            source_metadata={"source_urls": ["https://api.boerse-frankfurt.de"], "fetch_mode": "live_http"},
            rows=(
                {
                    "symbol": "SAP.DE",
                    "name": "SAP SE",
                    "exchange": "XETR",
                    "sector": "",
                    "industry": "",
                    "market_cap": None,
                    "isin": "DE0007164600",
                    "wkn": "SAP",
                },
            ),
        )
    )
    monkeypatch.setattr(build_script, "OfficialMarketUniverseSourceService", lambda: official_service)

    ingest_calls: list[dict[str, object]] = []
    stock_universe_service = SimpleNamespace(
        seed_reconciliation_baseline_from_active_rows=lambda db, **kwargs: None,
        ingest_de_snapshot_rows=lambda db, **kwargs: ingest_calls.append(kwargs)
        or {"added": 1, "updated": 0, "deactivated": 0},
    )
    monkeypatch.setattr(build_script, "get_stock_universe_service", lambda: stock_universe_service)

    hybrid_service = SimpleNamespace(
        yfinance_delay_per_ticker=1.5,
        fetch_fundamentals_batch=lambda symbols, **kwargs: (
            kwargs["progress_callback"](1, len(symbols))
            or {"SAP.DE": {"market_cap": 200.0, "sector": "Technology"}}
        ),
        store_all_caches=lambda *args, **kwargs: {
            "fundamentals_stored": 1,
            "persisted_symbols": 1,
            "failed_persistence_symbols": 0,
            "failed": 0,
            "provider_error_counts": {},
        },
    )
    monkeypatch.setattr(build_script, "get_hybrid_fundamentals_service", lambda: hybrid_service)
    monkeypatch.setattr(
        build_script,
        "get_fundamentals_cache",
        lambda: SimpleNamespace(
            get_many=lambda symbols: {"SAP.DE": {"market_cap": 200.0, "sector": "Technology"}}
        ),
    )

    export_calls: list[dict[str, object]] = []
    provider_snapshot_service = SimpleNamespace(
        build_market_snapshot_row=lambda **kwargs: {
            "symbol": kwargs["symbol"],
            "exchange": kwargs["exchange"],
            "row_hash": "row-hash",
            "normalized_payload": kwargs["normalized_payload"],
            "raw_payload": kwargs["raw_payload"],
        },
        publish_market_snapshot_run=lambda db, **kwargs: {
            "published": True,
            "source_revision": "fundamentals_v1_de:20260509121000",
            "snapshot_key": kwargs["snapshot_key"],
            "coverage": {"snapshot_symbols": 1, "active_symbols": 1},
            "coverage_thresholds": {
                "market": "DE",
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
                "source_revision": "fundamentals_v1_de:20260509121000",
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
            "DE",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert build_script.main() == 0
    assert fetch_calls == ["DE"]
    assert len(ingest_calls) == 1
    assert ingest_calls[0]["source_name"] == "dbg_official"
    assert ingest_calls[0]["snapshot_id"] == "dbg-equity-2026-05-09"
    assert export_calls[0]["output_path"] == (
        tmp_path / "weekly-reference-de-20260509-fundamentals_v1_de-20260509121000.json.gz"
    )
    assert export_calls[0]["latest_manifest_path"] == tmp_path / "weekly-reference-latest-de.json"
    assert export_calls[0]["market"] == "DE"
    stdout = capsys.readouterr().out
    assert "Starting official universe refresh for DE..." in stdout
    assert "Weekly reference bundle complete for DE:" in stdout


def test_build_weekly_reference_bundle_runs_sg_official_path(monkeypatch, tmp_path, capsys):
    """Mirror of the DE official-path test for the SG fetcher.

    Verifies the build script (1) fetches the SG snapshot via the official
    source service, (2) routes the snapshot to ``ingest_sg_snapshot_rows``
    rather than raising on ``Unsupported official weekly reference market``,
    and (3) publishes/exports the resulting bundle.
    """
    published_at = datetime(2026, 5, 17, 12, 10, 0)
    active_rows = [
        SimpleNamespace(
            symbol="D05.SI",
            market="SG",
            exchange="XSES",
            name="DBS Group Holdings",
            sector="Banks",
            industry="Banks",
            market_cap=95.0,
        )
    ]
    fake_query = MagicMock()
    fake_query.filter.return_value.order_by.return_value.all.return_value = active_rows
    fake_db = MagicMock()
    fake_db.query.return_value = fake_query

    monkeypatch.setattr(build_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(build_script, "SessionLocal", lambda: _fake_session(fake_db))

    fetch_calls: list[str] = []
    official_service = SimpleNamespace(
        fetch_market_snapshot=lambda market: fetch_calls.append(market)
        or SimpleNamespace(
            market=market,
            source_name="sg_manual_csv",
            snapshot_id="sg-csv-fallback-2026-05-17",
            snapshot_as_of="2026-05-17",
            source_metadata={"source_urls": [], "fetch_mode": "csv_fallback"},
            rows=(
                {
                    "symbol": "D05.SI",
                    "name": "DBS Group Holdings",
                    "exchange": "XSES",
                    "sector": "",
                    "industry": "",
                    "market_cap": None,
                    "isin": "SG1L01001701",
                },
            ),
        )
    )
    monkeypatch.setattr(build_script, "OfficialMarketUniverseSourceService", lambda: official_service)

    ingest_calls: list[dict[str, object]] = []
    stock_universe_service = SimpleNamespace(
        seed_reconciliation_baseline_from_active_rows=lambda db, **kwargs: None,
        ingest_sg_snapshot_rows=lambda db, **kwargs: ingest_calls.append(kwargs)
        or {"added": 1, "updated": 0, "deactivated": 0},
    )
    monkeypatch.setattr(build_script, "get_stock_universe_service", lambda: stock_universe_service)

    hybrid_service = SimpleNamespace(
        yfinance_delay_per_ticker=1.5,
        fetch_fundamentals_batch=lambda symbols, **kwargs: (
            kwargs["progress_callback"](1, len(symbols))
            or {"D05.SI": {"market_cap": 95.0, "sector": "Banks"}}
        ),
        store_all_caches=lambda *args, **kwargs: {
            "fundamentals_stored": 1,
            "persisted_symbols": 1,
            "failed_persistence_symbols": 0,
            "failed": 0,
            "provider_error_counts": {},
        },
    )
    monkeypatch.setattr(build_script, "get_hybrid_fundamentals_service", lambda: hybrid_service)
    monkeypatch.setattr(
        build_script,
        "get_fundamentals_cache",
        lambda: SimpleNamespace(
            get_many=lambda symbols: {"D05.SI": {"market_cap": 95.0, "sector": "Banks"}}
        ),
    )

    export_calls: list[dict[str, object]] = []
    provider_snapshot_service = SimpleNamespace(
        build_market_snapshot_row=lambda **kwargs: {
            "symbol": kwargs["symbol"],
            "exchange": kwargs["exchange"],
            "row_hash": "row-hash",
            "normalized_payload": kwargs["normalized_payload"],
            "raw_payload": kwargs["raw_payload"],
        },
        publish_market_snapshot_run=lambda db, **kwargs: {
            "published": True,
            "source_revision": "fundamentals_v1_sg:20260517121000",
            "snapshot_key": kwargs["snapshot_key"],
            "coverage": {"snapshot_symbols": 1, "active_symbols": 1},
            "coverage_thresholds": {
                "market": "SG",
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
                "source_revision": "fundamentals_v1_sg:20260517121000",
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
            "SG",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert build_script.main() == 0
    assert fetch_calls == ["SG"]
    assert len(ingest_calls) == 1
    assert ingest_calls[0]["source_name"] == "sg_manual_csv"
    assert ingest_calls[0]["snapshot_id"] == "sg-csv-fallback-2026-05-17"
    assert export_calls[0]["output_path"] == (
        tmp_path / "weekly-reference-sg-20260517-fundamentals_v1_sg-20260517121000.json.gz"
    )
    assert export_calls[0]["latest_manifest_path"] == tmp_path / "weekly-reference-latest-sg.json"
    assert export_calls[0]["market"] == "SG"
    stdout = capsys.readouterr().out
    assert "Starting official universe refresh for SG..." in stdout
    assert "Weekly reference bundle complete for SG:" in stdout


def test_build_weekly_reference_bundle_runs_au_official_path(monkeypatch, tmp_path, capsys):
    """AU weekly reference builds must route ASX snapshots into AU ingestion."""
    published_at = datetime(2026, 5, 30, 12, 10, 0)
    active_rows = [
        SimpleNamespace(
            symbol="BHP.AX",
            market="AU",
            exchange="XASX",
            name="BHP Group Limited",
            sector="Basic Materials",
            industry="Other Industrial Metals & Mining",
            market_cap=220.0,
        ),
        # Seeded from a prior bundle; the ASX source no longer emits it (#481).
        SimpleNamespace(
            symbol="PUT.AX",
            market="AU",
            exchange="XASX",
            name="PUMA SERIES 2023-1 TRUST",
            sector=None,
            industry=None,
            market_cap=None,
        ),
    ]
    fake_query = MagicMock()
    fake_query.filter.return_value.order_by.return_value.all.return_value = active_rows
    fake_db = MagicMock()
    fake_db.query.return_value = fake_query

    monkeypatch.setattr(build_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(build_script, "SessionLocal", lambda: _fake_session(fake_db))

    fetch_calls: list[str] = []
    official_service = SimpleNamespace(
        fetch_market_snapshot=lambda market: fetch_calls.append(market)
        or SimpleNamespace(
            market=market,
            source_name="asx_official_public_csv",
            snapshot_id="asx-listed-companies-2026-05-30",
            snapshot_as_of="2026-05-30",
            source_metadata={
                "source_urls": ["https://www.asx.com.au/asx/research/ASXListedCompanies.csv"],
                "fetch_mode": "live_http",
            },
            rows=(
                {
                    "symbol": "BHP.AX",
                    "name": "BHP Group Limited",
                    "exchange": "XASX",
                    "sector": "",
                    "industry": "",
                    "market_cap": None,
                    "isin": "AU000000BHP4",
                },
            ),
        )
    )
    monkeypatch.setattr(build_script, "OfficialMarketUniverseSourceService", lambda: official_service)

    ingest_calls: list[dict[str, object]] = []
    stock_universe_service = SimpleNamespace(
        seed_reconciliation_baseline_from_active_rows=lambda db, **kwargs: None,
        ingest_au_snapshot_rows=lambda db, **kwargs: ingest_calls.append(kwargs)
        or {"added": 1, "updated": 0, "deactivated": 0},
    )
    monkeypatch.setattr(build_script, "get_stock_universe_service", lambda: stock_universe_service)

    hybrid_service = SimpleNamespace(
        yfinance_delay_per_ticker=1.5,
        fetch_fundamentals_batch=lambda symbols, **kwargs: (
            kwargs["progress_callback"](1, len(symbols))
            or {"BHP.AX": {"market_cap": 220.0, "sector": "Basic Materials"}}
        ),
        store_all_caches=lambda *args, **kwargs: {
            "fundamentals_stored": 1,
            "persisted_symbols": 1,
            "failed_persistence_symbols": 0,
            "failed": 0,
            "provider_error_counts": {},
        },
    )
    monkeypatch.setattr(build_script, "get_hybrid_fundamentals_service", lambda: hybrid_service)
    monkeypatch.setattr(
        build_script,
        "get_fundamentals_cache",
        lambda: SimpleNamespace(
            get_many=lambda symbols: {"BHP.AX": {"market_cap": 220.0, "sector": "Basic Materials"}}
        ),
    )

    export_calls: list[dict[str, object]] = []
    provider_snapshot_service = SimpleNamespace(
        build_market_snapshot_row=lambda **kwargs: {
            "symbol": kwargs["symbol"],
            "exchange": kwargs["exchange"],
            "row_hash": "row-hash",
            "normalized_payload": kwargs["normalized_payload"],
            "raw_payload": kwargs["raw_payload"],
        },
        publish_market_snapshot_run=lambda db, **kwargs: {
            "published": True,
            "source_revision": "fundamentals_v1_au:20260530121000",
            "snapshot_key": kwargs["snapshot_key"],
            "coverage": {"snapshot_symbols": 1, "active_symbols": 1},
            "coverage_thresholds": {
                "market": "AU",
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
                "source_revision": "fundamentals_v1_au:20260530121000",
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
            "AU",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert build_script.main() == 0
    assert fetch_calls == ["AU"]
    assert len(ingest_calls) == 1
    assert ingest_calls[0]["source_name"] == "asx_official_public_csv"
    assert ingest_calls[0]["snapshot_id"] == "asx-listed-companies-2026-05-30"
    assert export_calls[0]["output_path"] == (
        tmp_path / "weekly-reference-au-20260530-fundamentals_v1_au-20260530121000.json.gz"
    )
    assert export_calls[0]["latest_manifest_path"] == tmp_path / "weekly-reference-latest-au.json"
    assert export_calls[0]["market"] == "AU"
    # The export re-queries active rows, so dropped listings must be passed on.
    assert export_calls[0]["excluded_symbols"] == {"PUT.AX"}
    stdout = capsys.readouterr().out
    assert "Starting official universe refresh for AU..." in stdout
    assert "Weekly reference bundle complete for AU:" in stdout


def _make_universe_row(symbol: str, market: str = "CN") -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        market=market,
        exchange="SSE" if symbol.endswith(".SS") else "SZSE",
        name=f"Co-{symbol}",
        sector="Industrials",
        industry="Misc",
        market_cap=1000.0,
    )


def test_build_weekly_reference_bundle_chunked_deadline_force_publishes(
    monkeypatch, tmp_path, capsys
):
    """Asia path stops between chunks once the wall-clock budget is exhausted.

    With --max-runtime-minutes set and the deadline tripping after the first
    chunk, the second chunk is never fetched, the snapshot is published with
    force_publish=True, and warnings record the deadline.
    """

    published_at = datetime(2026, 5, 5, 12, 10, 0)
    active_rows = [
        _make_universe_row("600000.SS"),
        _make_universe_row("600001.SS"),
        _make_universe_row("000001.SZ"),
    ]
    fake_query = MagicMock()
    fake_query.filter.return_value.order_by.return_value.all.return_value = active_rows
    fake_db = MagicMock()
    fake_db.query.return_value = fake_query

    monkeypatch.setattr(build_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(build_script, "SessionLocal", lambda: _fake_session(fake_db))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    official_service = SimpleNamespace(
        fetch_market_snapshot=lambda market: SimpleNamespace(
            market=market,
            source_name="cn_official",
            snapshot_id="cn-2026-05-05",
            snapshot_as_of="2026-05-05",
            source_metadata={},
            rows=tuple({"symbol": row.symbol} for row in active_rows),
        )
    )
    monkeypatch.setattr(
        build_script, "OfficialMarketUniverseSourceService", lambda: official_service
    )
    monkeypatch.setattr(
        build_script,
        "get_stock_universe_service",
        lambda: SimpleNamespace(
            seed_reconciliation_baseline_from_active_rows=lambda db, **kwargs: None,
            ingest_cn_snapshot_rows=lambda db, **kwargs: {"added": 3, "updated": 0, "deactivated": 0}
        ),
    )

    # Drive build_script.time.monotonic so the second chunk sees an exhausted
    # deadline. Each call advances by 60 s, so a 1.5-minute (90 s) budget is
    # consumed after the first chunk's start.
    fake_clock = {"value": 0.0}

    def fake_monotonic() -> float:
        fake_clock["value"] += 60.0
        return fake_clock["value"]

    monkeypatch.setattr(build_script.time, "monotonic", fake_monotonic)

    fetch_calls: list[list[str]] = []
    store_calls: list[list[str]] = []

    def fake_fetch_fundamentals_batch(symbols, **kwargs):
        fetch_calls.append(list(symbols))
        kwargs["progress_callback"](len(symbols), len(symbols))
        return {symbol: {"market_cap": 1000.0, "sector": "Industrials"} for symbol in symbols}

    def fake_store_all_caches(data, _cache, **_kwargs):
        store_calls.append(list(data))
        return {
            "fundamentals_stored": len(data),
            "persisted_symbols": len(data),
            "failed_persistence_symbols": 0,
            "failed": 0,
            "provider_error_counts": {},
        }

    hybrid_service = SimpleNamespace(
        yfinance_delay_per_ticker=1.5,
        fetch_fundamentals_batch=fake_fetch_fundamentals_batch,
        store_all_caches=fake_store_all_caches,
    )
    monkeypatch.setattr(build_script, "get_hybrid_fundamentals_service", lambda: hybrid_service)

    # Cache returns fresh data for both attempted symbols and (simulated)
    # last-week data for the symbol whose chunk was skipped — that's the whole
    # point of force-publish: skipped symbols inherit the prior bundle.
    cached = {
        "600000.SS": {"market_cap": 1000.0, "sector": "Industrials"},
        "600001.SS": {"market_cap": 1000.0, "sector": "Industrials"},
        "000001.SZ": {"market_cap": 999.0, "sector": "Industrials", "data_source": "bundle_import"},
    }
    monkeypatch.setattr(
        build_script,
        "get_fundamentals_cache",
        lambda: SimpleNamespace(get_many=lambda symbols: cached),
    )

    publish_calls: list[dict[str, object]] = []
    export_calls: list[dict[str, object]] = []
    provider_snapshot_service = SimpleNamespace(
        build_market_snapshot_row=lambda **kwargs: {
            "symbol": kwargs["symbol"],
            "exchange": kwargs["exchange"],
            "row_hash": "row-hash",
            "normalized_payload": kwargs["normalized_payload"],
            "raw_payload": kwargs["raw_payload"],
        },
        publish_market_snapshot_run=lambda db, **kwargs: publish_calls.append(kwargs)
        or {
            "published": True,
            "force_published": kwargs.get("force_publish", False),
            "source_revision": "fundamentals_v1_cn:20260505121000",
            "snapshot_key": kwargs["snapshot_key"],
            "coverage": dict(kwargs["coverage_stats"]),
            "warnings": list(kwargs["warnings"]),
            "coverage_thresholds": {
                "market": "CN",
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
                "source_revision": "fundamentals_v1_cn:20260505121000",
            },
        )(),
        export_weekly_reference_bundle=lambda db, **kwargs: export_calls.append(kwargs)
        or {"bundle_path": str(kwargs["output_path"])},
    )
    monkeypatch.setattr(
        build_script, "get_provider_snapshot_service", lambda: provider_snapshot_service
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_weekly_reference_bundle",
            "--market",
            "CN",
            "--output-dir",
            str(tmp_path),
            "--max-runtime-minutes",
            "1.5",
            "--fetch-chunk-size",
            "2",
            "--allow-partial-publish",
        ],
    )

    assert build_script.main() == 0

    # Only the first chunk fetched; the second chunk's deadline check tripped.
    assert fetch_calls == [["600000.SS", "600001.SS"]]
    assert store_calls == [["600000.SS", "600001.SS"]]

    # Snapshot rows still cover all 3 symbols (skipped one inherited from cache).
    publish_kwargs = publish_calls[0]
    assert publish_kwargs["force_publish"] is True
    assert publish_kwargs["coverage_stats"]["partial_run"] is True
    assert publish_kwargs["coverage_stats"]["attempted_symbols"] == 2
    assert publish_kwargs["coverage_stats"]["skipped_due_to_deadline"] == 1
    assert publish_kwargs["coverage_stats"]["snapshot_symbols"] == 3
    deadline_warning = next(
        (w for w in publish_kwargs["warnings"] if "deadline reached" in w), None
    )
    assert deadline_warning is not None, publish_kwargs["warnings"]

    stdout = capsys.readouterr().out
    assert "[fundamentals] CN chunk 1/2" in stdout
    assert "deadline reached before chunk 2/2" in stdout
    assert export_calls, "Bundle export should still run after a partial publish"


def test_build_weekly_reference_bundle_deadline_blocks_when_partial_disabled(
    monkeypatch, tmp_path
):
    """Disabling partial publish blocks deadline-hit runs even when cache coverage passes."""

    active_rows = [
        _make_universe_row("600000.SS"),
        _make_universe_row("600001.SS"),
        _make_universe_row("000001.SZ"),
    ]
    fake_query = MagicMock()
    fake_query.filter.return_value.order_by.return_value.all.return_value = active_rows
    fake_db = MagicMock()
    fake_db.query.return_value = fake_query

    monkeypatch.setattr(build_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(build_script, "SessionLocal", lambda: _fake_session(fake_db))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(
        build_script,
        "OfficialMarketUniverseSourceService",
        lambda: SimpleNamespace(
            fetch_market_snapshot=lambda market: SimpleNamespace(
                market=market,
                source_name="cn_official",
                snapshot_id="cn-2026-05-05",
                snapshot_as_of="2026-05-05",
                source_metadata={},
                rows=tuple({"symbol": row.symbol} for row in active_rows),
            )
        ),
    )
    monkeypatch.setattr(
        build_script,
        "get_stock_universe_service",
        lambda: SimpleNamespace(
            seed_reconciliation_baseline_from_active_rows=lambda db, **kwargs: None,
            ingest_cn_snapshot_rows=lambda db, **kwargs: {"added": 3, "updated": 0, "deactivated": 0}
        ),
    )

    fake_clock = {"value": 0.0}

    def fake_monotonic() -> float:
        fake_clock["value"] += 60.0
        return fake_clock["value"]

    monkeypatch.setattr(build_script.time, "monotonic", fake_monotonic)

    monkeypatch.setattr(
        build_script,
        "get_hybrid_fundamentals_service",
        lambda: SimpleNamespace(
            yfinance_delay_per_ticker=1.5,
            fetch_fundamentals_batch=lambda symbols, **kwargs: {
                symbol: {"market_cap": 1000.0, "sector": "Industrials"} for symbol in symbols
            },
            store_all_caches=lambda data, _cache, **_kwargs: {
                "fundamentals_stored": len(data),
                "persisted_symbols": len(data),
                "failed_persistence_symbols": 0,
                "failed": 0,
                "provider_error_counts": {},
            },
        ),
    )
    cached = {
        "600000.SS": {"market_cap": 1000.0, "sector": "Industrials"},
        "600001.SS": {"market_cap": 1000.0, "sector": "Industrials"},
        "000001.SZ": {"market_cap": 999.0, "sector": "Industrials", "data_source": "bundle_import"},
    }
    monkeypatch.setattr(
        build_script,
        "get_fundamentals_cache",
        lambda: SimpleNamespace(get_many=lambda symbols: cached),
    )

    publish_calls: list[dict[str, object]] = []

    def fake_publish_market_snapshot_run(db, **kwargs):
        publish_calls.append(kwargs)
        return {
            "published": False,
            "source_revision": "fundamentals_v1_cn:20260505121000",
            "snapshot_key": kwargs["snapshot_key"],
            "coverage": dict(kwargs["coverage_stats"]),
            "warnings": list(kwargs["warnings"]),
            "coverage_thresholds": {
                "market": "CN",
                "active_coverage": 1.0,
                "min_active_coverage": 0.70,
                "missing_ratio": 0.0,
                "max_missing_ratio": 0.30,
            },
        }

    monkeypatch.setattr(
        build_script,
        "get_provider_snapshot_service",
        lambda: SimpleNamespace(
            build_market_snapshot_row=lambda **kwargs: {
                "symbol": kwargs["symbol"],
                "exchange": kwargs["exchange"],
                "row_hash": "row-hash",
                "normalized_payload": kwargs["normalized_payload"],
                "raw_payload": kwargs["raw_payload"],
            },
            publish_market_snapshot_run=fake_publish_market_snapshot_run,
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_weekly_reference_bundle",
            "--market",
            "CN",
            "--output-dir",
            str(tmp_path),
            "--max-runtime-minutes",
            "1.5",
            "--fetch-chunk-size",
            "2",
        ],
    )

    with pytest.raises(RuntimeError, match="Partial publish disabled"):
        build_script.main()

    publish_kwargs = publish_calls[0]
    assert publish_kwargs["publish"] is False
    assert publish_kwargs["force_publish"] is False
    assert publish_kwargs["coverage_stats"]["partial_run"] is True
    assert publish_kwargs["coverage_stats"]["snapshot_symbols"] == 3


def _patch_cn_dependencies(
    monkeypatch,
    *,
    raise_universe: bool,
    hybrid_failures: int = 0,
    universe_error: Exception | None = None,
):
    """Wire the supporting Asia-bundle services for stale-universe fallback tests."""

    def fetch(market):
        if raise_universe:
            raise universe_error or RuntimeError("AKShare CN spot disconnected")
        return SimpleNamespace(
            market=market,
            source_name="cn_official",
            snapshot_id="cn-2026-05-09",
            snapshot_as_of="2026-05-09",
            source_metadata={},
            rows=(),
        )

    monkeypatch.setattr(
        build_script,
        "OfficialMarketUniverseSourceService",
        lambda: SimpleNamespace(fetch_market_snapshot=fetch),
    )
    monkeypatch.setattr(
        build_script,
        "get_stock_universe_service",
        lambda: SimpleNamespace(
            seed_reconciliation_baseline_from_active_rows=lambda db, **kwargs: None,
            ingest_cn_snapshot_rows=lambda db, **kwargs: {
                "added": 0,
                "updated": 0,
                "deactivated": 0,
            }
        ),
    )

    monkeypatch.setattr(
        build_script,
        "get_hybrid_fundamentals_service",
        lambda: SimpleNamespace(
            yfinance_delay_per_ticker=1.5,
            fetch_fundamentals_batch=lambda symbols, **kwargs: {
                symbol: {"market_cap": 1000.0, "sector": "Industrials"} for symbol in symbols
            },
            store_all_caches=lambda data, _cache, **_kwargs: {
                "fundamentals_stored": len(data),
                "persisted_symbols": len(data),
                "failed_persistence_symbols": 0,
                "failed": hybrid_failures,
                "provider_error_counts": {},
            },
        ),
    )


def _make_cn_db_mock(active_rows: list[SimpleNamespace], *, seeded_count: int | None = None):
    fake_query = MagicMock()
    fake_query.filter.return_value.order_by.return_value.all.return_value = active_rows
    fake_query.filter.return_value.count.return_value = (
        seeded_count if seeded_count is not None else len(active_rows)
    )
    fake_db = MagicMock()
    fake_db.query.return_value = fake_query
    return fake_db


def test_build_asia_bundle_falls_back_to_seeded_universe_when_official_fetch_fails(
    monkeypatch, tmp_path, capsys
):
    """When AKShare listing fails and partial publish is on, reuse seeded rows."""
    # Within github_weekly_reference_max_age_days: an older seed is refused.
    published_at = datetime.utcnow() - timedelta(days=2)
    active_rows = [
        _make_universe_row("600000.SS"),
        _make_universe_row("000001.SZ"),
    ]
    fake_db = _make_cn_db_mock(active_rows)

    monkeypatch.setattr(build_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(build_script, "SessionLocal", lambda: _fake_session(fake_db))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    _patch_cn_dependencies(monkeypatch, raise_universe=True)

    cached = {
        "600000.SS": {"market_cap": 1000.0, "sector": "Industrials"},
        "000001.SZ": {"market_cap": 1000.0, "sector": "Industrials"},
    }
    monkeypatch.setattr(
        build_script,
        "get_fundamentals_cache",
        lambda: SimpleNamespace(get_many=lambda symbols: cached),
    )

    publish_calls: list[dict[str, object]] = []
    export_calls: list[dict[str, object]] = []
    provider_snapshot_service = SimpleNamespace(
        build_market_snapshot_row=lambda **kwargs: {
            "symbol": kwargs["symbol"],
            "exchange": kwargs["exchange"],
            "row_hash": "row-hash",
            "normalized_payload": kwargs["normalized_payload"],
            "raw_payload": kwargs["raw_payload"],
        },
        publish_market_snapshot_run=lambda db, **kwargs: publish_calls.append(kwargs)
        or {
            "published": True,
            "force_published": kwargs.get("force_publish", False),
            "source_revision": "fundamentals_v1_cn:20260509121000",
            "snapshot_key": kwargs["snapshot_key"],
            "coverage": dict(kwargs["coverage_stats"]),
            "warnings": list(kwargs["warnings"]),
            "coverage_thresholds": {
                "market": "CN",
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
                "source_revision": "fundamentals_v1_cn:20260509121000",
                "coverage_stats_json": "{}",
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
            "CN",
            "--output-dir",
            str(tmp_path),
            "--allow-partial-publish",
        ],
    )

    assert build_script.main() == 0

    publish_kwargs = publish_calls[0]
    assert publish_kwargs["force_publish"] is True
    assert publish_kwargs["coverage_stats"]["stale_universe"] is True
    assert publish_kwargs["coverage_stats"]["partial_run"] is True
    stale_warning = next(
        (w for w in publish_kwargs["warnings"] if "Official CN universe fetch failed" in w),
        None,
    )
    assert stale_warning is not None, publish_kwargs["warnings"]
    assert "AKShare CN spot disconnected" in stale_warning
    assert export_calls, "Bundle export should still run after a stale-universe fallback"

    stdout = capsys.readouterr().out
    assert "[universe] CN official fetch failed" in stdout
    assert "falling back to 2 seeded rows from fundamentals_v1_cn:20260509121000" in stdout


def test_in_bundle_drops_seeded_bse_scrip_codes_only_for_in():
    """#480: IN is NSE-only; scrip codes seeded from prior bundles are dropped."""
    rows = [_make_universe_row(symbol) for symbol in ("RELIANCE.NS", "500325.BO", "TANFAC.BO")]

    assert [r.symbol for r in build_script._without_excluded_listings("IN", rows)] == ["RELIANCE.NS", "TANFAC.BO"]
    assert build_script._without_excluded_listings("JP", rows) == rows


def test_au_bundle_drops_seeded_securitisation_trusts():
    """#481: securitisation trusts are debt Yahoo never prices; equity trusts stay."""
    rows = [
        SimpleNamespace(symbol="PUT.AX", name="PUMA SERIES 2023-1 TRUST"),
        SimpleNamespace(symbol="HC1.AX", name="HOUSEHOLD CAPITAL 2025-1 RMBS TRUST"),
        SimpleNamespace(symbol="CDP.AX", name="CARINDALE PROPERTY TRUST"),
        SimpleNamespace(symbol="BHP.AX", name="BHP GROUP LIMITED"),
    ]

    assert [r.symbol for r in build_script._without_excluded_listings("AU", rows)] == ["CDP.AX", "BHP.AX"]


def test_build_asia_bundle_reraises_when_no_seeded_universe_rows(monkeypatch, tmp_path):
    """If AKShare fails AND no prior-week rows are seeded, surface the original error."""
    fake_db = _make_cn_db_mock([], seeded_count=0)

    monkeypatch.setattr(build_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(build_script, "SessionLocal", lambda: _fake_session(fake_db))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    _patch_cn_dependencies(monkeypatch, raise_universe=True)
    monkeypatch.setattr(
        build_script,
        "get_fundamentals_cache",
        lambda: SimpleNamespace(get_many=lambda symbols: {}),
    )
    monkeypatch.setattr(
        build_script,
        "get_provider_snapshot_service",
        lambda: SimpleNamespace(
            build_market_snapshot_row=lambda **kwargs: {},
            publish_market_snapshot_run=lambda db, **kwargs: {"published": False, "warnings": []},
        ),
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_weekly_reference_bundle",
            "--market",
            "CN",
            "--output-dir",
            str(tmp_path),
            "--allow-partial-publish",
        ],
    )

    with pytest.raises(RuntimeError, match="AKShare CN spot disconnected"):
        build_script.main()


def test_build_asia_bundle_reraises_when_partial_publish_disabled(monkeypatch, tmp_path):
    """Without --allow-partial-publish, an AKShare failure must abort the run."""
    active_rows = [_make_universe_row("600000.SS")]
    fake_db = _make_cn_db_mock(active_rows)

    monkeypatch.setattr(build_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(build_script, "SessionLocal", lambda: _fake_session(fake_db))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    _patch_cn_dependencies(monkeypatch, raise_universe=True)
    monkeypatch.setattr(
        build_script,
        "get_fundamentals_cache",
        lambda: SimpleNamespace(get_many=lambda symbols: {}),
    )
    monkeypatch.setattr(
        build_script,
        "get_provider_snapshot_service",
        lambda: SimpleNamespace(
            build_market_snapshot_row=lambda **kwargs: {},
            publish_market_snapshot_run=lambda db, **kwargs: {"published": False, "warnings": []},
        ),
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_weekly_reference_bundle",
            "--market",
            "CN",
            "--output-dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(RuntimeError, match="AKShare CN spot disconnected"):
        build_script.main()


def test_build_weekly_reference_bundle_resumes_partial_seed_by_skipping_cached_symbols(
    monkeypatch, tmp_path
):
    active_rows = [
        _make_universe_row("600000.SS"),
        _make_universe_row("600001.SS"),
        _make_universe_row("000001.SZ"),
    ]
    fake_db = _make_cn_db_mock(active_rows)

    monkeypatch.setattr(build_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(build_script, "SessionLocal", lambda: _fake_session(fake_db))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    _patch_cn_dependencies(monkeypatch, raise_universe=False)

    fetch_calls: list[list[str]] = []
    cached = {
        "600000.SS": {
            "market_cap": 999.0,
            "sector": "Industrials",
            "data_source": "bundle_import",
        },
    }

    def fake_fetch_fundamentals_batch(symbols, **kwargs):
        fetch_calls.append(list(symbols))
        kwargs["progress_callback"](len(symbols), len(symbols))
        return {symbol: {"market_cap": 1000.0, "sector": "Industrials"} for symbol in symbols}

    def fake_store_all_caches(data, _cache, **_kwargs):
        cached.update(data)
        return {
            "fundamentals_stored": len(data),
            "persisted_symbols": len(data),
            "failed_persistence_symbols": 0,
            "failed": 0,
            "provider_error_counts": {},
        }

    monkeypatch.setattr(
        build_script,
        "get_hybrid_fundamentals_service",
        lambda: SimpleNamespace(
            yfinance_delay_per_ticker=1.5,
            fetch_fundamentals_batch=fake_fetch_fundamentals_batch,
            store_all_caches=fake_store_all_caches,
        ),
    )
    monkeypatch.setattr(
        build_script,
        "get_fundamentals_cache",
        lambda: SimpleNamespace(get_many=lambda symbols: {s: cached[s] for s in symbols if s in cached}),
    )

    published_at = datetime(2026, 5, 5, 12, 10, 0)
    publish_calls: list[dict[str, object]] = []
    export_calls: list[dict[str, object]] = []
    provider_snapshot_service = SimpleNamespace(
        build_market_snapshot_row=lambda **kwargs: {
            "symbol": kwargs["symbol"],
            "exchange": kwargs["exchange"],
            "row_hash": "row-hash",
            "normalized_payload": kwargs["normalized_payload"],
            "raw_payload": kwargs["raw_payload"],
        },
        publish_market_snapshot_run=lambda db, **kwargs: publish_calls.append(kwargs)
        or {
            "published": True,
            "force_published": False,
            "source_revision": "fundamentals_v1_cn:20260505121000",
            "snapshot_key": kwargs["snapshot_key"],
            "coverage": dict(kwargs["coverage_stats"]),
            "warnings": list(kwargs["warnings"]),
            "coverage_thresholds": {
                "market": "CN",
                "active_coverage": 1.0,
                "min_active_coverage": 0.70,
                "missing_ratio": 0.0,
                "max_missing_ratio": 0.30,
            },
        },
        get_published_run=lambda db, snapshot_key: SimpleNamespace(
            published_at=published_at,
            created_at=published_at,
            source_revision="fundamentals_v1_cn:20260505121000",
            coverage_stats_json=(
                '{"partial_run": true, "active_symbols": 3, '
                '"snapshot_symbols": 1, "missing_active_symbols": 2}'
            ),
        ),
        export_weekly_reference_bundle=lambda db, **kwargs: export_calls.append(kwargs)
        or {"bundle_path": str(kwargs["output_path"])},
    )
    monkeypatch.setattr(
        build_script, "get_provider_snapshot_service", lambda: provider_snapshot_service
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_weekly_reference_bundle",
            "--market",
            "CN",
            "--output-dir",
            str(tmp_path),
            "--fetch-chunk-size",
            "2",
            "--resume-partial-seed",
        ],
    )

    assert build_script.main() == 0

    assert fetch_calls == [["600001.SS", "000001.SZ"]]
    publish_kwargs = publish_calls[0]
    assert publish_kwargs["coverage_stats"]["seeded_symbols"] == 1
    assert publish_kwargs["coverage_stats"]["fetch_symbols"] == 2
    assert publish_kwargs["coverage_stats"]["attempted_symbols"] == 2
    assert publish_kwargs["coverage_stats"]["partial_run"] is False
    assert publish_kwargs["coverage_stats"]["snapshot_symbols"] == 3
    assert export_calls, "Bundle export should run after completing a resumed seed"


def test_build_weekly_reference_bundle_does_not_resume_full_coverage_partial_seed(
    monkeypatch, tmp_path
):
    active_rows = [
        _make_universe_row("600000.SS"),
        _make_universe_row("600001.SS"),
        _make_universe_row("000001.SZ"),
    ]
    fake_db = _make_cn_db_mock(active_rows)

    monkeypatch.setattr(build_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(build_script, "SessionLocal", lambda: _fake_session(fake_db))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    _patch_cn_dependencies(monkeypatch, raise_universe=False)

    fetch_calls: list[list[str]] = []
    cached = {
        row.symbol: {
            "market_cap": 999.0,
            "sector": "Industrials",
            "data_source": "bundle_import",
        }
        for row in active_rows
    }

    def fake_fetch_fundamentals_batch(symbols, **kwargs):
        fetch_calls.append(list(symbols))
        kwargs["progress_callback"](len(symbols), len(symbols))
        return {symbol: {"market_cap": 1000.0, "sector": "Industrials"} for symbol in symbols}

    def fake_store_all_caches(data, _cache, **_kwargs):
        cached.update(data)
        return {
            "fundamentals_stored": len(data),
            "persisted_symbols": len(data),
            "failed_persistence_symbols": 0,
            "failed": 0,
            "provider_error_counts": {},
        }

    monkeypatch.setattr(
        build_script,
        "get_hybrid_fundamentals_service",
        lambda: SimpleNamespace(
            yfinance_delay_per_ticker=1.5,
            fetch_fundamentals_batch=fake_fetch_fundamentals_batch,
            store_all_caches=fake_store_all_caches,
        ),
    )
    monkeypatch.setattr(
        build_script,
        "get_fundamentals_cache",
        lambda: SimpleNamespace(get_many=lambda symbols: {s: cached[s] for s in symbols if s in cached}),
    )

    published_at = datetime(2026, 5, 5, 12, 10, 0)
    publish_calls: list[dict[str, object]] = []
    provider_snapshot_service = SimpleNamespace(
        build_market_snapshot_row=lambda **kwargs: {
            "symbol": kwargs["symbol"],
            "exchange": kwargs["exchange"],
            "row_hash": "row-hash",
            "normalized_payload": kwargs["normalized_payload"],
            "raw_payload": kwargs["raw_payload"],
        },
        publish_market_snapshot_run=lambda db, **kwargs: publish_calls.append(kwargs)
        or {
            "published": True,
            "force_published": False,
            "source_revision": "fundamentals_v1_cn:20260505121000",
            "snapshot_key": kwargs["snapshot_key"],
            "coverage": dict(kwargs["coverage_stats"]),
            "warnings": list(kwargs["warnings"]),
            "coverage_thresholds": {
                "market": "CN",
                "active_coverage": 1.0,
                "min_active_coverage": 0.70,
                "missing_ratio": 0.0,
                "max_missing_ratio": 0.30,
            },
        },
        get_published_run=lambda db, snapshot_key: SimpleNamespace(
            published_at=published_at,
            created_at=published_at,
            source_revision="fundamentals_v1_cn:20260505121000",
            coverage_stats_json=(
                '{"partial_run": true, "active_symbols": 3, '
                '"snapshot_symbols": 3, "missing_active_symbols": 0}'
            ),
        ),
        export_weekly_reference_bundle=lambda db, **kwargs: {"bundle_path": str(kwargs["output_path"])},
    )
    monkeypatch.setattr(
        build_script, "get_provider_snapshot_service", lambda: provider_snapshot_service
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_weekly_reference_bundle",
            "--market",
            "CN",
            "--output-dir",
            str(tmp_path),
            "--fetch-chunk-size",
            "2",
            "--resume-partial-seed",
        ],
    )

    assert build_script.main() == 0

    assert fetch_calls == [["600000.SS", "600001.SS"], ["000001.SZ"]]
    publish_kwargs = publish_calls[0]
    assert publish_kwargs["coverage_stats"]["seeded_symbols"] == 0
    assert publish_kwargs["coverage_stats"]["fetch_symbols"] == 3
    assert publish_kwargs["coverage_stats"]["attempted_symbols"] == 3


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


def _us_rows(*symbols: str) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            symbol=symbol,
            market="US",
            exchange="NASDAQ",
            name=f"{symbol} Inc.",
            sector="Technology",
            industry="Software",
            market_cap=1_000_000_000,
            currency="USD",
            timezone="America/New_York",
            local_code=symbol,
        )
        for symbol in symbols
    ]


def _run_us_build_with_finviz_error(
    monkeypatch,
    tmp_path,
    *,
    seed_run,
    error=None,
    soft_block=False,
    active_symbols=("AAPL", "MSFT"),
    finviz_rows=(),
    seed_lacks=(),
):
    """Drive the US build with create_snapshot_run raising ``error`` (default: Finviz 403).

    ``soft_block`` instead returns a blocked run holding only ``finviz_rows``,
    as when Finviz serves pages without a screener table.
    """
    import requests

    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.order_by.return_value.all.return_value = _us_rows(
        *active_symbols
    )
    monkeypatch.setattr(build_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(build_script, "SessionLocal", lambda: _fake_session(fake_db))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(build_script.settings, "github_weekly_reference_max_age_days", 8)
    monkeypatch.setattr(
        build_script,
        "get_fundamentals_cache",
        lambda: SimpleNamespace(
            get_many=lambda symbols: {symbol: {"symbol": symbol, "market_cap": 1.0} for symbol in symbols}
        ),
    )
    monkeypatch.setattr(
        build_script,
        "get_stock_universe_service",
        lambda: SimpleNamespace(
            seed_reconciliation_baseline_from_active_rows=lambda db, **kwargs: None,
            populate_universe=lambda db: {"total": 0},
        ),
    )

    def refuse(db, **kwargs):
        if soft_block:
            return {
                "run_id": 42,
                "published": False,
                "warnings": ["Active snapshot coverage 0.00% below minimum 98.00%"],
            }
        raise error or requests.HTTPError(
            "403 Client Error: Forbidden for url: https://finviz.com/screener.ashx"
        )

    publish_calls: list[dict[str, object]] = []
    export_calls: list[dict[str, object]] = []
    published_run = SimpleNamespace(
        published_at=datetime.utcnow(),
        created_at=datetime.utcnow(),
        source_revision="fundamentals_v1_us:20261003161500-seeded-fallback",
    )
    def snapshot_row(symbol, payload, raw):
        return SimpleNamespace(
            symbol=symbol,
            exchange="NASDAQ",
            row_hash=f"hash-{symbol.lower()}",
            normalized_payload_json=json.dumps(payload),
            raw_payload_json=json.dumps(raw),
        )

    rows_by_run = {
        42: [snapshot_row(s, {"symbol": s}, {"overview": {"Ticker": s}}) for s in finviz_rows],
    }
    if seed_run is not None and getattr(seed_run, "id", None) is not None:
        rows_by_run[seed_run.id] = [
            snapshot_row(s, {"symbol": s, "market_cap": 2.0}, {"overview": {"Ticker": s}})
            for s in active_symbols
            if s not in seed_lacks
        ]
    monkeypatch.setattr(build_script, "_run_rows", lambda db, run_id: rows_by_run.get(run_id, []))
    provider_snapshot_service = SimpleNamespace(
        create_snapshot_run=refuse,
        build_market_snapshot_row=lambda **kwargs: {
            "symbol": kwargs["symbol"],
            "exchange": kwargs["exchange"],
            "row_hash": f"hash-{kwargs['symbol'].lower()}",
            "normalized_payload": kwargs["normalized_payload"],
            "raw_payload": kwargs["raw_payload"],
        },
        publish_market_snapshot_run=lambda db, **kwargs: publish_calls.append(kwargs)
        or {
            "published": True,
            "coverage": dict(kwargs["coverage_stats"]),
            "warnings": list(kwargs["warnings"]),
        },
        # The seed until this build publishes; then the new run.
        get_published_run=lambda db, snapshot_key: published_run if publish_calls else seed_run,
        hydrate_published_snapshot=lambda db, snapshot_key, progress_callback=None: {"hydrated": 2},
        export_weekly_reference_bundle=lambda db, **kwargs: export_calls.append(kwargs)
        or {"bundle_path": str(kwargs["output_path"])},
    )
    monkeypatch.setattr(build_script, "get_provider_snapshot_service", lambda: provider_snapshot_service)
    monkeypatch.setattr(
        "sys.argv",
        ["build_weekly_reference_bundle", "--market", "US", "--output-dir", str(tmp_path)],
    )
    return publish_calls, export_calls


def _seed_run(*, age_days: int, coverage: dict | None = None) -> SimpleNamespace:
    published_at = datetime.utcnow() - timedelta(days=age_days)
    return SimpleNamespace(
        id=7,
        published_at=published_at,
        created_at=published_at,
        source_revision="fundamentals_v1_us:20260926171427",
        coverage_stats_json=json.dumps(coverage) if coverage is not None else None,
    )


def test_build_weekly_reference_bundle_us_reuses_recent_seed_when_finviz_refuses(
    monkeypatch, tmp_path
):
    """#520: Finviz HTTP 403 with a recent seed publishes the seed, labelled as such."""
    seed = _seed_run(age_days=7)
    publish_calls, export_calls = _run_us_build_with_finviz_error(monkeypatch, tmp_path, seed_run=seed)

    assert build_script.main() == 0

    publish_kwargs = publish_calls[0]
    assert [row["symbol"] for row in publish_kwargs["rows"]] == ["AAPL", "MSFT"]
    # Rows come from the validated seed run (market_cap 2.0), not the shared
    # cache (market_cap 1.0), so the recorded provenance describes them.
    assert all(row["normalized_payload"]["market_cap"] == 2.0 for row in publish_kwargs["rows"])
    assert all(
        row["raw_payload"]
        == {"source": "prior_weekly_reference_seed", "seed_source_revision": seed.source_revision}
        for row in publish_kwargs["rows"]
    )
    coverage = publish_kwargs["coverage_stats"]
    assert coverage["finviz_snapshot_failed"] is True
    assert coverage["seed_source_revision"] == seed.source_revision
    assert coverage["seed_as_of_date"] == seed.published_at.date().isoformat()
    assert coverage["backfilled_active_symbols"] == 2
    assert publish_kwargs["source_revision"].endswith("-seeded-fallback")
    assert any("Finviz snapshot fetch failed: HTTPError: 403" in w for w in publish_kwargs["warnings"])
    assert export_calls, "Bundle export should run after the seed is republished"


@pytest.mark.parametrize(
    ("seed_run", "reason"),
    [
        (None, "No prior weekly reference seed"),
        (_seed_run(age_days=9), "max age 8 day(s)"),
        # A seed that itself reused an older seed carries that older data date.
        (_seed_run(age_days=1, coverage={"seed_as_of_date": "2000-01-01"}), "as of 2000-01-01"),
        (_seed_run(age_days=1, coverage={"seed_as_of_date": "last week"}), "no usable data date"),
        (
            SimpleNamespace(
                published_at=None,
                created_at=None,
                source_revision="fundamentals_v1_us:20260926171427",
                coverage_stats_json="null",
            ),
            "no usable data date",
        ),
    ],
)
def test_build_weekly_reference_bundle_us_fails_without_a_usable_seed_when_finviz_refuses(
    monkeypatch, tmp_path, seed_run, reason
):
    publish_calls, export_calls = _run_us_build_with_finviz_error(
        monkeypatch, tmp_path, seed_run=seed_run
    )

    with pytest.raises(RuntimeError, match="did not publish") as excinfo:
        build_script.main()

    assert "Finviz snapshot fetch failed" in str(excinfo.value)
    assert reason in str(excinfo.value)
    assert publish_calls == []
    assert export_calls == []


def test_build_weekly_reference_bundle_us_does_not_swallow_non_provider_errors(
    monkeypatch, tmp_path
):
    _run_us_build_with_finviz_error(
        monkeypatch, tmp_path, seed_run=_seed_run(age_days=1), error=KeyError("Ticker")
    )

    with pytest.raises(KeyError):
        build_script.main()


def test_build_weekly_reference_bundle_us_records_seed_reuse_in_step_summary(
    monkeypatch, tmp_path, capsys
):
    """#520: a green run built from the prior seed says so in the summary and log."""
    seed = _seed_run(age_days=7)
    _run_us_build_with_finviz_error(monkeypatch, tmp_path, seed_run=seed)
    summary_path = tmp_path / "github-step-summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))

    assert build_script.main() == 0

    as_of = seed.published_at.date().isoformat()
    assert (
        f"| Data source | prior seed {seed.source_revision} (data as of {as_of}); "
        "Finviz snapshot failed |"
    ) in summary_path.read_text(encoding="utf-8")
    assert "::warning title=Weekly reference US reused prior seed::" in capsys.readouterr().out


@pytest.mark.parametrize(("age_days", "published"), [(7, True), (9, False)])
def test_build_weekly_reference_bundle_us_treats_an_empty_finviz_snapshot_as_failed(
    monkeypatch, tmp_path, age_days, published
):
    """#520: a Finviz page with no screener rows gets the same seed-age policy as a 403."""
    seed = _seed_run(age_days=age_days)
    publish_calls, _ = _run_us_build_with_finviz_error(
        monkeypatch, tmp_path, seed_run=seed, soft_block=True
    )

    if not published:
        with pytest.raises(RuntimeError, match="max age 8 day"):
            build_script.main()
        assert publish_calls == []
        return

    assert build_script.main() == 0
    coverage = publish_calls[0]["coverage_stats"]
    assert coverage["finviz_snapshot_failed"] is True
    assert coverage["seed_as_of_date"] == seed.published_at.date().isoformat()


def test_build_weekly_reference_bundle_us_treats_a_mostly_empty_finviz_snapshot_as_failed(
    monkeypatch, tmp_path
):
    """#520: Finviz serving page one and then challenge pages is a failure, not a backfill."""
    seed = _seed_run(age_days=7)
    publish_calls, _ = _run_us_build_with_finviz_error(
        monkeypatch,
        tmp_path,
        seed_run=seed,
        soft_block=True,
        active_symbols=("AAPL", "AMZN", "MSFT", "NVDA"),
        finviz_rows=("AAPL",),
    )

    assert build_script.main() == 0

    publish_kwargs = publish_calls[0]
    assert publish_kwargs["coverage_stats"]["seed_as_of_date"] == seed.published_at.date().isoformat()
    assert any("rows for only 1 of 4 active US symbols" in w for w in publish_kwargs["warnings"])
    rows = {row["symbol"]: row for row in publish_kwargs["rows"]}
    assert rows["AAPL"]["raw_payload"] == {"overview": {"Ticker": "AAPL"}}
    assert rows["MSFT"]["raw_payload"] == {
        "source": "prior_weekly_reference_seed",
        "seed_source_revision": seed.source_revision,
    }


def test_build_weekly_reference_bundle_us_backfills_seed_misses_from_the_cache(
    monkeypatch, tmp_path
):
    """#520: a symbol the reused seed lacks (e.g. a new listing) still gets a cache
    row, labelled as such, instead of being dropped from the fallback."""
    seed = _seed_run(age_days=7)
    publish_calls, _ = _run_us_build_with_finviz_error(
        monkeypatch,
        tmp_path,
        seed_run=seed,
        active_symbols=("AAPL", "MSFT", "NEWCO"),
        seed_lacks=("NEWCO",),
    )

    assert build_script.main() == 0

    rows = {row["symbol"]: row for row in publish_calls[0]["rows"]}
    assert sorted(rows) == ["AAPL", "MSFT", "NEWCO"]
    assert rows["AAPL"]["raw_payload"]["source"] == "prior_weekly_reference_seed"
    assert rows["NEWCO"]["raw_payload"] == {"source": "seeded_weekly_reference_cache"}
    assert publish_calls[0]["coverage_stats"]["missing_active_symbols"] == 0
    assert any(
        "Republished 2 US active symbols from the prior weekly reference seed" in w
        and "backfilled 1 the seed lacked from the seeded cache" in w
        for w in publish_calls[0]["warnings"]
    )


def test_build_weekly_reference_bundle_us_publishes_seed_rows_when_cache_backfill_fails(
    monkeypatch, tmp_path
):
    """#520: a broken cache must not discard the validated seed rows; the coverage
    gate decides whether the seed alone is enough."""
    seed = _seed_run(age_days=7)
    publish_calls, _ = _run_us_build_with_finviz_error(
        monkeypatch,
        tmp_path,
        seed_run=seed,
        active_symbols=("AAPL", "MSFT", "NEWCO"),
        seed_lacks=("NEWCO",),
    )

    def broken_get_many(symbols):
        raise ConnectionError("cache unavailable")

    monkeypatch.setattr(
        build_script, "get_fundamentals_cache", lambda: SimpleNamespace(get_many=broken_get_many)
    )

    assert build_script.main() == 0

    rows = [row["symbol"] for row in publish_calls[0]["rows"]]
    assert rows == ["AAPL", "MSFT"]
    assert publish_calls[0]["coverage_stats"]["missing_active_symbols"] == 1


@pytest.mark.parametrize(("age_days", "published"), [(7, True), (9, False)])
def test_build_weekly_reference_bundle_us_partial_backfill_requires_a_recent_seed(
    monkeypatch, tmp_path, age_days, published
):
    """#520: Finviz covering most symbols still only backfills the rest from a seed
    within the max age, and the bundle keeps its own date."""
    seed = _seed_run(age_days=age_days)
    publish_calls, _ = _run_us_build_with_finviz_error(
        monkeypatch,
        tmp_path,
        seed_run=seed,
        soft_block=True,
        active_symbols=("AAPL", "MSFT", "NVDA"),
        finviz_rows=("AAPL", "MSFT"),
    )

    if not published:
        with pytest.raises(RuntimeError, match="max age 8 day"):
            build_script.main()
        assert publish_calls == []
        return

    assert build_script.main() == 0
    coverage = publish_calls[0]["coverage_stats"]
    assert "seed_as_of_date" not in coverage  # mostly fresh: not backdated
    assert coverage["backfill_seed_as_of_date"] == seed.published_at.date().isoformat()
    assert coverage["backfill_seed_source_revision"] == seed.source_revision
    rows = {row["symbol"]: row for row in publish_calls[0]["rows"]}
    assert rows["NVDA"]["raw_payload"] == {"source": "seeded_weekly_reference_cache"}


def test_build_weekly_reference_bundle_us_warns_when_cache_backfill_fails(monkeypatch, tmp_path):
    seed = _seed_run(age_days=7)
    publish_calls, _ = _run_us_build_with_finviz_error(
        monkeypatch,
        tmp_path,
        seed_run=seed,
        active_symbols=("AAPL", "MSFT", "NEWCO"),
        seed_lacks=("NEWCO",),
    )

    def broken_get_many(symbols):
        raise ConnectionError("cache unavailable")

    monkeypatch.setattr(
        build_script, "get_fundamentals_cache", lambda: SimpleNamespace(get_many=broken_get_many)
    )

    assert build_script.main() == 0
    assert any(
        "Seeded cache backfill failed: ConnectionError: cache unavailable" in w
        for w in publish_calls[0]["warnings"]
    )


def test_build_weekly_reference_bundle_us_reuses_the_seed_when_the_finviz_reader_fails(
    monkeypatch, tmp_path
):
    """A slice Finviz can't serve completely is a provider failure: seed fallback."""
    from app.services.finviz_screener_slices import FinvizSliceTooLarge

    seed = _seed_run(age_days=7)
    publish_calls, _ = _run_us_build_with_finviz_error(
        monkeypatch,
        tmp_path,
        seed_run=seed,
        error=FinvizSliceTooLarge("Finviz slice 'exch_nyse' has 2500 rows"),
    )

    assert build_script.main() == 0
    assert publish_calls[0]["coverage_stats"]["finviz_snapshot_failed"] is True
    assert any(
        "Finviz snapshot fetch failed: FinvizSliceTooLarge" in w for w in publish_calls[0]["warnings"]
    )


# ── Stale-universe fallback policy (#521) ───────────────────────────────


def _jp_snapshot_service(publish_calls, *, seed_published_at, seed_coverage=None):
    return SimpleNamespace(
        build_market_snapshot_row=lambda **kwargs: {
            "symbol": kwargs["symbol"],
            "exchange": kwargs["exchange"],
            "row_hash": "row-hash",
            "normalized_payload": kwargs["normalized_payload"],
            "raw_payload": kwargs["raw_payload"],
        },
        publish_market_snapshot_run=lambda db, **kwargs: publish_calls.append(kwargs)
        or {
            "published": True,
            "source_revision": "fundamentals_v1_jp:20261005",
            "snapshot_key": kwargs["snapshot_key"],
            "coverage": dict(kwargs["coverage_stats"]),
            "warnings": list(kwargs["warnings"]),
        },
        get_published_run=lambda db, snapshot_key: SimpleNamespace(
            published_at=seed_published_at,
            created_at=seed_published_at,
            source_revision="fundamentals_v1_jp:seed",
            coverage_stats_json=json.dumps(seed_coverage or {}),
        ),
        export_weekly_reference_bundle=lambda db, **kwargs: {"bundle_path": str(kwargs["output_path"])},
    )


def _run_jp_stale_universe(
    monkeypatch,
    tmp_path,
    *,
    seed_published_at,
    seeded_count=None,
    flags=None,
    seed_coverage=None,
    universe_error=None,
):
    active_rows = [_make_universe_row("7203.T", market="JP"), _make_universe_row("1301.T", market="JP")]
    fake_db = _make_cn_db_mock(active_rows, seeded_count=seeded_count)
    monkeypatch.setattr(build_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(build_script, "SessionLocal", lambda: _fake_session(fake_db))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    _patch_cn_dependencies(
        monkeypatch,
        raise_universe=True,
        universe_error=universe_error or requests.ConnectionError("JPX listing unreachable"),
    )
    monkeypatch.setattr(
        build_script,
        "get_fundamentals_cache",
        lambda: SimpleNamespace(get_many=lambda symbols: {}),
    )
    publish_calls: list[dict] = []
    monkeypatch.setattr(
        build_script,
        "get_provider_snapshot_service",
        lambda: _jp_snapshot_service(
            publish_calls, seed_published_at=seed_published_at, seed_coverage=seed_coverage
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["build_weekly_reference_bundle", "--market", "JP", "--output-dir", str(tmp_path),
         *(flags if flags is not None else ["--allow-stale-universe"])],
    )
    return build_script.main(), publish_calls


def test_stale_universe_flag_reuses_a_recent_seed_and_records_its_age(monkeypatch, tmp_path):
    seed_at = datetime.utcnow() - timedelta(days=3)

    exit_code, publish_calls = _run_jp_stale_universe(monkeypatch, tmp_path, seed_published_at=seed_at)

    assert exit_code == 0
    kwargs = publish_calls[0]
    coverage = kwargs["coverage_stats"]
    assert coverage["stale_universe"] is True
    assert coverage["universe_seed_source_revision"] == "fundamentals_v1_jp:seed"
    assert coverage["universe_seed_as_of_date"] == seed_at.date().isoformat()
    assert "JPX listing unreachable" in coverage["universe_error"]
    # Only the universe is reused; the coverage gate still applies.
    assert kwargs["force_publish"] is False
    assert any(seed_at.date().isoformat() in warning for warning in kwargs["warnings"])


def test_stale_universe_refuses_a_seed_past_the_max_age(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match=r"JPX listing unreachable.*max age 8 day"):
        _run_jp_stale_universe(
            monkeypatch, tmp_path, seed_published_at=datetime.utcnow() - timedelta(days=20)
        )


def test_stale_universe_fails_clearly_without_a_seed(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match=r"JPX listing unreachable.*no prior-week universe"):
        _run_jp_stale_universe(
            monkeypatch, tmp_path, seed_published_at=datetime.utcnow(), seeded_count=0
        )


def test_official_source_failure_still_fails_without_a_fallback_flag(monkeypatch, tmp_path):
    with pytest.raises(requests.ConnectionError, match="JPX listing unreachable"):
        _run_jp_stale_universe(
            monkeypatch, tmp_path, seed_published_at=datetime.utcnow(), flags=[]
        )


def test_stale_universe_dates_a_reused_universe_by_its_original_data(monkeypatch, tmp_path):
    # Published two days ago, but it reused a universe from three weeks ago.
    old_universe = (datetime.utcnow() - timedelta(days=21)).date().isoformat()
    with pytest.raises(RuntimeError, match="max age 8 day"):
        _run_jp_stale_universe(
            monkeypatch,
            tmp_path,
            seed_published_at=datetime.utcnow() - timedelta(days=2),
            seed_coverage={"stale_universe": True, "universe_seed_as_of_date": old_universe},
        )


def test_stale_universe_refuses_an_undated_reused_universe(monkeypatch, tmp_path):
    # Bundles written before #521 recorded stale_universe without a date.
    with pytest.raises(RuntimeError, match="reused an undated universe"):
        _run_jp_stale_universe(
            monkeypatch,
            tmp_path,
            seed_published_at=datetime.utcnow() - timedelta(days=2),
            seed_coverage={"stale_universe": True},
        )


def test_stale_universe_only_covers_source_outages(monkeypatch, tmp_path):
    # A parser or schema defect must fail the job, not hide behind last week.
    with pytest.raises(ValueError, match="multiple snapshot dates"):
        _run_jp_stale_universe(
            monkeypatch,
            tmp_path,
            seed_published_at=datetime.utcnow() - timedelta(days=2),
            universe_error=ValueError("JP official universe parse saw multiple snapshot dates"),
        )


def test_partial_publish_fallback_also_refuses_an_old_seed(monkeypatch, tmp_path):
    # Deliberate change in #521: CN/TW reuse the universe under the same age rule.
    with pytest.raises(RuntimeError, match="max age 8 day"):
        _run_jp_stale_universe(
            monkeypatch,
            tmp_path,
            seed_published_at=datetime.utcnow() - timedelta(days=20),
            flags=["--allow-partial-publish"],
            universe_error=RuntimeError("AKShare CN spot disconnected"),
        )


def test_universe_age_is_the_official_listing_date_not_the_publish_date(monkeypatch, tmp_path):
    # Published two days ago, but its listing file was twelve days old then.
    listing_date = (datetime.utcnow() - timedelta(days=12)).date().isoformat()
    with pytest.raises(RuntimeError, match="max age 8 day"):
        _run_jp_stale_universe(
            monkeypatch,
            tmp_path,
            seed_published_at=datetime.utcnow() - timedelta(days=2),
            seed_coverage={"universe_as_of_date": listing_date},
        )


def test_successful_universe_refresh_records_the_listing_date(monkeypatch, tmp_path):
    active_rows = [_make_universe_row("7203.T", market="JP")]
    fake_db = _make_cn_db_mock(active_rows)
    monkeypatch.setattr(build_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(build_script, "SessionLocal", lambda: _fake_session(fake_db))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    _patch_cn_dependencies(monkeypatch, raise_universe=False)
    monkeypatch.setattr(
        build_script, "ingest_official_market_snapshot", lambda db, service, snapshot: {"added": 0}
    )
    monkeypatch.setattr(
        build_script, "get_fundamentals_cache", lambda: SimpleNamespace(get_many=lambda symbols: {})
    )
    publish_calls: list[dict] = []
    monkeypatch.setattr(
        build_script,
        "get_provider_snapshot_service",
        lambda: _jp_snapshot_service(publish_calls, seed_published_at=datetime.utcnow()),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["build_weekly_reference_bundle", "--market", "JP", "--output-dir", str(tmp_path)],
    )

    assert build_script.main() == 0

    # _patch_cn_dependencies' snapshot is dated 2026-05-09.
    assert publish_calls[0]["coverage_stats"]["universe_as_of_date"] == "2026-05-09"


@pytest.mark.parametrize(
    ("source_name", "expected_calls"),
    [("jpx_official", 1), ("de_manual_csv", 0)],
)
def test_official_baseline_is_seeded_only_for_live_sources(source_name, expected_calls):
    calls = []
    service = SimpleNamespace(
        seed_reconciliation_baseline_from_active_rows=lambda db, **kwargs: calls.append(kwargs)
    )
    snapshot = SimpleNamespace(market="JP", source_name=source_name, source_metadata={})

    build_script._seed_official_reconciliation_baseline(object(), service, snapshot)

    assert len(calls) == expected_calls
    if calls:
        assert calls[0]["source_name"] == "jpx_official"
        assert calls[0]["row_source"] == "jp_ingest"


def test_official_baseline_is_not_seeded_from_an_nse_only_snapshot():
    calls = []
    service = SimpleNamespace(
        seed_reconciliation_baseline_from_active_rows=lambda db, **kwargs: calls.append(kwargs)
    )
    snapshot = SimpleNamespace(
        market="IN",
        source_name="in_reference_bundle",
        source_metadata={"bse_unavailable": "HTTPError: 403"},
    )

    build_script._seed_official_reconciliation_baseline(object(), service, snapshot)

    assert calls == []


def test_official_baseline_is_not_seeded_when_part_of_the_fetch_failed():
    calls = []
    service = SimpleNamespace(
        seed_reconciliation_baseline_from_active_rows=lambda db, **kwargs: calls.append(kwargs)
    )
    snapshot = SimpleNamespace(
        market="CA",
        source_name="tmx_official",
        source_metadata={"fetch_errors": {"tsx": {"Q": "ReadTimeout"}, "tsxv": {}}},
    )

    build_script._seed_official_reconciliation_baseline(object(), service, snapshot)

    assert calls == []


@pytest.mark.parametrize("key", ["validated_cn_baseline_breaches", "validated_krx_baseline_breaches"])
def test_official_baseline_is_not_seeded_when_board_counts_breach(key):
    calls = []
    service = SimpleNamespace(
        seed_reconciliation_baseline_from_active_rows=lambda db, **kwargs: calls.append(kwargs)
    )
    snapshot = SimpleNamespace(
        market="CN",
        source_name="cn_akshare_eastmoney",
        source_metadata={key: [{"exchange": "bse", "actual": 0}]},
    )

    build_script._seed_official_reconciliation_baseline(object(), service, snapshot)

    assert calls == []
