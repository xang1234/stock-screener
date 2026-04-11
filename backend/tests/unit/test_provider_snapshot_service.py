from __future__ import annotations

import gzip
import json
from datetime import date, datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.provider_snapshot import (
    ProviderSnapshotPointer,
    ProviderSnapshotRow,
    ProviderSnapshotRun,
)
from app.models.stock import StockFundamental
from app.models.stock_universe import StockUniverse, UNIVERSE_STATUS_ACTIVE
import app.services.fundamentals_cache_service as fundamentals_cache_module
from app.services.fundamentals_cache_service import FundamentalsCacheService
from app.services.provider_snapshot_service import ProviderSnapshotService, settings


class _StubFundamentalsCache:
    def __init__(self, cached: dict[str, dict] | None = None):
        self.stored: dict[str, dict] = {}
        self.cached = cached or {}

    def get_many(self, symbols):
        return {symbol: dict(self.cached.get(symbol) or {}) for symbol in symbols}

    @staticmethod
    def _merge_fundamentals(primary, fallback):
        merged = dict(primary)
        for key, value in fallback.items():
            if key not in merged or merged[key] is None:
                merged[key] = value
        return merged

    def store(self, symbol, data, data_source="snapshot"):
        payload = dict(data)
        payload["data_source"] = data_source
        self.stored[symbol] = payload


class _StubPriceCache:
    @staticmethod
    def get_many(symbols, period="2y"):
        return {symbol: None for symbol in symbols}

    @staticmethod
    def get_many_cached_only(symbols, period="2y"):
        return {symbol: None for symbol in symbols}

    @staticmethod
    def get_many_cached_only_fresh(symbols, period="2y"):
        return {symbol: None for symbol in symbols}


class _StubTechnicalCalc:
    @staticmethod
    def calculate_batch(price_data):
        return {}


def _make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return TestingSessionLocal


def test_create_snapshot_run_blocks_publish_when_coverage_below_threshold(monkeypatch):
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    db.add_all(
        [
            StockUniverse(
                symbol="AAPL",
                exchange="NASDAQ",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                status_reason="active",
            ),
            StockUniverse(
                symbol="MSFT",
                exchange="NASDAQ",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                status_reason="active",
            ),
        ]
    )
    db.commit()

    service = ProviderSnapshotService()
    monkeypatch.setattr(
        service,
        "_build_snapshot_rows",
        lambda exchange_filter=None, **kwargs: {
            "AAPL": {
                "exchange": "NASDAQ",
                "row_hash": "hash-aapl",
                "normalized_payload": {"symbol": "AAPL", "exchange": "NASDAQ"},
                "raw_payload": {"overview": {"Ticker": "AAPL"}},
            }
        },
    )
    monkeypatch.setattr(settings, "provider_snapshot_min_active_coverage", 0.98)
    monkeypatch.setattr(settings, "provider_snapshot_max_missing_active_symbols", 50)

    result = service.create_snapshot_run(db, run_mode="publish", publish=True)

    blocked_run = db.query(ProviderSnapshotRun).filter(ProviderSnapshotRun.id == result["run_id"]).one()
    assert result["published"] is False
    assert result["warnings"]
    assert blocked_run.status == "publish_blocked"
    assert db.query(ProviderSnapshotPointer).count() == 0
    db.close()


def test_hydrate_published_snapshot_fetches_yahoo_only_fields_for_missing_scan_data():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    db.add(
        StockUniverse(
            symbol="AAPL",
            exchange="NASDAQ",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
            status_reason="active",
        )
    )
    run = ProviderSnapshotRun(
        snapshot_key=ProviderSnapshotService.SNAPSHOT_KEY_FUNDAMENTALS,
        run_mode="publish",
        status="published",
        source_revision="fundamentals_v1:20260319010101",
        created_at=datetime.utcnow(),
        published_at=datetime.utcnow(),
    )
    db.add(run)
    db.flush()
    db.add(
        ProviderSnapshotRow(
            run_id=run.id,
            symbol="AAPL",
            exchange="NASDAQ",
            row_hash="row-hash",
            normalized_payload_json=json.dumps(
                {"symbol": "AAPL", "exchange": "NASDAQ", "market_cap": 1000, "sector": "Tech"}
            ),
            raw_payload_json=json.dumps({"overview": {"Ticker": "AAPL"}}),
        )
    )
    db.add(
        ProviderSnapshotPointer(
            snapshot_key=ProviderSnapshotService.SNAPSHOT_KEY_FUNDAMENTALS,
            run_id=run.id,
        )
    )
    db.commit()

    service = ProviderSnapshotService()
    service.fundamentals_cache = _StubFundamentalsCache()
    service.price_cache = _StubPriceCache()
    service.technical_calc = _StubTechnicalCalc()
    service._fetch_yahoo_only_fields = lambda symbol: {
        "ipo_date": date(2020, 1, 2),
        "first_trade_date": 1577923200,
        "eps_growth_qq": 12.3,
        "sales_growth_qq": 4.5,
        "eps_growth_yy": 22.0,
        "sales_growth_yy": 9.0,
        "recent_quarter_date": "2025-12-31",
        "previous_quarter_date": "2025-09-30",
        "eps_5yr_cagr": 18.1,
        "eps_q1_yoy": 15.0,
        "eps_q2_yoy": 13.0,
        "eps_raw_score": 78.0,
        "eps_years_available": 5,
        "yahoo_profile_refreshed_at": "2026-03-19T00:00:00",
        "yahoo_statements_refreshed_at": "2026-03-19T00:00:00",
    }

    stats = service.hydrate_published_snapshot(db)

    stored = service.fundamentals_cache.stored["AAPL"]
    assert stats["yahoo_hydrated"] == 1
    assert stats["missing_yahoo"] == 0
    assert stored["ipo_date"] == date(2020, 1, 2)
    assert stored["first_trade_date"] == 1577923200
    assert stored["eps_growth_qq"] == 12.3
    assert stored["data_source"] == "snapshot"
    db.close()


def test_hydrate_published_snapshot_can_skip_yahoo_when_disabled():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    db.add(
        StockUniverse(
            symbol="AAPL",
            exchange="NASDAQ",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
            status_reason="active",
        )
    )
    run = ProviderSnapshotRun(
        snapshot_key=ProviderSnapshotService.SNAPSHOT_KEY_FUNDAMENTALS,
        run_mode="publish",
        status="published",
        source_revision="fundamentals_v1:20260319010101",
        created_at=datetime.utcnow(),
        published_at=datetime.utcnow(),
    )
    db.add(run)
    db.flush()
    db.add(
        ProviderSnapshotRow(
            run_id=run.id,
            symbol="AAPL",
            exchange="NASDAQ",
            row_hash="row-hash",
            normalized_payload_json=json.dumps({"symbol": "AAPL", "exchange": "NASDAQ", "market_cap": 1000}),
            raw_payload_json=None,
        )
    )
    db.add(
        ProviderSnapshotPointer(
            snapshot_key=ProviderSnapshotService.SNAPSHOT_KEY_FUNDAMENTALS,
            run_id=run.id,
        )
    )
    db.commit()

    service = ProviderSnapshotService()
    service.fundamentals_cache = _StubFundamentalsCache()
    service.price_cache = _StubPriceCache()
    service.technical_calc = _StubTechnicalCalc()
    yahoo_calls: list[str] = []
    service._fetch_yahoo_only_fields = lambda symbol: yahoo_calls.append(symbol) or {}

    stats = service.hydrate_published_snapshot(db, allow_yahoo_hydration=False)

    assert stats["yahoo_hydrated"] == 0
    assert stats["missing_yahoo"] == 1
    assert yahoo_calls == []
    db.close()


def test_hydrate_published_snapshot_skips_unsupported_yahoo_symbols():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    db.add_all(
        [
            StockUniverse(
                symbol="AAPL",
                exchange="NASDAQ",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                status_reason="active",
            ),
            StockUniverse(
                symbol="BIII-U",
                exchange="NASDAQ",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                status_reason="active",
            ),
        ]
    )
    run = ProviderSnapshotRun(
        snapshot_key=ProviderSnapshotService.SNAPSHOT_KEY_FUNDAMENTALS,
        run_mode="publish",
        status="published",
        source_revision="fundamentals_v1:20260404161000",
        created_at=datetime.utcnow(),
        published_at=datetime.utcnow(),
    )
    db.add(run)
    db.flush()
    for symbol in ("AAPL", "BIII-U"):
        db.add(
            ProviderSnapshotRow(
                run_id=run.id,
                symbol=symbol,
                exchange="NASDAQ",
                row_hash=f"row-hash-{symbol}",
                normalized_payload_json=json.dumps({"symbol": symbol, "exchange": "NASDAQ", "market_cap": 1000}),
                raw_payload_json=None,
            )
        )
    db.add(
        ProviderSnapshotPointer(
            snapshot_key=ProviderSnapshotService.SNAPSHOT_KEY_FUNDAMENTALS,
            run_id=run.id,
        )
    )
    db.commit()

    class _RecordingPriceCache:
        def __init__(self):
            self.live_calls: list[tuple[list[str], str]] = []
            self.cached_only_fresh_calls: list[tuple[list[str], str]] = []

        def get_many(self, symbols, period="2y"):
            self.live_calls.append((list(symbols), period))
            return {symbol: None for symbol in symbols}

        def get_many_cached_only_fresh(self, symbols, period="2y"):
            self.cached_only_fresh_calls.append((list(symbols), period))
            return {symbol: None for symbol in symbols}

    service = ProviderSnapshotService()
    service.fundamentals_cache = _StubFundamentalsCache()
    service.price_cache = _RecordingPriceCache()
    service.technical_calc = _StubTechnicalCalc()
    yahoo_calls: list[str] = []
    service._fetch_yahoo_only_fields = lambda symbol: yahoo_calls.append(symbol) or {}

    stats = service.hydrate_published_snapshot(db, allow_yahoo_hydration=True)

    assert service.price_cache.live_calls == [(["AAPL"], "2y")]
    assert service.price_cache.cached_only_fresh_calls == [(["BIII-U"], "2y")]
    assert yahoo_calls == ["AAPL", "BIII-U"]
    assert stats["missing_prices"] == 2
    assert stats["skipped_yahoo_price_symbols"] == 1
    assert stats["skipped_yahoo_field_symbols"] == 0
    db.close()


def test_hydrate_published_snapshot_emits_progress_events():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    db.add(
        StockUniverse(
            symbol="AAPL",
            exchange="NASDAQ",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
            status_reason="active",
        )
    )
    run = ProviderSnapshotRun(
        snapshot_key=ProviderSnapshotService.SNAPSHOT_KEY_FUNDAMENTALS,
        run_mode="publish",
        status="published",
        source_revision="fundamentals_v1:20260404161000",
        created_at=datetime.utcnow(),
        published_at=datetime.utcnow(),
    )
    db.add(run)
    db.flush()
    db.add(
        ProviderSnapshotRow(
            run_id=run.id,
            symbol="AAPL",
            exchange="NASDAQ",
            row_hash="row-hash",
            normalized_payload_json=json.dumps({"symbol": "AAPL", "exchange": "NASDAQ", "market_cap": 1000}),
            raw_payload_json=None,
        )
    )
    db.add(
        ProviderSnapshotPointer(
            snapshot_key=ProviderSnapshotService.SNAPSHOT_KEY_FUNDAMENTALS,
            run_id=run.id,
        )
    )
    db.commit()

    service = ProviderSnapshotService()
    service.fundamentals_cache = _StubFundamentalsCache()
    service.price_cache = _StubPriceCache()
    service.technical_calc = _StubTechnicalCalc()
    progress_events: list[dict[str, object]] = []

    service.hydrate_published_snapshot(
        db,
        allow_yahoo_hydration=False,
        progress_callback=progress_events.append,
    )

    assert progress_events[0]["stage"] == "hydrate_start"
    assert progress_events[0]["total_symbols"] == 1
    assert progress_events[1]["stage"] == "hydrate_chunk_complete"
    assert progress_events[1]["processed_symbols"] == 1
    assert progress_events[1]["percent_complete"] == 100.0
    db.close()


def test_snapshot_active_coverage_ignores_status_active_rows_marked_inactive(monkeypatch):
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    db.add_all(
        [
            StockUniverse(
                symbol="AAPL",
                exchange="NASDAQ",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                status_reason="active",
            ),
            StockUniverse(
                symbol="OLD",
                exchange="NYSE",
                is_active=False,
                status=UNIVERSE_STATUS_ACTIVE,
                status_reason=None,
            ),
        ]
    )
    db.commit()

    service = ProviderSnapshotService()
    monkeypatch.setattr(
        service,
        "_build_snapshot_rows",
        lambda exchange_filter=None, **kwargs: {
            "AAPL": {
                "exchange": "NASDAQ",
                "row_hash": "hash-aapl",
                "normalized_payload": {"symbol": "AAPL", "exchange": "NASDAQ"},
                "raw_payload": {"overview": {"Ticker": "AAPL"}},
            }
        },
    )
    result = service.create_snapshot_run(db, run_mode="preview", publish=False)

    assert result["coverage"]["active_symbols"] == 1
    assert result["coverage"]["covered_active_symbols"] == 1
    assert result["coverage"]["missing_active_symbols"] == 0
    db.close()


def test_weekly_reference_bundle_round_trips_active_universe_and_enriched_snapshot(tmp_path):
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    db.add_all(
        [
            StockUniverse(
                symbol="AAPL",
                market="US",
                exchange="NASDAQ",
                currency="USD",
                timezone="America/New_York",
                local_code="AAPL",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                status_reason="active",
                sector="Technology",
                industry="Software",
                market_cap=123.0,
            ),
            StockUniverse(
                symbol="0700.HK",
                market="HK",
                exchange="HKEX",
                currency="HKD",
                timezone="Asia/Hong_Kong",
                local_code="0700",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                status_reason="active",
                sector="Technology",
                industry="Internet Content & Information",
                market_cap=456.0,
            ),
            StockUniverse(
                symbol="OLD",
                exchange="NYSE",
                is_active=False,
                status=UNIVERSE_STATUS_ACTIVE,
                status_reason=None,
            ),
        ]
    )
    run = ProviderSnapshotRun(
        snapshot_key=ProviderSnapshotService.SNAPSHOT_KEY_FUNDAMENTALS,
        run_mode="publish",
        status="published",
        source_revision="fundamentals_v1:20260402081000",
        coverage_stats_json=json.dumps({"active_symbols": 1, "covered_active_symbols": 1, "missing_active_symbols": 0}),
        parity_stats_json=json.dumps({"missing_active_symbols": []}),
        warnings_json=json.dumps([]),
        symbols_total=1,
        symbols_published=1,
        created_at=datetime.utcnow(),
        published_at=datetime.utcnow(),
    )
    db.add(run)
    db.flush()
    db.add(
        ProviderSnapshotRow(
            run_id=run.id,
            symbol="AAPL",
            exchange="NASDAQ",
            row_hash="row-hash",
            normalized_payload_json=json.dumps({"symbol": "AAPL", "exchange": "NASDAQ", "market_cap": 1000}),
            raw_payload_json=json.dumps({"overview": {"Ticker": "AAPL"}}),
        )
    )
    db.add(
        ProviderSnapshotPointer(
            snapshot_key=ProviderSnapshotService.SNAPSHOT_KEY_FUNDAMENTALS,
            run_id=run.id,
        )
    )
    db.commit()

    service = ProviderSnapshotService()
    service.fundamentals_cache = _StubFundamentalsCache(
        cached={
            "AAPL": {
                "ipo_date": date(2020, 1, 2),
                "first_trade_date": 1577923200,
                "eps_growth_qq": 12.3,
                "sales_growth_qq": 4.5,
                "eps_growth_yy": 22.0,
                "sales_growth_yy": 9.0,
                "description_yfinance": "Long summary",
            }
        }
    )

    bundle_path = tmp_path / "weekly-reference-20260402.json.gz"
    latest_manifest_path = tmp_path / "weekly-reference-latest.json"
    export_stats = service.export_weekly_reference_bundle(
        db,
        output_path=bundle_path,
        bundle_asset_name=bundle_path.name,
        latest_manifest_path=latest_manifest_path,
    )

    with gzip.open(bundle_path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)

    assert export_stats["rows"] == 1
    assert export_stats["universe_rows"] == 2
    assert payload["schema_version"] == ProviderSnapshotService.WEEKLY_REFERENCE_BUNDLE_SCHEMA_VERSION
    assert len(payload["snapshot"]["rows"]) == 1
    assert payload["snapshot"]["rows"][0]["normalized_payload"]["ipo_date"] == "2020-01-02"
    assert payload["snapshot"]["rows"][0]["normalized_payload"]["description_yfinance"] == "Long summary"
    assert len(payload["universe"]) == 2
    hk_row = next(row for row in payload["universe"] if row["symbol"] == "0700.HK")
    assert hk_row["market"] == "HK"
    assert hk_row["currency"] == "HKD"
    assert hk_row["timezone"] == "Asia/Hong_Kong"
    assert hk_row["local_code"] == "0700"

    # Simulate backward-compatible bundle with market present but currency/timezone missing.
    hk_row.pop("currency", None)
    hk_row.pop("timezone", None)
    with gzip.open(bundle_path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, default=str)

    manifest = json.loads(latest_manifest_path.read_text(encoding="utf-8"))
    assert manifest["bundle_asset_name"] == bundle_path.name
    assert manifest["sha256"]

    unrelated_run = ProviderSnapshotRun(
        snapshot_key="other_snapshot",
        run_mode="publish",
        status="published",
        source_revision="other-snapshot:20260401000000",
    )
    db.add(unrelated_run)
    db.flush()
    db.add(
        ProviderSnapshotRow(
            run_id=unrelated_run.id,
            symbol="QQQ",
            exchange="NASDAQ",
            row_hash="other-row-hash",
            normalized_payload_json=json.dumps({"symbol": "QQQ"}),
            raw_payload_json=None,
        )
    )
    db.add(
        ProviderSnapshotPointer(
            snapshot_key="other_snapshot",
            run_id=unrelated_run.id,
        )
    )
    db.commit()

    import_stats = service.import_weekly_reference_bundle(db, input_path=bundle_path)

    imported_run = service.get_published_run(db)
    imported_row = db.query(ProviderSnapshotRow).filter(ProviderSnapshotRow.run_id == imported_run.id).one()
    imported_payload = json.loads(imported_row.normalized_payload_json)
    imported_universe_rows = db.query(StockUniverse).order_by(StockUniverse.symbol.asc()).all()
    imported_symbols = [row.symbol for row in imported_universe_rows]
    imported_hk_row = next(row for row in imported_universe_rows if row.symbol == "0700.HK")

    assert import_stats["rows"] == 1
    assert import_stats["universe_rows"] == 2
    assert imported_run.source_revision == "fundamentals_v1:20260402081000"
    assert imported_payload["ipo_date"] == "2020-01-02"
    assert imported_symbols == ["0700.HK", "AAPL"]
    assert imported_hk_row.market == "HK"
    # Missing bundle values should hydrate from market-aware defaults.
    assert imported_hk_row.currency == "HKD"
    assert imported_hk_row.timezone == "Asia/Hong_Kong"
    assert imported_hk_row.local_code == "0700"
    assert db.query(ProviderSnapshotRun).filter(
        ProviderSnapshotRun.snapshot_key == ProviderSnapshotService.SNAPSHOT_KEY_FUNDAMENTALS
    ).count() == 1
    assert db.query(ProviderSnapshotRun).filter(
        ProviderSnapshotRun.snapshot_key == "other_snapshot"
    ).count() == 1
    assert db.query(ProviderSnapshotPointer).count() == 2
    db.close()


def test_imported_weekly_reference_bundle_hydrates_ipo_date_back_to_database(tmp_path, monkeypatch):
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    db.add(
        StockUniverse(
            symbol="AAPL",
            exchange="NASDAQ",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
            status_reason="active",
        )
    )
    run = ProviderSnapshotRun(
        snapshot_key=ProviderSnapshotService.SNAPSHOT_KEY_FUNDAMENTALS,
        run_mode="publish",
        status="published",
        source_revision="fundamentals_v1:20260404161000",
        coverage_stats_json=json.dumps({"active_symbols": 1, "covered_active_symbols": 1, "missing_active_symbols": 0}),
        parity_stats_json=json.dumps({"missing_active_symbols": []}),
        warnings_json=json.dumps([]),
        symbols_total=1,
        symbols_published=1,
        created_at=datetime.utcnow(),
        published_at=datetime.utcnow(),
    )
    db.add(run)
    db.flush()
    db.add(
        ProviderSnapshotRow(
            run_id=run.id,
            symbol="AAPL",
            exchange="NASDAQ",
            row_hash="row-hash",
            normalized_payload_json=json.dumps({"symbol": "AAPL", "exchange": "NASDAQ", "market_cap": 1000}),
            raw_payload_json=json.dumps({"overview": {"Ticker": "AAPL"}}),
        )
    )
    db.add(
        ProviderSnapshotPointer(
            snapshot_key=ProviderSnapshotService.SNAPSHOT_KEY_FUNDAMENTALS,
            run_id=run.id,
        )
    )
    db.commit()

    service = ProviderSnapshotService()
    service.fundamentals_cache = _StubFundamentalsCache(
        cached={
            "AAPL": {
                "ipo_date": date(2020, 1, 2),
                "first_trade_date": 1577923200,
                "eps_growth_qq": 12.3,
                "sales_growth_qq": 4.5,
                "eps_growth_yy": 22.0,
                "sales_growth_yy": 9.0,
                "eps_rating": 95,
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000,
                "avg_volume": 500000,
            }
        }
    )

    bundle_path = tmp_path / "weekly-reference-20260404.json.gz"
    service.export_weekly_reference_bundle(
        db,
        output_path=bundle_path,
        bundle_asset_name=bundle_path.name,
    )
    service.import_weekly_reference_bundle(db, input_path=bundle_path)

    monkeypatch.setattr(fundamentals_cache_module, "SessionLocal", TestingSessionLocal)
    monkeypatch.setattr(fundamentals_cache_module, "get_redis_client", lambda: None)

    service.fundamentals_cache = FundamentalsCacheService()
    service.price_cache = _StubPriceCache()
    service.technical_calc = _StubTechnicalCalc()

    stats = service.hydrate_published_snapshot(db, allow_yahoo_hydration=False)

    stored = db.query(StockFundamental).filter(StockFundamental.symbol == "AAPL").one()
    assert stats["hydrated"] == 1
    assert stored.ipo_date == date(2020, 1, 2)
    db.close()


def test_get_fundamentals_fetches_on_demand_when_fresh_cache_is_missing_required_fields(monkeypatch):
    service = FundamentalsCacheService(redis_client=None)
    incomplete = {
        "sector": "Technology",
        "industry": "Software",
        "eps_growth_qq": 10.0,
        "sales_growth_qq": 8.0,
        "eps_growth_yy": 12.0,
        "sales_growth_yy": 9.0,
        "market_cap": 1000,
        "avg_volume": 500000,
        "eps_rating": 92,
        "ipo_date": None,
        "first_trade_date": None,
    }
    enriched = dict(incomplete)
    enriched.update({"ipo_date": date(2020, 1, 2), "first_trade_date": 1577923200})

    monkeypatch.setattr(service, "_get_from_database", lambda symbol: (dict(incomplete), datetime.utcnow()))
    fetch_calls = []

    def fake_fetch(symbol):
        fetch_calls.append(symbol)
        return dict(enriched)

    monkeypatch.setattr(service, "_fetch_and_cache", fake_fetch)

    result = service.get_fundamentals("AAPL")

    assert result == enriched
    assert fetch_calls == ["AAPL"]
