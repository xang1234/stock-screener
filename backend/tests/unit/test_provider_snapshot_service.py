from __future__ import annotations

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
from app.models.stock_universe import StockUniverse, UNIVERSE_STATUS_ACTIVE
from app.services.fundamentals_cache_service import FundamentalsCacheService
from app.services.provider_snapshot_service import ProviderSnapshotService, settings


class _StubFundamentalsCache:
    def __init__(self):
        self.stored: dict[str, dict] = {}

    def get_many(self, symbols):
        return {symbol: {} for symbol in symbols}

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
        lambda exchange_filter=None: {
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
