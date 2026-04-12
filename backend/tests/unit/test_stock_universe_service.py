from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.stock_universe import (
    StockUniverse,
    StockUniverseStatusEvent,
    UNIVERSE_STATUS_ACTIVE,
    UNIVERSE_STATUS_INACTIVE_MANUAL,
)
from app.services.stock_universe_service import StockUniverseService

stock_universe_service = StockUniverseService()


def _make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return TestingSessionLocal


def test_get_active_symbols_uses_is_active_over_stale_active_status():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    db.add_all(
        [
            StockUniverse(
                symbol="AAPL",
                exchange="NASDAQ",
                market_cap=1000,
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                status_reason="Present in Finviz universe sync",
            ),
            StockUniverse(
                symbol="OLD",
                exchange="NYSE",
                market_cap=10,
                is_active=False,
                status=UNIVERSE_STATUS_ACTIVE,
                status_reason=None,
            ),
        ]
    )
    db.commit()

    symbols = stock_universe_service.get_active_symbols(db)

    assert symbols == ["AAPL"]
    db.close()


def test_normalize_status_treats_active_status_plus_inactive_flag_as_inactive():
    record = StockUniverse(
        symbol="OLD",
        is_active=False,
        status=UNIVERSE_STATUS_ACTIVE,
        status_reason=None,
    )

    normalized = stock_universe_service._normalize_status(record)

    assert normalized != UNIVERSE_STATUS_ACTIVE


def test_get_active_symbols_market_filter_falls_back_to_exchange_when_market_blank():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    db.add_all(
        [
            StockUniverse(
                symbol="AAPL",
                exchange="NASDAQ",
                market="US",
                market_cap=1000,
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                status_reason="Present in Finviz universe sync",
            ),
            StockUniverse(
                symbol="IBM",
                exchange="NYSE",
                market="",
                market_cap=500,
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                status_reason="Present in Finviz universe sync",
            ),
            StockUniverse(
                symbol="2330.TW",
                exchange="TWSE",
                market="",
                market_cap=1200,
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                status_reason="Present in source universe sync",
            ),
        ]
    )
    db.commit()

    symbols = stock_universe_service.get_active_symbols(db, market="US")

    assert symbols == ["AAPL", "IBM"]
    db.close()


def test_populate_from_csv_sets_market_identity_fields_from_resolver():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()

    csv_content = "\n".join(
        [
            "symbol,name,exchange,sector,industry,market_cap",
            "0700.HK,Tencent,SEHK,Technology,Internet,500B",
        ]
    )
    stats = stock_universe_service.populate_from_csv(db, csv_content)

    row = db.query(StockUniverse).filter(StockUniverse.symbol == "0700.HK").one()
    assert stats["added"] == 1
    assert row.market == "HK"
    assert row.currency == "HKD"
    assert row.timezone == "Asia/Hong_Kong"
    assert row.local_code == "0700"
    db.close()


def test_populate_from_csv_persists_canonical_symbol_from_security_master():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()

    csv_content = "\n".join(
        [
            "symbol,name,exchange,sector,industry,market_cap",
            "700,Tencent,SEHK,Technology,Internet,500B",
        ]
    )
    stats = stock_universe_service.populate_from_csv(db, csv_content)

    row = db.query(StockUniverse).filter(StockUniverse.symbol == "700.HK").one()
    assert stats["added"] == 1
    assert row.exchange == "SEHK"
    assert row.market == "HK"
    assert row.local_code == "700"
    db.close()


def test_populate_from_csv_uses_tpex_two_suffix_for_unsuffixed_symbols():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()

    csv_content = "\n".join(
        [
            "symbol,name,exchange,sector,industry,market_cap",
            "3008,Largan,TPEX,Technology,Electronics,120B",
        ]
    )
    stock_universe_service.populate_from_csv(db, csv_content)

    row = db.query(StockUniverse).filter(StockUniverse.symbol == "3008.TWO").one()
    assert row.exchange == "TPEX"
    assert row.market == "TW"
    assert row.timezone == "Asia/Taipei"
    db.close()


def test_ingest_hk_from_csv_normalizes_variants_with_zero_padding_and_lineage():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()

    csv_content = "\n".join(
        [
            "symbol,name,exchange,sector,industry,market_cap",
            "700,Tencent,SEHK,Technology,Internet,500B",
            "0700.HK,Tencent Holdings,HKEX,Technology,Internet,500B",
            "00700,Tencent Holdings,XHKG,Technology,Internet,500B",
        ]
    )

    stats = stock_universe_service.ingest_hk_from_csv(
        db,
        csv_content,
        source_name="hkex_official",
        snapshot_id="hk-20260412",
    )

    row = db.query(StockUniverse).filter(StockUniverse.symbol == "0700.HK").one()
    events = (
        db.query(StockUniverseStatusEvent)
        .filter(StockUniverseStatusEvent.symbol == "0700.HK")
        .all()
    )

    assert stats["added"] == 1
    assert stats["updated"] == 0
    assert stats["total"] == 1
    assert stats["rejected"] == 0
    assert row.local_code == "0700"
    assert row.exchange == "XHKG"
    assert row.market == "HK"
    assert row.source == "hk_ingest"
    assert len(events) == 1
    payload = json.loads(events[0].payload_json)
    assert payload["snapshot_id"] == "hk-20260412"
    assert payload["source_name"] == "hkex_official"
    assert len(payload["lineage_hash"]) == 64
    assert len(payload["row_hash"]) == 64
    db.close()


def test_ingest_hk_from_csv_reactivates_existing_inactive_symbol():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    db.add(
        StockUniverse(
            symbol="0700.HK",
            exchange="SEHK",
            market="HK",
            is_active=False,
            status=UNIVERSE_STATUS_INACTIVE_MANUAL,
            status_reason="manual off",
        )
    )
    db.commit()

    csv_content = "\n".join(
        [
            "symbol,name,exchange,sector,industry,market_cap",
            "700,Tencent,SEHK,Technology,Internet,500B",
        ]
    )
    stats = stock_universe_service.ingest_hk_from_csv(
        db,
        csv_content,
        source_name="sehk_official",
        snapshot_id="hk-20260412",
    )
    row = db.query(StockUniverse).filter(StockUniverse.symbol == "0700.HK").one()

    assert stats["added"] == 0
    assert stats["updated"] == 1
    assert row.is_active is True
    assert row.status == UNIVERSE_STATUS_ACTIVE
    assert row.exchange == "XHKG"
    assert row.local_code == "0700"
    db.close()


def test_ingest_hk_from_csv_rejects_unapproved_source():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    csv_content = "symbol,name,exchange\n700,Tencent,SEHK\n"

    with pytest.raises(ValueError, match="Unapproved HK source"):
        stock_universe_service.ingest_hk_from_csv(
            db,
            csv_content,
            source_name="random_vendor",
            snapshot_id="hk-20260412",
        )
    db.close()


def test_ingest_hk_from_csv_reports_rejected_rows_in_non_strict_mode():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    csv_content = "\n".join(
        [
            "symbol,name,exchange",
            "ABC.HK,Invalid,SEHK",
            "700,Tencent,SEHK",
        ]
    )

    stats = stock_universe_service.ingest_hk_from_csv(
        db,
        csv_content,
        source_name="sehk_official",
        snapshot_id="hk-20260412",
        strict=False,
    )

    assert stats["added"] == 1
    assert stats["total"] == 1
    assert stats["rejected"] == 1
    assert stats["rejected_rows"][0]["source_symbol"] == "ABC.HK"
    db.close()
