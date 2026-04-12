from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.stock_universe import StockUniverse, UNIVERSE_STATUS_ACTIVE
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
