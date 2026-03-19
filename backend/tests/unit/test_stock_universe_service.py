from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.stock_universe import StockUniverse, UNIVERSE_STATUS_ACTIVE
from app.services.stock_universe_service import stock_universe_service


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
