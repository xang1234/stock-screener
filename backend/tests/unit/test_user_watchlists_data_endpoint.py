from __future__ import annotations

from datetime import date, timedelta

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.wiring.bootstrap as bootstrap_module
from app.database import Base, get_db
from app.main import app
from app.models.industry import IBDIndustryGroup
from app.models.stock import StockPrice
from app.models.stock_universe import StockUniverse
from app.models.user_watchlist import UserWatchlist, WatchlistItem
from app.services import server_auth

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _disable_server_auth(monkeypatch):
    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(
        engine,
        tables=[
            StockUniverse.__table__,
            StockPrice.__table__,
            IBDIndustryGroup.__table__,
            UserWatchlist.__table__,
            WatchlistItem.__table__,
        ],
    )
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(
            engine,
            tables=[
                WatchlistItem.__table__,
                UserWatchlist.__table__,
                IBDIndustryGroup.__table__,
                StockPrice.__table__,
                StockUniverse.__table__,
            ],
        )


def _override_db(db):
    def _get_db():
        try:
            yield db
        finally:
            pass

    return _get_db


def _seed_price_history(session, symbol: str, start_close: float, end_close: float, *, days: int = 280) -> None:
    start_date = date.today() - timedelta(days=days - 1)
    price_step = (end_close - start_close) / float(max(days - 1, 1))
    for offset in range(days):
        close = round(start_close + (price_step * offset), 4)
        session.add(
            StockPrice(
                symbol=symbol,
                date=start_date + timedelta(days=offset),
                open=close * 0.99,
                high=close * 1.01,
                low=close * 0.98,
                close=close,
                adj_close=close,
                volume=1_000_000 + offset,
            )
        )


@pytest.mark.asyncio
async def test_watchlist_data_endpoint_uses_db_price_history_without_price_cache(client, session, monkeypatch):
    app.dependency_overrides[get_db] = _override_db(session)

    watchlist = UserWatchlist(name="Leaders", position=0)
    session.add_all(
        [
            watchlist,
            StockUniverse(symbol="AAPL", name="Apple Inc."),
            StockUniverse(symbol="SPY", name="SPDR S&P 500 ETF"),
            IBDIndustryGroup(symbol="AAPL", industry_group="Computer-Hardware/Peripherals"),
        ]
    )
    session.flush()
    session.add(WatchlistItem(watchlist_id=watchlist.id, symbol="AAPL", position=0))
    _seed_price_history(session, "AAPL", 150.0, 220.0)
    _seed_price_history(session, "SPY", 400.0, 520.0)
    session.commit()

    def _unexpected_price_cache_call():
        raise AssertionError("watchlist data endpoint should not hit the live price cache path")

    monkeypatch.setattr(bootstrap_module, "get_price_cache", _unexpected_price_cache_call)

    response = await client.get(f"/api/v1/user-watchlists/{watchlist.id}/data")

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "Leaders"
    assert len(payload["items"]) == 1
    item = payload["items"][0]
    assert item["symbol"] == "AAPL"
    assert item["company_name"] == "Apple Inc."
    assert item["ibd_industry"] == "Computer-Hardware/Peripherals"
    assert len(item["price_data"]) == 30
    assert len(item["rs_data"]) == 30
    assert item["change_12m"] is not None
