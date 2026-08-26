from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.database import Base, get_db
from app.main import app as fastapi_app
from app.models.stock_universe import StockUniverse


@pytest.fixture
def _sqlite_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sess = sessionmaker(bind=engine)()
    try:
        yield sess
    finally:
        sess.close()
        engine.dispose()


@pytest_asyncio.fixture
async def client(_sqlite_session, monkeypatch):
    monkeypatch.setattr(settings, "server_auth_enabled", False)

    def _override_get_db():
        try:
            yield _sqlite_session
        finally:
            pass

    fastapi_app.dependency_overrides[get_db] = _override_get_db
    try:
        transport = httpx.ASGITransport(app=fastapi_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
    finally:
        fastapi_app.dependency_overrides.pop(get_db, None)


@pytest.mark.asyncio
async def test_universe_symbols_endpoint_returns_active_symbols(client, _sqlite_session):
    _sqlite_session.add_all([
        StockUniverse(symbol="AAPL", name="Apple", market="US", exchange="NASDAQ", is_active=True),
        StockUniverse(symbol="MSFT", name="Microsoft", market="US", exchange="NASDAQ", is_active=True),
        StockUniverse(symbol="TSLA", name="Tesla", market="US", exchange="NASDAQ", is_active=False),
    ])
    _sqlite_session.commit()

    response = await client.get("/api/v1/universe/symbols", params={"q": "aap", "limit": 10})

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbols"][0]["symbol"] == "AAPL"
    assert payload["symbols"][0]["name"] == "Apple"
    assert payload["symbols"][0]["exchange"] == "NASDAQ"
