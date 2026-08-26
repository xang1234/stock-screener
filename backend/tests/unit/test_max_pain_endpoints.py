from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.models.max_pain import MaxPainSnapshot
from app.services import server_auth


@pytest_asyncio.fixture
async def client(monkeypatch):
    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(engine, tables=[MaxPainSnapshot.__table__])
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(engine, tables=[MaxPainSnapshot.__table__])
        app.dependency_overrides.clear()


def _override_db(session):
    def _yield_db():
        try:
            yield session
        finally:
            pass

    return _yield_db


@pytest.mark.asyncio
async def test_max_pain_symbol_query_returns_404_when_missing(client, session):
    app.dependency_overrides[get_db] = _override_db(session)

    response = await client.get("/api/v1/max-pain/dashboard", params={"symbol": "AAPL"})

    assert response.status_code == 404
    payload = response.json()
    assert payload["detail"] == "No max pain data available for symbol AAPL"


@pytest.mark.asyncio
async def test_max_pain_dashboard_returns_mock_data_when_no_rows(client, session):
    app.dependency_overrides[get_db] = _override_db(session)

    response = await client.get("/api/v1/max-pain/dashboard")

    assert response.status_code == 200
    payload = response.json()
    assert payload["rows"]
    assert any(row["symbol"] == "AAPL" for row in payload["rows"])
    assert payload["tickers_ok"] == 100
