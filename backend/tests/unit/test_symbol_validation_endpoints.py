"""End-to-end tests for symbol-format validation on stock + watchlist endpoints.

Verifies that suffixed non-US tickers (`0700.HK`, `6758.T`, `2330.TW`) are
accepted uniformly, and that malformed input produces a clean 422 (not a
500 from an upstream data-provider failure).
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
# Register ORM models referenced by create_all.
import app.models.user_watchlist  # noqa: F401
import app.models.stock_universe  # noqa: F401
import app.models.stock  # noqa: F401
import app.models.theme  # noqa: F401
import app.models.scan_result  # noqa: F401
import app.infra.db.models.feature_store  # noqa: F401

from app.api.v1 import stocks as stocks_router_module
from app.api.v1 import user_watchlists as user_watchlists_router_module
from app.database import get_db
from app.main import app as fastapi_app
from app.models.stock_universe import StockUniverse
from app.models.user_watchlist import UserWatchlist


@pytest.fixture
def _sqlite_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    sess = sessionmaker(bind=engine)()
    try:
        yield sess
    finally:
        sess.close()
        engine.dispose()


@pytest_asyncio.fixture
async def client(_sqlite_session):
    # Seed a watchlist + a handful of symbols spanning US + Asia markets.
    watchlist = UserWatchlist(name="Test")
    _sqlite_session.add(watchlist)
    _sqlite_session.add_all([
        StockUniverse(symbol="AAPL", name="Apple", market="US", is_active=True),
        StockUniverse(symbol="0700.HK", name="Tencent", market="HK", is_active=True),
        StockUniverse(symbol="6758.T", name="Sony", market="JP", is_active=True),
        StockUniverse(symbol="2330.TW", name="TSMC", market="TW", is_active=True),
    ])
    _sqlite_session.commit()
    watchlist_id = watchlist.id

    def _override_get_db():
        try:
            yield _sqlite_session
        finally:
            pass

    fastapi_app.dependency_overrides[get_db] = _override_get_db
    try:
        transport = httpx.ASGITransport(app=fastapi_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            c.watchlist_id = watchlist_id
            yield c
    finally:
        fastapi_app.dependency_overrides.pop(get_db, None)


# ---------------------------------------------------------------------------
# Stock detail endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStockSymbolPathValidation:
    async def test_info_rejects_malformed_symbol_with_422(self, client):
        resp = await client.get("/api/v1/stocks/NV DA/info")
        # URL-encoded space — FastAPI routes it through our dependency.
        assert resp.status_code == 422
        assert "Invalid symbol format" in resp.json()["detail"]

    async def test_info_rejects_non_ascii_symbol_with_422(self, client):
        resp = await client.get("/api/v1/stocks/日経/info")
        assert resp.status_code == 422

    async def test_info_rejects_oversized_symbol_with_422(self, client):
        oversized = "X" * 21
        resp = await client.get(f"/api/v1/stocks/{oversized}/info")
        assert resp.status_code == 422

    async def test_chart_data_rejects_malformed_shape_before_db_access(self, client):
        # The format-validation dependency must fire BEFORE the uow
        # dependency runs — otherwise malformed input reaches the DB
        # layer and we waste a connection on a 500-producing round trip.
        resp = await client.get("/api/v1/stocks/NV DA/chart-data")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Watchlist add + bulk endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWatchlistSymbolValidation:
    async def test_add_item_rejects_malformed_symbol_with_422(self, client):
        resp = await client.post(
            f"/api/v1/user-watchlists/{client.watchlist_id}/items",
            json={"symbol": "NV DA"},
        )
        assert resp.status_code == 422
        assert "Invalid symbol format" in resp.json()["detail"]

    async def test_add_item_rejects_unknown_symbol_with_400(self, client):
        # Format-valid but not in the active universe.
        resp = await client.post(
            f"/api/v1/user-watchlists/{client.watchlist_id}/items",
            json={"symbol": "UNKNOWN.HK"},
        )
        assert resp.status_code == 400
        assert "not in the active stock universe" in resp.json()["detail"]

    async def test_add_item_accepts_hk_suffixed_ticker(self, client):
        resp = await client.post(
            f"/api/v1/user-watchlists/{client.watchlist_id}/items",
            json={"symbol": "0700.HK"},
        )
        assert resp.status_code == 200
        assert resp.json()["symbol"] == "0700.HK"

    async def test_add_item_normalizes_lowercase_input(self, client):
        resp = await client.post(
            f"/api/v1/user-watchlists/{client.watchlist_id}/items",
            json={"symbol": "aapl"},
        )
        assert resp.status_code == 200
        assert resp.json()["symbol"] == "AAPL"

    async def test_bulk_add_drops_malformed_silently_and_adds_valid(self, client):
        resp = await client.post(
            f"/api/v1/user-watchlists/{client.watchlist_id}/items/bulk",
            json={"symbols": ["AAPL", "0700.HK", "NV DA", "UNKNOWN.XX"]},
        )
        assert resp.status_code == 200
        added = {item["symbol"] for item in resp.json()}
        assert added == {"AAPL", "0700.HK"}

    async def test_bulk_add_dedupes_case_variants(self, client):
        resp = await client.post(
            f"/api/v1/user-watchlists/{client.watchlist_id}/items/bulk",
            json={"symbols": ["aapl", "AAPL", "AaPl"]},
        )
        assert resp.status_code == 200
        added = [item["symbol"] for item in resp.json()]
        assert added == ["AAPL"]
