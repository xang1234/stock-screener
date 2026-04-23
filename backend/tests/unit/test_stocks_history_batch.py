from __future__ import annotations

from datetime import datetime, timedelta

import httpx
import pandas as pd
import pytest
import pytest_asyncio

from app.api.v1 import stocks as stocks_module
from app.main import app
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


def _make_ohlcv_frame(days: int = 30) -> pd.DataFrame:
    end = datetime.now()
    index = pd.date_range(end=end, periods=days, freq="D", name="Date")
    return pd.DataFrame(
        {
            "Open": [100.0 + i * 0.1 for i in range(days)],
            "High": [101.0 + i * 0.1 for i in range(days)],
            "Low": [99.0 + i * 0.1 for i in range(days)],
            "Close": [100.5 + i * 0.1 for i in range(days)],
            "Volume": [1_000_000 + i * 1000 for i in range(days)],
        },
        index=index,
    )


class _FakeCache:
    def __init__(self, frames):
        self._frames = frames
        self.last_period = None
        self.last_symbols = None

    def get_many(self, symbols, period="2y"):
        self.last_symbols = list(symbols)
        self.last_period = period
        return {sym: self._frames.get(sym) for sym in symbols}


@pytest.mark.asyncio
async def test_history_batch_returns_points_and_missing(client, monkeypatch):
    frames = {"AAPL": _make_ohlcv_frame(30), "MSFT": _make_ohlcv_frame(30)}
    fake = _FakeCache(frames)
    monkeypatch.setattr(stocks_module, "get_price_cache", lambda: fake)

    response = await client.post(
        "/api/v1/stocks/history/batch",
        json={"symbols": ["AAPL", "msft", "ZZZZ"], "period": "1mo"},
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body["data"].keys()) == {"AAPL", "MSFT"}
    assert body["missing"] == ["ZZZZ"]

    points = body["data"]["AAPL"]
    assert len(points) > 0
    first = points[0]
    assert set(first.keys()) == {"date", "open", "high", "low", "close", "volume"}
    assert isinstance(first["volume"], int)
    assert isinstance(first["close"], float)

    # Cache invoked with normalized (uppercase, deduped) symbols
    assert fake.last_symbols == ["AAPL", "MSFT", "ZZZZ"]
    # Cache period is "2y" for any non-5y request (mirrors single-symbol behavior)
    assert fake.last_period == "2y"


@pytest.mark.asyncio
async def test_history_batch_rejects_unsupported_period(client, monkeypatch):
    monkeypatch.setattr(stocks_module, "get_price_cache", lambda: _FakeCache({}))

    response = await client.post(
        "/api/v1/stocks/history/batch",
        json={"symbols": ["AAPL"], "period": "10y"},
    )

    assert response.status_code == 422
    assert "Unsupported period" in response.json()["detail"]


@pytest.mark.asyncio
async def test_history_batch_rejects_empty_symbol_list(client, monkeypatch):
    monkeypatch.setattr(stocks_module, "get_price_cache", lambda: _FakeCache({}))

    response = await client.post(
        "/api/v1/stocks/history/batch",
        json={"symbols": [], "period": "6mo"},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_history_batch_enforces_upper_bound(client, monkeypatch):
    monkeypatch.setattr(stocks_module, "get_price_cache", lambda: _FakeCache({}))

    response = await client.post(
        "/api/v1/stocks/history/batch",
        json={"symbols": [f"SYM{i}" for i in range(101)], "period": "6mo"},
    )

    assert response.status_code == 422
    assert "maximum" in response.json()["detail"]


@pytest.mark.asyncio
async def test_history_batch_handles_all_missing(client, monkeypatch):
    fake = _FakeCache(frames={})
    monkeypatch.setattr(stocks_module, "get_price_cache", lambda: fake)

    response = await client.post(
        "/api/v1/stocks/history/batch",
        json={"symbols": ["NOPE", "ALSO"], "period": "6mo"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["data"] == {}
    assert body["missing"] == ["NOPE", "ALSO"]
