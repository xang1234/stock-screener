"""GET /v1/user-themes/{id}/data reads prices cache first and fetches only misses."""

from __future__ import annotations

from datetime import date, timedelta

import httpx
import pandas as pd
import pytest
import pytest_asyncio

from app.api.v1 import user_themes
from app.database import SessionLocal
from app.main import app
from app.models.stock import StockPrice
from app.models.stock_universe import UNIVERSE_STATUS_ACTIVE, StockUniverse
from app.models.user_theme import UserTheme, UserThemeStock, UserThemeSubgroup
from app.services import price_cache_service, redis_pool, server_auth
from app.services.bulk_data_fetcher import BulkDataFetcher
from app.services.price_cache_service import PriceCacheService


def _frame(end: date, days: int = 260) -> pd.DataFrame:
    index = pd.bdate_range(end=end, periods=days)
    close = pd.Series(range(100, 100 + days), index=index, dtype=float)
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close,
         "Adj Close": close, "Volume": 1_000_000},
        index=index,
    )


@pytest.fixture
def provider_calls(monkeypatch):
    """Record every symbol a price provider is asked for; serve a fresh frame."""
    calls: list[str] = []
    fresh = _frame(date.today())

    def fetch_prices_in_batches(self, symbols, **_kwargs):
        calls.extend(symbols)
        return {s: {"price_data": fresh, "has_error": False} for s in symbols}

    def fetch_direct(self, symbol, *, period):
        calls.append(symbol)
        return fresh

    monkeypatch.setattr(BulkDataFetcher, "fetch_prices_in_batches", fetch_prices_in_batches)
    monkeypatch.setattr(PriceCacheService, "_fetch_direct_historical_data", fetch_direct)
    return calls


@pytest.fixture
def theme_id(monkeypatch):
    # No Redis: the service reads the database tier.
    monkeypatch.setattr(redis_pool, "get_redis_client", lambda: None)
    monkeypatch.setattr(price_cache_service, "get_redis_client", lambda: None)
    cache = PriceCacheService(session_factory=SessionLocal)
    monkeypatch.setattr(user_themes, "get_price_cache", lambda: cache)

    db = SessionLocal()
    for symbol in ("AAPL", "NEWCO"):
        db.add(StockUniverse(
            symbol=symbol, market="US", exchange="XNAS", is_active=True,
            status=UNIVERSE_STATUS_ACTIVE, status_reason="active",
        ))
    # Cached but a month behind: still usable, so it must not be refetched.
    stale = date.today() - timedelta(days=30)
    for symbol in ("AAPL", "SPY"):
        for day, row in _frame(stale).iterrows():
            db.add(StockPrice(
                symbol=symbol, date=day.date(), open=row.Open, high=row.High,
                low=row.Low, close=row.Close, adj_close=row.Close, volume=1_000_000,
            ))
    theme = UserTheme(name="AI", position=0)
    db.add(theme)
    db.flush()
    subgroup = UserThemeSubgroup(theme_id=theme.id, name="Chips", position=0)
    db.add(subgroup)
    db.flush()
    db.add_all([
        UserThemeStock(subgroup_id=subgroup.id, symbol="AAPL", position=0),
        UserThemeStock(subgroup_id=subgroup.id, symbol="NEWCO", position=1),
    ])
    db.commit()
    theme_id = theme.id
    db.close()
    return theme_id


@pytest_asyncio.fixture
async def client(monkeypatch):
    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_theme_data_fetches_only_symbols_missing_from_cache(client, theme_id, provider_calls):
    response = await client.get(f"/api/v1/user-themes/{theme_id}/data")

    assert response.status_code == 200
    assert provider_calls == ["NEWCO"]
    stocks = {s["symbol"]: s for s in response.json()["subgroups"][0]["stocks"]}
    assert stocks["AAPL"]["price_data"] and stocks["NEWCO"]["price_data"]

    # The fetched miss was written to the cache, so the next request fetches nothing.
    provider_calls.clear()
    assert (await client.get(f"/api/v1/user-themes/{theme_id}/data")).status_code == 200
    assert provider_calls == []


@pytest.mark.asyncio
async def test_theme_data_serves_cached_symbols_when_the_miss_fetch_fails(
    client, theme_id, monkeypatch
):
    def provider_down(self, symbols, **_kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(BulkDataFetcher, "fetch_prices_in_batches", provider_down)
    response = await client.get(f"/api/v1/user-themes/{theme_id}/data")

    assert response.status_code == 200
    stocks = {s["symbol"]: s for s in response.json()["subgroups"][0]["stocks"]}
    assert stocks["AAPL"]["price_data"]
    assert not stocks["NEWCO"]["price_data"]
