"""SqlUniverseRepository.resolve_markets against in-memory SQLite."""

from __future__ import annotations

import app.models.stock_universe  # noqa: F401 — register the table
from app.infra.db.repositories.universe_repo import SqlUniverseRepository
from app.models.stock_universe import StockUniverse


def test_resolve_markets_returns_authoritative_market_per_known_symbol(session):
    session.add_all([
        StockUniverse(symbol="AAPL", market="US", exchange="NASDAQ"),
        StockUniverse(symbol="0700.HK", market="HK", exchange="XHKG", currency="HKD"),
    ])
    session.flush()

    markets = SqlUniverseRepository(session).resolve_markets(["aapl", "0700.HK", "GONE"])

    assert markets == {"AAPL": "US", "0700.HK": "HK"}
