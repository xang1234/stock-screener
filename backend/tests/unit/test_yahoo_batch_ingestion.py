from __future__ import annotations

from datetime import date, timedelta
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.stock import StockPrice
from app.models.stock_universe import (
    StockUniverse,
    UNIVERSE_STATUS_ACTIVE,
    UNIVERSE_STATUS_INACTIVE_NO_DATA,
)
from app.services.bulk_data_fetcher import BulkDataFetcher
from app.services.price_cache_service import PriceCacheService
from app.tasks.cache_tasks import cleanup_old_price_data


def _price_df(day: date, close: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [close - 1],
            "High": [close + 1],
            "Low": [close - 2],
            "Close": [close],
            "Adj Close": [close - 0.5],
            "Volume": [1_000_000],
        },
        index=pd.to_datetime([day]),
    )


def _success_result(symbol: str) -> dict:
    return {
        "symbol": symbol,
        "price_data": _price_df(date(2026, 3, 18), 100.0),
        "info": None,
        "fundamentals": None,
        "has_error": False,
        "error": None,
    }


def test_fetch_batch_prices_uses_required_yfinance_flags(monkeypatch):
    captured = {}

    def fake_download(**kwargs):
        captured.update(kwargs)
        return _price_df(date(2026, 3, 18), 100.0)

    import app.services.bulk_data_fetcher as module

    monkeypatch.setattr(module.yf, "download", fake_download)

    fetcher = BulkDataFetcher()
    results = fetcher.fetch_batch_prices(["AAPL"], period="1y")

    assert results["AAPL"]["has_error"] is False
    assert captured["threads"] is False
    assert captured["progress"] is False
    assert captured["group_by"] == "ticker"
    assert captured["auto_adjust"] is False
    assert captured["actions"] is True


def test_fetch_price_batch_with_retries_degrades_batch_size(monkeypatch):
    fetcher = BulkDataFetcher()
    calls = []
    sleeps = []
    symbols = [f"SYM{i}" for i in range(60)]

    def fake_fetch_batch_prices(batch_symbols, period="2y"):
        calls.append(list(batch_symbols))
        if len(calls) == 1:
            return {
                symbol: {
                    "symbol": symbol,
                    "price_data": None,
                    "info": None,
                    "fundamentals": None,
                    "has_error": True,
                    "error": "429 rate limited",
                }
                for symbol in batch_symbols
            }
        return {symbol: _success_result(symbol) for symbol in batch_symbols}

    monkeypatch.setattr(fetcher, "fetch_batch_prices", fake_fetch_batch_prices)
    monkeypatch.setattr("app.services.bulk_data_fetcher.time.sleep", lambda seconds: sleeps.append(seconds))

    results = fetcher._fetch_price_batch_with_retries(
        symbols,
        period="2y",
        initial_batch_size=100,
    )

    assert len(results) == len(symbols)
    assert calls[0] == symbols
    assert len(calls[1]) == 50
    assert len(calls[2]) == 10
    assert sleeps == [30]


def test_store_in_database_replaces_latest_day_row(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    import app.services.price_cache_service as module

    monkeypatch.setattr(module, "SessionLocal", TestingSessionLocal)

    service = PriceCacheService(redis_client=None)
    target_day = date(2026, 3, 18)

    db = TestingSessionLocal()
    db.add(
        StockPrice(
            symbol="AAPL",
            date=target_day,
            open=90.0,
            high=91.0,
            low=89.0,
            close=90.5,
            volume=100,
            adj_close=90.5,
        )
    )
    db.commit()
    db.close()

    service._store_in_database("AAPL", _price_df(target_day, 110.0))

    db = TestingSessionLocal()
    rows = db.query(StockPrice).filter(StockPrice.symbol == "AAPL").all()
    assert len(rows) == 1
    assert rows[0].close == 110.0
    assert rows[0].adj_close == 109.5
    db.close()


def test_cleanup_old_price_data_skips_inactive_symbols(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    import app.tasks.cache_tasks as module

    monkeypatch.setattr(module, "SessionLocal", TestingSessionLocal)

    db = TestingSessionLocal()
    cutoff_candidate = date.today() - timedelta(days=366)

    db.add_all(
        [
            StockUniverse(
                symbol="AAPL",
                exchange="NASDAQ",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                status_reason="active",
            ),
            StockUniverse(
                symbol="DEAD",
                exchange="NASDAQ",
                is_active=False,
                status=UNIVERSE_STATUS_INACTIVE_NO_DATA,
                status_reason="inactive",
            ),
            StockPrice(
                symbol="AAPL",
                date=cutoff_candidate,
                open=10.0,
                high=10.5,
                low=9.5,
                close=10.0,
                volume=100,
                adj_close=10.0,
            ),
            StockPrice(
                symbol="DEAD",
                date=cutoff_candidate,
                open=20.0,
                high=20.5,
                low=19.5,
                close=20.0,
                volume=100,
                adj_close=20.0,
            ),
        ]
    )
    db.commit()
    db.close()

    monkeypatch.setattr(cleanup_old_price_data, "update_state", lambda *args, **kwargs: None)
    result = cleanup_old_price_data.run(keep_years=1)

    assert result["deleted_rows"] == 1

    db = TestingSessionLocal()
    remaining_symbols = {
        row[0]
        for row in db.query(StockPrice.symbol).all()
    }
    assert remaining_symbols == {"DEAD"}
    db.close()
