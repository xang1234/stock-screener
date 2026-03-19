from __future__ import annotations

import json
import pickle
from datetime import date, datetime, timedelta
import sqlite3
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
from app.tasks.cache_tasks import (
    _force_refresh_stale_intraday_impl,
    _track_symbol_failures,
    cleanup_old_price_data,
)


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


class _FakePipeline:
    def __init__(self, results):
        self._results = results

    def get(self, key):
        return self

    def execute(self):
        return list(self._results)


class _FakeRedis:
    def __init__(self, results):
        self._results = results

    def pipeline(self):
        return _FakePipeline(self._results)


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


def test_get_many_reloads_after_close_if_redis_meta_marks_intraday_stale(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    import app.services.price_cache_service as module

    db = TestingSessionLocal()
    db.add(
        StockUniverse(
            symbol="AAPL",
            exchange="NASDAQ",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
            status_reason="active",
        )
    )
    db.commit()
    db.close()

    stale_df = _price_df(date(2026, 3, 18), 100.0)
    fake_redis = _FakeRedis(
        [
            pickle.dumps(stale_df),
            json.dumps({"needs_refresh_after_close": True}),
        ]
    )

    service = PriceCacheService(redis_client=fake_redis)

    monkeypatch.setattr(module, "SessionLocal", TestingSessionLocal)
    monkeypatch.setattr(module, "get_bulk_redis_client", lambda: None)
    monkeypatch.setattr(module, "get_eastern_now", lambda: datetime(2026, 3, 18, 17, 0, 0))
    monkeypatch.setattr(module, "is_market_open", lambda now=None: False)
    monkeypatch.setattr(service, "_get_expected_data_date", lambda: date(2026, 3, 18))
    monkeypatch.setattr(service, "_get_many_from_database", lambda symbols, period: {"AAPL": (None, None)})
    monkeypatch.setattr(service, "store_batch_in_cache", lambda batch_data, also_store_db=True: None)

    fetched_symbols = []

    def fake_fetch(self, symbols, period="2y", start_batch_size=None):
        fetched_symbols.append(list(symbols))
        return {
            "AAPL": {
                "symbol": "AAPL",
                "price_data": _price_df(date(2026, 3, 18), 200.0),
                "info": None,
                "fundamentals": None,
                "has_error": False,
                "error": None,
            }
        }

    monkeypatch.setattr(BulkDataFetcher, "fetch_prices_in_batches", fake_fetch)

    result = service.get_many(["AAPL"], period="2y")

    assert fetched_symbols == [["AAPL"]]
    assert float(result["AAPL"]["Close"].iloc[-1]) == 200.0


def test_track_symbol_failures_commits_success_only_counter_resets(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    import app.tasks.cache_tasks as module

    monkeypatch.setattr(module, "SessionLocal", TestingSessionLocal)

    db = TestingSessionLocal()
    db.add(
        StockUniverse(
            symbol="AAPL",
            exchange="NASDAQ",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
            status_reason="active",
            consecutive_fetch_failures=2,
        )
    )
    db.commit()
    db.close()

    class _StubPriceCache:
        SYMBOL_FAILURE_THRESHOLD = 3

        @staticmethod
        def clear_symbol_failure(symbol):
            return None

        @staticmethod
        def record_symbol_failure(symbol):
            return 0

    _track_symbol_failures(_StubPriceCache(), successes=["AAPL"], failures=[])

    db = TestingSessionLocal()
    record = db.query(StockUniverse).filter(StockUniverse.symbol == "AAPL").one()
    assert record.consecutive_fetch_failures == 0
    assert record.last_fetch_success_at is not None
    db.close()


def test_track_symbol_failures_skips_corrupt_symbol_updates_and_commits_others(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    import app.tasks.cache_tasks as module
    import app.services.stock_universe_service as universe_module

    monkeypatch.setattr(module, "SessionLocal", TestingSessionLocal)

    db = TestingSessionLocal()
    db.add_all(
        [
            StockUniverse(
                symbol="AAPL",
                exchange="NASDAQ",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                status_reason="active",
                consecutive_fetch_failures=2,
            ),
            StockUniverse(
                symbol="MSFT",
                exchange="NASDAQ",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                status_reason="active",
                consecutive_fetch_failures=2,
            ),
        ]
    )
    db.commit()
    db.close()

    original = universe_module.stock_universe_service.record_fetch_success

    def corrupt_one_symbol(session, symbol):
        if symbol == "MSFT":
            raise sqlite3.DatabaseError("database disk image is malformed")
        return original(session, symbol)

    monkeypatch.setattr(
        universe_module.stock_universe_service,
        "record_fetch_success",
        corrupt_one_symbol,
    )

    class _StubPriceCache:
        SYMBOL_FAILURE_THRESHOLD = 3

        @staticmethod
        def clear_symbol_failure(symbol):
            return None

        @staticmethod
        def record_symbol_failure(symbol):
            return 0

    _track_symbol_failures(_StubPriceCache(), successes=["AAPL", "MSFT"], failures=[])

    db = TestingSessionLocal()
    aapl = db.query(StockUniverse).filter(StockUniverse.symbol == "AAPL").one()
    msft = db.query(StockUniverse).filter(StockUniverse.symbol == "MSFT").one()
    assert aapl.consecutive_fetch_failures == 0
    assert aapl.last_fetch_success_at is not None
    assert msft.consecutive_fetch_failures == 2
    assert msft.last_fetch_success_at is None
    db.close()


def test_force_refresh_stale_intraday_skips_inactive_symbols(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    import app.tasks.cache_tasks as module

    monkeypatch.setattr(module, "SessionLocal", TestingSessionLocal)

    db = TestingSessionLocal()
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
        ]
    )
    db.commit()
    db.close()

    class _StubPriceCache:
        @staticmethod
        def get_stale_intraday_symbols():
            return ["AAPL", "DEAD"]

        @staticmethod
        def store_batch_in_cache(batch_data, also_store_db=True):
            return None

    fetched_batches = []

    def fake_fetch(self, symbols, period="2y", start_batch_size=None):
        fetched_batches.append(list(symbols))
        return {symbol: _success_result(symbol) for symbol in symbols}

    monkeypatch.setattr(
        "app.services.price_cache_service.PriceCacheService.get_instance",
        lambda: _StubPriceCache(),
    )
    monkeypatch.setattr(
        "app.services.bulk_data_fetcher.BulkDataFetcher.fetch_prices_in_batches",
        fake_fetch,
    )

    result = _force_refresh_stale_intraday_impl(task=None, symbols=None)

    assert fetched_batches == [["AAPL"]]
    assert result["total"] == 1
    assert result["refreshed"] == 1
