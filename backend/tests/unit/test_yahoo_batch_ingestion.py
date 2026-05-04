from __future__ import annotations

import json
import pickle
from datetime import date, datetime, timedelta
import sqlite3
from unittest.mock import MagicMock
import pandas as pd
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.database import Base
from app.models.stock import StockPrice
from app.models.stock_universe import (
    StockUniverse,
    UNIVERSE_STATUS_ACTIVE,
    UNIVERSE_STATUS_INACTIVE_NO_DATA,
)
from app.services.bulk_data_fetcher import BulkDataFetcher
from app.services.price_cache_service import PriceCacheService
from app.services.stock_universe_service import StockUniverseService
from app.tasks.cache_tasks import (
    _force_refresh_stale_intraday_impl,
    _track_symbol_failures,
    cleanup_old_price_data,
)


def _patch_cache_tasks_session_factory(monkeypatch, module, testing_session_local):
    if hasattr(module, "get_session_factory"):
        monkeypatch.setattr(module, "get_session_factory", lambda: testing_session_local)
    else:
        monkeypatch.setattr(module, "SessionLocal", testing_session_local)


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
        self.keys = []

    def get(self, key):
        self.keys.append(key)
        return self

    def execute(self):
        return list(self._results)


class _FakeRedis:
    def __init__(self, results):
        self._results = results
        self.pipeline_instance = None

    def pipeline(self):
        self.pipeline_instance = _FakePipeline(self._results)
        return self.pipeline_instance


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


def test_bulk_data_fetcher_reuses_fallback_rate_limiter():
    fetcher_one = BulkDataFetcher()
    fetcher_two = BulkDataFetcher()

    assert fetcher_one._rate_limiter is fetcher_two._rate_limiter


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


def test_fetch_prices_in_batches_delays_growth_until_cooldown_expires(monkeypatch):
    fetcher = BulkDataFetcher()
    observed_batch_sizes = []

    def fake_fetch_price_batch_with_retries(batch_symbols, *, period, initial_batch_size):
        _ = period
        observed_batch_sizes.append(initial_batch_size)
        if len(observed_batch_sizes) == 1:
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
        if len(observed_batch_sizes) == 2:
            results = {symbol: _success_result(symbol) for symbol in batch_symbols}
            for symbol in batch_symbols[:2]:
                results[symbol] = {
                    "symbol": symbol,
                    "price_data": None,
                    "info": None,
                    "fundamentals": None,
                    "has_error": True,
                    "error": "429 rate limited",
                }
            return results
        return {symbol: _success_result(symbol) for symbol in batch_symbols}

    class _StubRateLimiter:
        @staticmethod
        def wait(*args, **kwargs):
            return None

    monkeypatch.setattr(fetcher, "_fetch_price_batch_with_retries", fake_fetch_price_batch_with_retries)
    monkeypatch.setattr(fetcher, "_rate_limiter", _StubRateLimiter())
    monkeypatch.setattr("app.services.bulk_data_fetcher.settings.yfinance_batch_rate_limit_interval", 0)

    symbols = [f"SYM{i}" for i in range(550)]
    fetcher.fetch_prices_in_batches(symbols, period="2y", start_batch_size=100)

    assert observed_batch_sizes[:8] == [100, 50, 50, 50, 50, 50, 50, 75]


def test_fetch_prices_in_batches_uses_krx_first_for_korea(monkeypatch):
    price_frame = _price_df(date(2026, 4, 29), 105.0)
    krx_service = MagicMock()
    krx_service.daily_ohlcv_dataframe.return_value = price_frame
    fetcher = BulkDataFetcher(krx_price_service=krx_service)
    yahoo_fallback = MagicMock(return_value={})
    monkeypatch.setattr(fetcher, "_fetch_yfinance_prices_in_batches", yahoo_fallback)

    results = fetcher.fetch_prices_in_batches(["005930.KS"], period="7d", market="KR")

    krx_service.daily_ohlcv_dataframe.assert_called_once_with("005930", period="7d")
    yahoo_fallback.assert_not_called()
    assert results["005930.KS"]["provider"] == "krx"
    assert results["005930.KS"]["price_data"].equals(price_frame)


def test_fetch_prices_in_batches_falls_back_to_yahoo_for_krx_misses(monkeypatch):
    price_frame = _price_df(date(2026, 4, 29), 105.0)
    krx_service = MagicMock()
    krx_service.daily_ohlcv_dataframe.side_effect = [price_frame, None]
    fetcher = BulkDataFetcher(krx_price_service=krx_service)

    def fake_yahoo(symbols, *, period, start_batch_size=None, market=None):
        assert symbols == ["091990.KQ"]
        assert period == "7d"
        assert market == "KR"
        return {"091990.KQ": _success_result("091990.KQ")}

    monkeypatch.setattr(fetcher, "_fetch_yfinance_prices_in_batches", fake_yahoo)

    results = fetcher.fetch_prices_in_batches(
        ["005930.KS", "091990.KQ"],
        period="7d",
        market="KR",
    )

    assert results["005930.KS"]["provider"] == "krx"
    assert results["091990.KQ"]["has_error"] is False
    assert results["091990.KQ"]["provider"] == "yfinance"
    assert results["091990.KQ"]["fallback_from"] == "krx"
    assert results["091990.KQ"]["primary_provider_failed"] is True
    assert results["091990.KQ"]["primary_provider_error"] == "KRX returned empty price data"


def test_price_cache_fetch_direct_historical_data_prefers_krx_for_korea(monkeypatch):
    service = PriceCacheService(redis_client=None, session_factory=MagicMock())
    price_frame = _price_df(date(2026, 4, 29), 105.0)
    monkeypatch.setattr(
        service,
        "_fetch_kr_historical_data",
        lambda symbol, *, period: price_frame,
    )

    result = service._fetch_direct_historical_data("005930.KS", period="7d")

    assert result is price_frame


def test_price_cache_bulk_fallback_passes_market_to_batch_fetcher(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db = TestingSessionLocal()
    db.add(
        StockUniverse(
            symbol="005930.KS",
            name="Samsung Electronics",
            market="KR",
            exchange="KOSPI",
            currency="KRW",
            timezone="Asia/Seoul",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
        )
    )
    db.commit()
    db.close()

    service = PriceCacheService(redis_client=None, session_factory=TestingSessionLocal)
    monkeypatch.setattr(service, "store_batch_in_cache", lambda payload, also_store_db=True: len(payload))
    calls = []
    price_frame = _price_df(date(2026, 4, 29), 105.0)

    class _FakeFetcher:
        def fetch_prices_in_batches(self, symbols, period="2y", start_batch_size=None, market=None):
            calls.append({"symbols": list(symbols), "period": period, "market": market})
            return {
                symbol: {"price_data": price_frame, "has_error": False}
                for symbol in symbols
            }

    import app.services.bulk_data_fetcher as bulk_module

    monkeypatch.setattr(bulk_module, "BulkDataFetcher", lambda: _FakeFetcher())

    result = service._resolve_bulk_fallback(
        ["005930.KS"],
        period="7d",
        expected_date=date(2026, 4, 29),
        now_et=datetime(2026, 4, 29, 16, 0),
    )

    assert calls == [{"symbols": ["005930.KS"], "period": "7d", "market": "KR"}]
    assert result["005930.KS"] is price_frame


def test_store_in_database_replaces_latest_day_row(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    import app.services.price_cache_service as module

    service = PriceCacheService(redis_client=None, session_factory=TestingSessionLocal)
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

    _patch_cache_tasks_session_factory(monkeypatch, module, TestingSessionLocal)

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

    service = PriceCacheService(redis_client=fake_redis, session_factory=TestingSessionLocal)

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

    _patch_cache_tasks_session_factory(monkeypatch, module, TestingSessionLocal)

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
        SYMBOL_FAILURE_THRESHOLD = 5

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


def test_get_many_without_redis_uses_bulk_database_fallback(monkeypatch):
    service = PriceCacheService(redis_client=None, session_factory=lambda: MagicMock())
    service._redis_client = None
    expected_df = _price_df(date(2026, 3, 18), 123.0)

    bulk_db_lookup = MagicMock(return_value={"AAPL": (expected_df, date(2026, 3, 18))})
    monkeypatch.setattr(service, "_get_many_from_database", bulk_db_lookup)
    monkeypatch.setattr(service, "get_historical_data", MagicMock(side_effect=AssertionError("per-symbol fallback should not run")))
    monkeypatch.setattr(service, "_get_expected_data_date", lambda: date(2026, 3, 18))
    monkeypatch.setattr("app.services.price_cache_service.get_eastern_now", lambda: datetime(2026, 3, 18, 17, 0, 0))

    result = service.get_many(["AAPL"], period="2y")

    bulk_db_lookup.assert_called_once_with(["AAPL"], "2y")
    assert result["AAPL"] is expected_df


def test_get_many_reads_market_scoped_redis_keys(monkeypatch):
    import app.services.price_cache_service as module

    data = pd.DataFrame(
        {"Close": list(range(200))},
        index=pd.date_range(end="2026-03-18", periods=200),
    )
    fake_redis = _FakeRedis([pickle.dumps(data), json.dumps({"needs_refresh_after_close": False})])
    service = PriceCacheService(redis_client=fake_redis, session_factory=lambda: MagicMock())

    monkeypatch.setattr(module, "get_bulk_redis_client", lambda: None)
    monkeypatch.setattr(module, "get_eastern_now", lambda: datetime(2026, 3, 18, 17, 0, 0))
    monkeypatch.setattr(service, "_get_expected_data_date", lambda: date(2026, 3, 18))

    result = service.get_many(["0700.HK"], period="2y", market_by_symbol={"0700.HK": "HK"})

    assert result["0700.HK"] is not None
    assert fake_redis.pipeline_instance.keys == [
        "price:HK:0700.HK:recent",
        "price:HK:0700.HK:fetch_meta",
    ]


def test_bulk_fallback_writes_fetched_prices_to_symbol_market_scope(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    service = PriceCacheService(redis_client=None, session_factory=TestingSessionLocal)
    service._redis_client = object()

    db = TestingSessionLocal()
    db.add(
        StockUniverse(
            symbol="0700.HK",
            market="HK",
            exchange="HKEX",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
            status_reason="active",
        )
    )
    db.commit()
    db.close()

    monkeypatch.setattr(service, "_get_many_from_database", lambda symbols, period: {"0700.HK": (None, None)})

    def fake_fetch(self, symbols, period="2y", start_batch_size=None, market=None):
        assert market == "HK"
        return {
            "0700.HK": {
                "symbol": "0700.HK",
                "price_data": _price_df(date(2026, 3, 18), 200.0),
                "info": None,
                "fundamentals": None,
                "has_error": False,
                "error": None,
            }
        }

    stored = []
    monkeypatch.setattr(BulkDataFetcher, "fetch_prices_in_batches", fake_fetch)
    monkeypatch.setattr(
        service,
        "store_batch_in_cache",
        lambda batch_data, also_store_db=True, market=None: stored.append((set(batch_data), market)),
    )

    result = service._resolve_bulk_fallback(
        ["0700.HK"],
        period="2y",
        expected_date=date(2026, 3, 18),
        now_et=datetime(2026, 3, 18, 17, 0, 0),
        market_by_symbol={"0700.HK": "HK"},
    )

    assert float(result["0700.HK"]["Close"].iloc[-1]) == 200.0
    assert stored == [({"0700.HK"}, "HK")]


def test_bulk_fallback_warms_fresh_db_hits_to_inferred_symbol_market(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    service = PriceCacheService(redis_client=None, session_factory=TestingSessionLocal)
    service._redis_client = object()

    db = TestingSessionLocal()
    db.add(
        StockUniverse(
            symbol="0700.HK",
            market="HK",
            exchange="HKEX",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
            status_reason="active",
        )
    )
    db.commit()
    db.close()

    fresh_df = _price_df(date(2026, 3, 18), 200.0)
    monkeypatch.setattr(service, "_get_many_from_database", lambda symbols, period: {"0700.HK": (fresh_df, date(2026, 3, 18))})
    stored = []
    monkeypatch.setattr(
        service,
        "_store_recent_in_redis",
        lambda symbol, data, market=None: stored.append((symbol, market)),
    )

    result = service._resolve_bulk_fallback(
        ["0700.HK"],
        period="2y",
        expected_date=date(2026, 3, 18),
        now_et=datetime(2026, 3, 18, 17, 0, 0),
    )

    assert result["0700.HK"] is fresh_df
    assert stored == [("0700.HK", "HK")]


def test_get_many_cached_only_fresh_filters_stale_database_rows(monkeypatch):
    service = PriceCacheService(redis_client=None, session_factory=lambda: MagicMock())
    fresh_df = _price_df(date(2026, 3, 18), 123.0)
    stale_df = _price_df(date(2026, 3, 17), 111.0)

    monkeypatch.setattr(
        service,
        "_get_many_from_database",
        lambda symbols, period: {
            "AAPL": (fresh_df, date(2026, 3, 18)),
            "MSFT": (stale_df, date(2026, 3, 17)),
            "NVDA": (None, None),
        },
    )
    monkeypatch.setattr(service, "_is_data_fresh", lambda last_date: last_date == date(2026, 3, 18))

    result = service.get_many_cached_only_fresh(["AAPL", "MSFT", "NVDA"], period="2y")

    assert result["AAPL"] is fresh_df
    assert result["MSFT"] is None
    assert result["NVDA"] is None


def test_get_many_from_database_chunks_large_symbol_sets(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    service = PriceCacheService(redis_client=None, session_factory=TestingSessionLocal)
    symbols = [f"SYM{i}" for i in range(5)]
    end_day = date.today()

    db = TestingSessionLocal()
    for symbol in symbols:
        for offset in range(60):
            day = end_day - timedelta(days=59 - offset)
            db.add(
                StockPrice(
                    symbol=symbol,
                    date=day,
                    open=10.0,
                    high=11.0,
                    low=9.0,
                    close=10.5,
                    volume=1000,
                    adj_close=10.5,
                )
            )
    db.commit()
    db.close()

    select_statements: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def _capture_selects(conn, cursor, statement, parameters, context, executemany):
        if "FROM stock_prices" in statement:
            select_statements.append(statement)

    monkeypatch.setattr(settings, "price_cache_db_chunk_size", 2)

    result = service._get_many_from_database(symbols, "2y")

    assert [symbol for symbol, (df, _) in result.items() if df is not None] == symbols
    assert len(select_statements) == 3


def test_track_symbol_failures_skips_corrupt_symbol_updates_and_commits_others(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    import app.tasks.cache_tasks as module
    _patch_cache_tasks_session_factory(monkeypatch, module, TestingSessionLocal)

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

    stock_universe_service = StockUniverseService()
    original = stock_universe_service.record_fetch_success

    def corrupt_one_symbol(session, symbol):
        if symbol == "MSFT":
            raise sqlite3.DatabaseError("database disk image is malformed")
        return original(session, symbol)

    monkeypatch.setattr(stock_universe_service, "record_fetch_success", corrupt_one_symbol)
    monkeypatch.setattr(module, "get_stock_universe_service", lambda: stock_universe_service)

    class _StubPriceCache:
        SYMBOL_FAILURE_THRESHOLD = 5

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


def test_track_symbol_failures_passes_updated_deactivation_threshold(monkeypatch):
    import app.tasks.cache_tasks as module

    captured = {}

    def record_fetch_failure(db, symbol, **kwargs):
        captured["symbol"] = symbol
        captured["deactivate_threshold"] = kwargs["deactivate_threshold"]
        return {"deactivated": kwargs["deactivate_threshold"] <= 5}

    stock_universe_service = StockUniverseService()
    monkeypatch.setattr(stock_universe_service, "record_fetch_failure", record_fetch_failure)
    monkeypatch.setattr(module, "get_stock_universe_service", lambda: stock_universe_service)

    class _StubPriceCache:
        SYMBOL_FAILURE_THRESHOLD = 5

        @staticmethod
        def clear_symbol_failure(symbol):
            return None

        @staticmethod
        def record_symbol_failure(symbol):
            return 5

    fake_db = MagicMock()
    _track_symbol_failures(
        _StubPriceCache(),
        successes=[],
        failures=["AAPL"],
        db=fake_db,
        failure_details={"AAPL": "Possibly delisted; no data found"},
    )

    assert captured == {"symbol": "AAPL", "deactivate_threshold": 5}


def test_force_refresh_stale_intraday_skips_inactive_symbols(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    import app.tasks.cache_tasks as module

    _patch_cache_tasks_session_factory(monkeypatch, module, TestingSessionLocal)

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
        "app.wiring.bootstrap.get_price_cache",
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
