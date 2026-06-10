from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.stock import StockPrice
from app.models.stock_universe import StockUniverse
from app.services.static_daily_price_refresh_service import (
    STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD,
    STATIC_DAILY_PRICE_REFRESH_PERIOD,
    STATIC_DAILY_PRICE_REFRESH_BATCH_SIZE,
    STATIC_RATE_LIMITED_RETRY_BATCH_SIZE,
    STATIC_RATE_LIMITED_RETRY_WAIT_SECONDS,
    StaticDailyPriceRefreshService,
    static_daily_price_refresh_batch_size,
)


IN_KEY_MARKET_PRICE_SYMBOLS = ["^NSEI", "NIFTYBEES.NS", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
HK_KEY_MARKET_PRICE_SYMBOLS = ["^HSI", "2800.HK", "0700.HK", "3690.HK", "0941.HK"]
US_KEY_MARKET_PRICE_SYMBOLS = [
    "SPY",
    "QQQ",
    "IWM",
    "DX-Y.NYB",
    "SGD=X",
    "BTC-USD",
    "GLD",
    "TLT",
    "^VIX",
]


def _sqlite_session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        engine,
        tables=[StockUniverse.__table__, StockPrice.__table__],
    )
    return sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )


def test_static_daily_price_refresh_service_fetches_stale_and_no_history_groups() -> None:
    session_factory = _sqlite_session_factory()

    with session_factory() as db:
        db.add_all(
            [
                StockUniverse(symbol="OLD.NS", market="IN", is_active=True, market_cap=100.0),
                StockUniverse(symbol="NEW.NS", market="IN", is_active=True, market_cap=90.0),
            ]
        )
        db.add(
            StockPrice(
                symbol="OLD.NS",
                date=date(2026, 6, 3),
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                volume=1000,
            )
        )
        db.commit()

    fetch_calls: list[dict] = []
    stored_batches: list[dict] = []

    class _FakeFetcher:
        def fetch_prices_in_batches(self, symbols, period="2y", start_batch_size=None, market=None):
            fetch_calls.append(
                {
                    "symbols": list(symbols),
                    "period": period,
                    "start_batch_size": start_batch_size,
                    "market": market,
                }
            )
            return {
                symbol: {"price_data": SimpleNamespace(empty=False), "has_error": False}
                for symbol in symbols
            }

    service = StaticDailyPriceRefreshService(
        session_factory=session_factory,
        price_cache=SimpleNamespace(
            store_batch_in_cache=lambda payload, also_store_db=True, market=None: stored_batches.append(
                {
                    "symbols": sorted(payload.keys()),
                    "also_store_db": also_store_db,
                    "market": market,
                }
            )
        ),
        fetcher=_FakeFetcher(),
        batch_size_for_market=lambda market: 25,
        sleep=lambda _seconds: None,
    )

    result = service.refresh(as_of_date=date(2026, 6, 4), market="IN")

    assert fetch_calls == [
        {
            "symbols": ["OLD.NS"],
            "period": STATIC_DAILY_PRICE_REFRESH_PERIOD,
            "start_batch_size": 25,
            "market": "IN",
        },
        {
            "symbols": ["NEW.NS", *IN_KEY_MARKET_PRICE_SYMBOLS],
            "period": STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD,
            "start_batch_size": 25,
            "market": "IN",
        },
    ]
    assert stored_batches == [
        {"symbols": ["OLD.NS"], "also_store_db": True, "market": "IN"},
        {
            "symbols": ["HDFCBANK.NS", "NEW.NS", "NIFTYBEES.NS", "RELIANCE.NS", "TCS.NS", "^NSEI"],
            "also_store_db": True,
            "market": "IN",
        },
    ]
    assert result["stale_symbols"] == 1
    assert result["key_market_symbols"] == len(IN_KEY_MARKET_PRICE_SYMBOLS)
    assert result["no_history_symbols"] == 6
    assert result["yahoo_fetched_symbols"] == 7


def test_static_daily_price_refresh_service_filters_to_selected_market() -> None:
    session_factory = _sqlite_session_factory()

    with session_factory() as db:
        db.add_all(
            [
                StockUniverse(symbol="0700.HK", market="HK", is_active=True, market_cap=100.0),
                StockUniverse(symbol="9988.HK", market="HK", is_active=True, market_cap=90.0),
                StockUniverse(symbol="AAPL", market="US", is_active=True, market_cap=120.0),
                StockUniverse(symbol="BAD-W", market="HK", is_active=True, market_cap=80.0),
            ]
        )
        for symbol in ("0700.HK", "AAPL"):
            db.add(
                StockPrice(
                    symbol=symbol,
                    date=date(2026, 4, 1),
                    open=1.0,
                    high=1.0,
                    low=1.0,
                    close=1.0,
                    volume=1000,
                )
            )
        db.commit()

    fetch_calls: list[dict] = []
    stored_batches: list[dict] = []

    class _FakeFetcher:
        def fetch_prices_in_batches(self, symbols, period="2y", start_batch_size=None, market=None):
            fetch_calls.append(
                {
                    "symbols": list(symbols),
                    "period": period,
                    "start_batch_size": start_batch_size,
                    "market": market,
                }
            )
            return {
                symbol: {"price_data": SimpleNamespace(empty=False), "has_error": False}
                for symbol in symbols
            }

    service = StaticDailyPriceRefreshService(
        session_factory=session_factory,
        price_cache=SimpleNamespace(
            store_batch_in_cache=lambda payload, also_store_db=True, market=None: stored_batches.append(
                {
                    "symbols": sorted(payload.keys()),
                    "also_store_db": also_store_db,
                    "market": market,
                }
            )
        ),
        fetcher=_FakeFetcher(),
        batch_size_for_market=lambda _market: 25,
        sleep=lambda _seconds: None,
    )

    result = service.refresh(as_of_date=date(2026, 4, 2), market="HK")

    assert result["market"] == "HK"
    assert result["total_active_symbols"] == 3
    assert result["supported_symbols"] == 6
    assert result["key_market_symbols"] == len(HK_KEY_MARKET_PRICE_SYMBOLS)
    assert result["skipped_unsupported_symbols"] == 1
    assert fetch_calls == [
        {
            "symbols": ["0700.HK"],
            "period": STATIC_DAILY_PRICE_REFRESH_PERIOD,
            "start_batch_size": 25,
            "market": "HK",
        },
        {
            "symbols": ["9988.HK", "^HSI", "2800.HK", "3690.HK", "0941.HK"],
            "period": STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD,
            "start_batch_size": 25,
            "market": "HK",
        },
    ]
    assert stored_batches == [
        {"symbols": ["0700.HK"], "also_store_db": True, "market": "HK"},
        {
            "symbols": ["0941.HK", "2800.HK", "3690.HK", "9988.HK", "^HSI"],
            "also_store_db": True,
            "market": "HK",
        },
    ]


def test_static_daily_price_refresh_batch_size_uses_market_policy(monkeypatch) -> None:
    import app.services.rate_budget_policy as rate_budget_policy

    calls = []

    class _FakePolicy:
        def get_batch_size(self, provider, market):
            calls.append((provider, market))
            return 31

    monkeypatch.setattr(rate_budget_policy, "get_rate_budget_policy", lambda: _FakePolicy())

    assert static_daily_price_refresh_batch_size("IN") == 31
    assert calls == [("yfinance", "IN")]
    assert static_daily_price_refresh_batch_size(None) == STATIC_DAILY_PRICE_REFRESH_BATCH_SIZE


def test_static_daily_price_refresh_includes_us_key_market_data_symbols() -> None:
    session_factory = _sqlite_session_factory()

    fetch_calls: list[dict] = []
    stored_batches: list[dict] = []

    class _FakeFetcher:
        def fetch_prices_in_batches(self, symbols, period="2y", start_batch_size=None, market=None):
            fetch_calls.append(
                {
                    "symbols": list(symbols),
                    "period": period,
                    "start_batch_size": start_batch_size,
                    "market": market,
                }
            )
            return {
                symbol: {"price_data": SimpleNamespace(empty=False), "has_error": False}
                for symbol in symbols
            }

    service = StaticDailyPriceRefreshService(
        session_factory=session_factory,
        price_cache=SimpleNamespace(
            store_batch_in_cache=lambda payload, also_store_db=True, market=None: stored_batches.append(
                {
                    "symbols": sorted(payload.keys()),
                    "also_store_db": also_store_db,
                    "market": market,
                }
            )
        ),
        fetcher=_FakeFetcher(),
        batch_size_for_market=lambda _market: 25,
        sleep=lambda _seconds: None,
    )

    result = service.refresh(as_of_date=date(2026, 6, 4), market="US")

    assert fetch_calls == [
        {
            "symbols": US_KEY_MARKET_PRICE_SYMBOLS,
            "period": STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD,
            "start_batch_size": 25,
            "market": "US",
        },
    ]
    assert stored_batches == [
        {
            "symbols": ["BTC-USD", "DX-Y.NYB", "GLD", "IWM", "QQQ", "SGD=X", "SPY", "TLT", "^VIX"],
            "also_store_db": True,
            "market": "US",
        },
    ]
    assert result["total_active_symbols"] == 0
    assert result["key_market_symbols"] == len(US_KEY_MARKET_PRICE_SYMBOLS)
    assert result["no_history_symbols"] == len(US_KEY_MARKET_PRICE_SYMBOLS)
    assert result["yahoo_fetched_symbols"] == len(US_KEY_MARKET_PRICE_SYMBOLS)


def _seed_in_universe(session_factory) -> None:
    with session_factory() as db:
        db.add_all(
            [
                StockUniverse(symbol="RELIANCE.NS", market="IN", is_active=True, market_cap=300.0),
                StockUniverse(symbol="TCS.NS", market="IN", is_active=True, market_cap=200.0),
                StockUniverse(symbol="INFY.NS", market="IN", is_active=True, market_cap=100.0),
            ]
        )
        for symbol in ("RELIANCE.NS", "TCS.NS", "INFY.NS"):
            db.add(
                StockPrice(
                    symbol=symbol,
                    date=date(2026, 4, 1),
                    open=1.0,
                    high=1.0,
                    low=1.0,
                    close=1.0,
                    volume=1000,
                )
            )
        db.commit()


def test_static_daily_price_refresh_retries_rate_limited_failures() -> None:
    session_factory = _sqlite_session_factory()
    _seed_in_universe(session_factory)

    fetch_calls: list[dict] = []
    stored_batches: list[dict] = []
    sleeps: list[float] = []

    class _FakeFetcher:
        def fetch_prices_in_batches(self, symbols, period="2y", start_batch_size=None, market=None):
            fetch_calls.append(
                {
                    "symbols": list(symbols),
                    "period": period,
                    "start_batch_size": start_batch_size,
                    "market": market,
                }
            )
            if len(fetch_calls) == 1:
                return {
                    "RELIANCE.NS": {
                        "price_data": SimpleNamespace(empty=False),
                        "has_error": False,
                    },
                    "TCS.NS": {
                        "price_data": None,
                        "has_error": True,
                        "error": "Too Many Requests (429)",
                    },
                    "INFY.NS": {
                        "price_data": None,
                        "has_error": True,
                        "error": "delisted: no price data",
                    },
                }
            if period == STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD:
                return {
                    symbol: {"price_data": SimpleNamespace(empty=False), "has_error": False}
                    for symbol in symbols
                }
            return {
                "TCS.NS": {
                    "price_data": SimpleNamespace(empty=False),
                    "has_error": False,
                },
            }

    service = StaticDailyPriceRefreshService(
        session_factory=session_factory,
        price_cache=SimpleNamespace(
            store_batch_in_cache=lambda payload, also_store_db=True, market=None: stored_batches.append(
                {
                    "symbols": sorted(payload.keys()),
                    "also_store_db": also_store_db,
                    "market": market,
                }
            )
        ),
        fetcher=_FakeFetcher(),
        sleep=lambda seconds: sleeps.append(seconds),
    )

    result = service.refresh(as_of_date=date(2026, 4, 2), market="IN")

    assert sleeps == [STATIC_RATE_LIMITED_RETRY_WAIT_SECONDS]
    assert len(fetch_calls) == 3
    assert fetch_calls[1]["symbols"] == ["^NSEI", "NIFTYBEES.NS", "HDFCBANK.NS"]
    assert fetch_calls[2]["symbols"] == ["TCS.NS"]
    assert fetch_calls[2]["start_batch_size"] == STATIC_RATE_LIMITED_RETRY_BATCH_SIZE
    assert fetch_calls[2]["market"] == "IN"
    stored_symbols = {symbol for batch in stored_batches for symbol in batch["symbols"]}
    assert stored_symbols == {"RELIANCE.NS", "TCS.NS", "^NSEI", "NIFTYBEES.NS", "HDFCBANK.NS"}
    assert result["rate_limited_retry"] == {
        "attempted": 1,
        "recovered": 1,
        "still_failed": 0,
        "wait_seconds": STATIC_RATE_LIMITED_RETRY_WAIT_SECONDS,
        "batch_size": STATIC_RATE_LIMITED_RETRY_BATCH_SIZE,
    }
    assert result["key_market_symbols"] == len(IN_KEY_MARKET_PRICE_SYMBOLS)
    assert result["yahoo_fetched_symbols"] == 5
    assert result["yahoo_failed_symbols"] == 1


def test_static_daily_price_refresh_retries_no_history_rate_limits_with_bootstrap_period() -> None:
    session_factory = _sqlite_session_factory()

    with session_factory() as db:
        db.add(StockUniverse(symbol="NEW.NS", market="IN", is_active=True, market_cap=100.0))
        db.commit()

    fetch_calls: list[dict] = []
    stored_batches: list[dict] = []
    sleeps: list[float] = []

    class _FakeFetcher:
        def fetch_prices_in_batches(self, symbols, period="2y", start_batch_size=None, market=None):
            fetch_calls.append(
                {
                    "symbols": list(symbols),
                    "period": period,
                    "start_batch_size": start_batch_size,
                    "market": market,
                }
            )
            if len(fetch_calls) == 1:
                return {
                    "NEW.NS": {
                        "price_data": None,
                        "has_error": True,
                        "error": "Too Many Requests (429)",
                    },
                }
            return {
                "NEW.NS": {
                    "price_data": SimpleNamespace(empty=False),
                    "has_error": False,
                },
            }

    service = StaticDailyPriceRefreshService(
        session_factory=session_factory,
        price_cache=SimpleNamespace(
            store_batch_in_cache=lambda payload, also_store_db=True, market=None: stored_batches.append(
                {
                    "symbols": sorted(payload.keys()),
                    "also_store_db": also_store_db,
                    "market": market,
                }
            )
        ),
        fetcher=_FakeFetcher(),
        batch_size_for_market=lambda _market: 25,
        sleep=lambda seconds: sleeps.append(seconds),
    )

    result = service.refresh(as_of_date=date(2026, 6, 4), market="IN")

    assert sleeps == [STATIC_RATE_LIMITED_RETRY_WAIT_SECONDS]
    assert fetch_calls == [
        {
            "symbols": ["NEW.NS", *IN_KEY_MARKET_PRICE_SYMBOLS],
            "period": STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD,
            "start_batch_size": 25,
            "market": "IN",
        },
        {
            "symbols": ["NEW.NS"],
            "period": STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD,
            "start_batch_size": STATIC_RATE_LIMITED_RETRY_BATCH_SIZE,
            "market": "IN",
        },
    ]
    assert stored_batches == [
        {"symbols": ["NEW.NS"], "also_store_db": True, "market": "IN"},
    ]
    assert result["key_market_symbols"] == len(IN_KEY_MARKET_PRICE_SYMBOLS)
    assert result["no_history_symbols"] == 6
    assert result["rate_limited_retry"]["recovered"] == 1
    assert result["yahoo_fetched_symbols"] == 1
    assert result["yahoo_failed_symbols"] == 0


def test_static_daily_price_refresh_skips_retry_for_non_in_markets() -> None:
    session_factory = _sqlite_session_factory()

    with session_factory() as db:
        db.add(StockUniverse(symbol="0700.HK", market="HK", is_active=True, market_cap=100.0))
        db.add(
            StockPrice(
                symbol="0700.HK",
                date=date(2026, 4, 1),
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                volume=1000,
            )
        )
        db.commit()

    fetch_calls: list[dict] = []
    sleeps: list[float] = []

    class _FakeFetcher:
        def fetch_prices_in_batches(self, symbols, period="2y", start_batch_size=None, market=None):
            fetch_calls.append({"symbols": list(symbols), "market": market})
            return {
                symbol: {
                    "price_data": None,
                    "has_error": True,
                    "error": "429 rate limited",
                }
                for symbol in symbols
            }

    service = StaticDailyPriceRefreshService(
        session_factory=session_factory,
        price_cache=SimpleNamespace(store_batch_in_cache=lambda *_args, **_kwargs: None),
        fetcher=_FakeFetcher(),
        sleep=lambda seconds: sleeps.append(seconds),
    )

    result = service.refresh(as_of_date=date(2026, 4, 2), market="HK")

    assert len(fetch_calls) == 2
    assert fetch_calls[1]["symbols"] == ["^HSI", "2800.HK", "3690.HK", "0941.HK"]
    assert sleeps == []
    assert result["rate_limited_retry"] == {
        "attempted": 0,
        "recovered": 0,
        "still_failed": 0,
        "wait_seconds": 0,
        "batch_size": STATIC_RATE_LIMITED_RETRY_BATCH_SIZE,
    }


def test_static_daily_price_refresh_skips_retry_when_no_rate_limited_failures() -> None:
    session_factory = _sqlite_session_factory()
    _seed_in_universe(session_factory)

    fetch_calls: list[dict] = []
    sleeps: list[float] = []

    class _FakeFetcher:
        def fetch_prices_in_batches(self, symbols, period="2y", start_batch_size=None, market=None):
            fetch_calls.append({"symbols": list(symbols)})
            return {
                symbol: {
                    "price_data": None,
                    "has_error": True,
                    "error": "delisted: no price data",
                }
                for symbol in symbols
            }

    service = StaticDailyPriceRefreshService(
        session_factory=session_factory,
        price_cache=SimpleNamespace(store_batch_in_cache=lambda *_args, **_kwargs: None),
        fetcher=_FakeFetcher(),
        sleep=lambda seconds: sleeps.append(seconds),
    )

    result = service.refresh(as_of_date=date(2026, 4, 2), market="IN")

    assert len(fetch_calls) == 2
    assert sleeps == []
    assert result["rate_limited_retry"]["attempted"] == 0
    assert result["key_market_symbols"] == len(IN_KEY_MARKET_PRICE_SYMBOLS)
    assert result["yahoo_failed_symbols"] == 6
