from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.domain.relative_strength import HORIZON_SESSIONS
from app.models.stock import StockPrice
from app.models.stock_universe import StockUniverse
from app.services.static_daily_price_refresh_service import (
    STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD,
    STATIC_DAILY_PRICE_REFRESH_BATCH_SIZE,
    STATIC_DAILY_PRICE_REFRESH_PERIOD,
    STATIC_RATE_LIMITED_RETRY_BATCH_SIZE,
    STATIC_RATE_LIMITED_RETRY_WAIT_SECONDS,
    STATIC_SESSION_REPAIR_WAIT_SECONDS,
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
DE_KEY_MARKET_PRICE_SYMBOLS = ["^GDAXI", "EXS1.DE", "SIE.DE", "ALV.DE"]


class _RRGStartupCalendar:
    @staticmethod
    def trading_days(market, start, end):
        assert market == "IN"
        assert start < end
        return [date(2026, 3, 2), end]

    @staticmethod
    def session_anchors(market, as_of_date, *, offsets):
        assert market == "IN"
        assert tuple(offsets) == tuple(HORIZON_SESSIONS.values())
        return {
            0: as_of_date,
            1: as_of_date - timedelta(days=1),
            5: as_of_date - timedelta(days=7),
            21: date(2026, 1, 30),
            63: date(2025, 12, 1),
            126: date(2025, 9, 1),
            189: date(2025, 6, 2),
            252: date(2025, 3, 3),
        }


class _CompleteBreadthHistoryCoverage:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def classify(self, db, *, market, through_date, symbols):
        self.calls.append(
            {
                "market": market,
                "through_date": through_date,
                "symbols": tuple(symbols),
            }
        )
        return SimpleNamespace(
            incomplete_symbols=(),
            history_incomplete_symbols=(),
            missing_through_date_symbols=(),
            required_price_date_count=2,
        )


class _CompleteGroupHistoryCoverage:
    @staticmethod
    def required_anchor_dates(*, market, through_date):
        return (through_date,)

    @staticmethod
    def classify(db, *, market, through_date, symbols, required_anchor_dates):
        return SimpleNamespace(incomplete_symbols=())


class _CurrentSessionGapBreadthHistoryCoverage:
    @staticmethod
    def classify(db, *, market, through_date, symbols):
        return SimpleNamespace(
            incomplete_symbols=tuple(symbols),
            history_incomplete_symbols=(),
            missing_through_date_symbols=tuple(symbols),
            required_price_date_count=70,
        )


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


def test_static_daily_price_refresh_refetches_fresh_rows_with_missing_or_zero_volume() -> None:
    session_factory = _sqlite_session_factory()

    with session_factory() as db:
        db.add_all(
            [
                StockUniverse(symbol="GOOD", market="US", is_active=True, market_cap=100.0),
                StockUniverse(symbol="ZERO", market="US", is_active=True, market_cap=90.0),
                StockUniverse(symbol="MISSING", market="US", is_active=True, market_cap=80.0),
            ]
        )
        db.add_all(
            [
                StockPrice(
                    symbol="GOOD",
                    date=date(2026, 6, 8),
                    open=1.0,
                    high=1.0,
                    low=1.0,
                    close=1.0,
                    volume=1000,
                ),
                StockPrice(
                    symbol="ZERO",
                    date=date(2026, 6, 8),
                    open=1.0,
                    high=1.0,
                    low=1.0,
                    close=1.0,
                    volume=0,
                ),
                StockPrice(
                    symbol="MISSING",
                    date=date(2026, 6, 8),
                    open=1.0,
                    high=1.0,
                    low=1.0,
                    close=1.0,
                    volume=None,
                ),
            ]
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

    result = service.refresh(as_of_date=date(2026, 6, 8), market="US")

    assert fetch_calls[0] == {
        "symbols": ["ZERO", "MISSING"],
        "period": STATIC_DAILY_PRICE_REFRESH_PERIOD,
        "start_batch_size": 25,
        "market": "US",
    }
    assert "GOOD" not in fetch_calls[0]["symbols"]
    assert stored_batches[0] == {"symbols": ["MISSING", "ZERO"], "also_store_db": True, "market": "US"}
    assert result["db_fresh_symbols"] == 1
    assert result["stale_symbols"] == 2


def test_static_daily_price_refresh_keeps_breadth_current_gap_on_stale_top_up() -> None:
    session_factory = _sqlite_session_factory()

    with session_factory() as db:
        db.add(StockUniverse(symbol="SAP.DE", market="DE", is_active=True, market_cap=100.0))
        db.add(
            StockPrice(
                symbol="SAP.DE",
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

    class _FakeFetcher:
        def fetch_prices_in_batches(
            self, symbols, period="2y", start_batch_size=None, market=None
        ):
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
        price_cache=SimpleNamespace(store_batch_in_cache=lambda *_args, **_kwargs: None),
        fetcher=_FakeFetcher(),
        batch_size_for_market=lambda _market: 25,
        group_history_price_coverage=_CompleteGroupHistoryCoverage(),
        breadth_history_price_coverage=_CurrentSessionGapBreadthHistoryCoverage(),
        sleep=lambda _seconds: None,
    )

    result = service.refresh(
        as_of_date=date(2026, 6, 4),
        market="DE",
        ensure_static_history=True,
    )

    assert fetch_calls == [
        {
            "symbols": ["SAP.DE"],
            "period": STATIC_DAILY_PRICE_REFRESH_PERIOD,
            "start_batch_size": 25,
            "market": "DE",
        },
        {
            "symbols": DE_KEY_MARKET_PRICE_SYMBOLS,
            "period": STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD,
            "start_batch_size": 25,
            "market": "DE",
        }
    ]
    assert result["stale_symbols"] == 1
    assert result["no_history_symbols"] == len(DE_KEY_MARKET_PRICE_SYMBOLS)
    assert result["history_incomplete_symbols"] == 0
    assert result["breadth_history_incomplete_symbols"] == 1
    assert result["breadth_history_missing_through_date_symbols"] == 1


def test_static_daily_price_refresh_top_up_breadth_only_current_gap() -> None:
    session_factory = _sqlite_session_factory()

    with session_factory() as db:
        db.add(StockUniverse(symbol="SAP.DE", market="DE", is_active=True, market_cap=100.0))
        db.add_all(
            [
                StockPrice(
                    symbol="SAP.DE",
                    date=date(2026, 6, 3),
                    open=1.0,
                    high=1.0,
                    low=1.0,
                    close=1.0,
                    volume=1000,
                ),
                StockPrice(
                    symbol="SAP.DE",
                    date=date(2026, 6, 4),
                    open=None,
                    high=2.0,
                    low=2.0,
                    close=2.0,
                    volume=1000,
                ),
            ]
        )
        db.commit()

    fetch_calls: list[dict] = []

    class _FakeFetcher:
        def fetch_prices_in_batches(
            self, symbols, period="2y", start_batch_size=None, market=None
        ):
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
        price_cache=SimpleNamespace(store_batch_in_cache=lambda *_args, **_kwargs: None),
        fetcher=_FakeFetcher(),
        batch_size_for_market=lambda _market: 25,
        group_history_price_coverage=_CompleteGroupHistoryCoverage(),
        breadth_history_price_coverage=_CurrentSessionGapBreadthHistoryCoverage(),
        sleep=lambda _seconds: None,
    )

    result = service.refresh(
        as_of_date=date(2026, 6, 4),
        market="DE",
        ensure_static_history=True,
    )

    assert fetch_calls == [
        {
            "symbols": ["SAP.DE"],
            "period": STATIC_DAILY_PRICE_REFRESH_PERIOD,
            "start_batch_size": 25,
            "market": "DE",
        },
        {
            "symbols": DE_KEY_MARKET_PRICE_SYMBOLS,
            "period": STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD,
            "start_batch_size": 25,
            "market": "DE",
        },
    ]
    assert result["db_fresh_symbols"] == 1
    assert result["stale_symbols"] == 1
    assert result["no_history_symbols"] == len(DE_KEY_MARKET_PRICE_SYMBOLS)
    assert result["history_incomplete_symbols"] == 0
    assert result["breadth_history_incomplete_symbols"] == 1
    assert result["breadth_history_missing_through_date_symbols"] == 1


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


def test_static_daily_price_refresh_hydrates_short_history_for_rrg_startup() -> None:
    session_factory = _sqlite_session_factory()

    with session_factory() as db:
        db.add(
            StockUniverse(symbol="OLD.NS", market="IN", is_active=True, market_cap=100.0)
        )
        db.add(
            StockPrice(
                symbol="OLD.NS",
                date=date(2026, 6, 4),
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                volume=1000,
            )
        )
        db.commit()

    fetch_calls: list[dict] = []

    class _FakeFetcher:
        def fetch_prices_in_batches(
            self, symbols, period="2y", start_batch_size=None, market=None
        ):
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

    breadth_coverage = _CompleteBreadthHistoryCoverage()
    service = StaticDailyPriceRefreshService(
        session_factory=session_factory,
        price_cache=SimpleNamespace(store_batch_in_cache=lambda *_args, **_kwargs: None),
        fetcher=_FakeFetcher(),
        batch_size_for_market=lambda _market: 25,
        calendar_service=_RRGStartupCalendar(),
        breadth_history_price_coverage=breadth_coverage,
        sleep=lambda _seconds: None,
    )

    result = service.refresh(
        as_of_date=date(2026, 6, 4),
        market="IN",
        ensure_static_history=True,
    )

    assert fetch_calls == [
        {
            "symbols": ["OLD.NS", *IN_KEY_MARKET_PRICE_SYMBOLS],
            "period": STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD,
            "start_batch_size": 25,
            "market": "IN",
        }
    ]
    assert result["db_fresh_symbols"] == 1
    assert result["stale_symbols"] == 0
    assert result["no_history_symbols"] == len(IN_KEY_MARKET_PRICE_SYMBOLS)
    assert result["history_incomplete_symbols"] == 1
    assert result["rrg_history_coverage_status"] == "verified"
    assert result["breadth_history_coverage_status"] == "verified"
    assert breadth_coverage.calls == [
        {
            "market": "IN",
            "through_date": date(2026, 6, 4),
            "symbols": ("OLD.NS",),
        }
    ]
    assert result["yahoo_fetched_symbols"] == 6


def test_static_daily_price_refresh_hydrates_sparse_old_rows_for_rrg_startup() -> None:
    session_factory = _sqlite_session_factory()

    with session_factory() as db:
        db.add(
            StockUniverse(
                symbol="SPARSE.NS",
                market="IN",
                is_active=True,
                market_cap=100.0,
            )
        )
        db.add_all(
            [
                StockPrice(
                    symbol="SPARSE.NS",
                    date=date(2025, 1, 2),
                    open=1.0,
                    high=1.0,
                    low=1.0,
                    close=1.0,
                    adj_close=1.0,
                    volume=1000,
                ),
                StockPrice(
                    symbol="SPARSE.NS",
                    date=date(2026, 6, 4),
                    open=2.0,
                    high=2.0,
                    low=2.0,
                    close=2.0,
                    adj_close=2.0,
                    volume=1000,
                ),
            ]
        )
        db.commit()

    fetch_calls: list[dict] = []

    class _FakeFetcher:
        def fetch_prices_in_batches(
            self, symbols, period="2y", start_batch_size=None, market=None
        ):
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

    breadth_coverage = _CompleteBreadthHistoryCoverage()
    service = StaticDailyPriceRefreshService(
        session_factory=session_factory,
        price_cache=SimpleNamespace(store_batch_in_cache=lambda *_args, **_kwargs: None),
        fetcher=_FakeFetcher(),
        batch_size_for_market=lambda _market: 25,
        calendar_service=_RRGStartupCalendar(),
        breadth_history_price_coverage=breadth_coverage,
        sleep=lambda _seconds: None,
    )

    result = service.refresh(
        as_of_date=date(2026, 6, 4),
        market="IN",
        ensure_static_history=True,
    )

    assert fetch_calls == [
        {
            "symbols": ["SPARSE.NS", *IN_KEY_MARKET_PRICE_SYMBOLS],
            "period": STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD,
            "start_batch_size": 25,
            "market": "IN",
        }
    ]
    assert result["db_fresh_symbols"] == 1
    assert result["history_incomplete_symbols"] == 1
    assert result["rrg_history_coverage_status"] == "verified"
    assert result["breadth_history_coverage_status"] == "verified"
    assert breadth_coverage.calls == [
        {
            "market": "IN",
            "through_date": date(2026, 6, 4),
            "symbols": ("SPARSE.NS",),
        }
    ]


def test_static_daily_price_refresh_continues_when_rrg_calendar_lookup_fails() -> None:
    session_factory = _sqlite_session_factory()

    with session_factory() as db:
        db.add(
            StockUniverse(symbol="OLD.NS", market="IN", is_active=True, market_cap=100.0)
        )
        db.add(
            StockPrice(
                symbol="OLD.NS",
                date=date(2026, 6, 4),
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                volume=1000,
            )
        )
        db.commit()

    fetch_calls: list[dict] = []

    class _FailingCalendar:
        @staticmethod
        def trading_days(_market, _start, _end):
            raise RuntimeError("calendar unavailable")

    class _FakeFetcher:
        def fetch_prices_in_batches(
            self, symbols, period="2y", start_batch_size=None, market=None
        ):
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
        price_cache=SimpleNamespace(store_batch_in_cache=lambda *_args, **_kwargs: None),
        fetcher=_FakeFetcher(),
        batch_size_for_market=lambda _market: 25,
        calendar_service=_FailingCalendar(),
        sleep=lambda _seconds: None,
    )

    result = service.refresh(
        as_of_date=date(2026, 6, 4),
        market="IN",
        ensure_static_history=True,
    )

    assert fetch_calls == [
        {
            "symbols": IN_KEY_MARKET_PRICE_SYMBOLS,
            "period": STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD,
            "start_batch_size": 25,
            "market": "IN",
        }
    ]
    assert result["db_fresh_symbols"] == 1
    assert result["history_incomplete_symbols"] == 0
    assert result["rrg_history_coverage_status"] == "unverified"
    assert result["rrg_history_coverage_error"] == "calendar unavailable"
    assert result["yahoo_fetched_symbols"] == len(IN_KEY_MARKET_PRICE_SYMBOLS)


def test_static_daily_price_refresh_uses_date_only_freshness_for_key_market_symbols() -> None:
    session_factory = _sqlite_session_factory()

    with session_factory() as db:
        db.add_all(
            [
                StockPrice(
                    symbol="SPY",
                    date=date(2026, 6, 8),
                    open=1.0,
                    high=1.0,
                    low=1.0,
                    close=1.0,
                    volume=0,
                ),
                StockPrice(
                    symbol="SGD=X",
                    date=date(2026, 6, 8),
                    open=1.0,
                    high=1.0,
                    low=1.0,
                    close=1.0,
                    volume=None,
                ),
            ]
        )
        db.commit()

    fetch_calls: list[dict] = []

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
        price_cache=SimpleNamespace(store_batch_in_cache=lambda *_args, **_kwargs: None),
        fetcher=_FakeFetcher(),
        batch_size_for_market=lambda _market: 25,
        sleep=lambda _seconds: None,
    )

    result = service.refresh(as_of_date=date(2026, 6, 8), market="US")

    assert fetch_calls == [
        {
            "symbols": [
                symbol
                for symbol in US_KEY_MARKET_PRICE_SYMBOLS
                if symbol not in {"SPY", "SGD=X"}
            ],
            "period": STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD,
            "start_batch_size": 25,
            "market": "US",
        }
    ]
    assert result["db_fresh_symbols"] == 2
    assert result["stale_symbols"] == 0
    assert result["no_history_symbols"] == len(US_KEY_MARKET_PRICE_SYMBOLS) - 2


def test_static_daily_price_refresh_classifies_all_supported_symbols_with_volume_policy(
    monkeypatch,
) -> None:
    import app.services.static_daily_price_refresh_service as service_module
    from app.services.price_history_coverage import PriceHistoryCoverage

    session_factory = _sqlite_session_factory()

    with session_factory() as db:
        db.add(StockUniverse(symbol="GOOD", market="US", is_active=True, market_cap=100.0))
        db.commit()

    classify_calls: list[dict] = []
    fetch_calls: list[dict] = []

    def _fake_classify_price_history(
        db,
        *,
        symbols,
        as_of_date,
        require_positive_volume=False,
        symbols_requiring_positive_volume=(),
    ):
        classify_calls.append(
            {
                "symbols": list(symbols),
                "as_of_date": as_of_date,
                "require_positive_volume": require_positive_volume,
                "symbols_requiring_positive_volume": list(symbols_requiring_positive_volume),
            }
        )
        return PriceHistoryCoverage(no_history=tuple(symbols))

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

    monkeypatch.setattr(
        service_module,
        "classify_price_history",
        _fake_classify_price_history,
    )

    service = StaticDailyPriceRefreshService(
        session_factory=session_factory,
        price_cache=SimpleNamespace(store_batch_in_cache=lambda *_args, **_kwargs: None),
        fetcher=_FakeFetcher(),
        batch_size_for_market=lambda _market: 25,
        sleep=lambda _seconds: None,
    )

    service.refresh(as_of_date=date(2026, 6, 8), market="US")

    assert classify_calls == [
        {
            "symbols": ["GOOD", *US_KEY_MARKET_PRICE_SYMBOLS],
            "as_of_date": date(2026, 6, 8),
            "require_positive_volume": False,
            "symbols_requiring_positive_volume": ["GOOD"],
        }
    ]
    assert fetch_calls[0]["symbols"] == ["GOOD", *US_KEY_MARKET_PRICE_SYMBOLS]


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


_RRG_STARTUP_SEEDED_DATES = (
    date(2026, 6, 3),
    date(2026, 5, 28),
    date(2026, 3, 2),
    date(2026, 3, 1),
    date(2026, 2, 23),
    date(2026, 1, 30),
    date(2025, 12, 1),
    date(2025, 9, 1),
    date(2025, 6, 2),
    date(2025, 3, 3),
)


def _seed_rrg_startup_history(session_factory, symbols) -> None:
    """Seed every _RRGStartupCalendar anchor except the as-of session itself."""
    with session_factory() as db:
        for rank, symbol in enumerate(symbols):
            db.add(
                StockUniverse(
                    symbol=symbol, market="IN", is_active=True, market_cap=100.0 - rank
                )
            )
            db.add_all(
                StockPrice(
                    symbol=symbol,
                    date=seeded_date,
                    open=1.0,
                    high=1.0,
                    low=1.0,
                    close=1.0,
                    adj_close=1.0,
                    volume=1000,
                )
                for seeded_date in _RRG_STARTUP_SEEDED_DATES
            )
        db.commit()


def _price_frame(dates, adj_close: float):
    import pandas as pd

    size = len(dates)
    return pd.DataFrame(
        {
            "Open": [1.0] * size,
            "High": [1.0] * size,
            "Low": [1.0] * size,
            "Close": [adj_close] * size,
            "Adj Close": [adj_close] * size,
            "Volume": [1000] * size,
        },
        index=pd.to_datetime(list(dates)),
    )


def _top_up_frame(adj_close: float):
    return _price_frame([date(2026, 6, 3), date(2026, 6, 4)], adj_close)


def _persisting_store(session_factory, stored: list[list[str]]):
    """Store through the real StockPrice persistence (latest-row update) policy."""
    from app.services.price_row_normalization import stock_price_row_from_ohlcv
    from app.services.stock_price_persistence import persist_stock_price_mappings

    def store(payload, also_store_db=True, market=None):
        stored.append(sorted(payload))
        with session_factory() as db:
            persist_stock_price_mappings(
                db,
                {
                    symbol: [
                        stock_price_row_from_ohlcv(
                            symbol=symbol, row_date=stamp.date(), row=row
                        )
                        for stamp, row in frame.iterrows()
                    ]
                    for symbol, frame in payload.items()
                },
            )
            db.commit()

    return store


def _adj_closes(session_factory, symbol: str) -> set[float]:
    with session_factory() as db:
        return {
            adj_close
            for (adj_close,) in db.query(StockPrice.adj_close).filter(
                StockPrice.symbol == symbol
            )
        }


def _rrg_startup_service(session_factory, fetcher, stored: list[list[str]]):
    return StaticDailyPriceRefreshService(
        session_factory=session_factory,
        price_cache=SimpleNamespace(
            store_batch_in_cache=_persisting_store(session_factory, stored)
        ),
        fetcher=fetcher,
        batch_size_for_market=lambda _market: 25,
        calendar_service=_RRGStartupCalendar(),
        breadth_history_price_coverage=_CompleteBreadthHistoryCoverage(),
        sleep=lambda _seconds: None,
    )


def test_static_daily_price_refresh_tops_up_group_history_missing_only_the_current_session() -> None:
    session_factory = _sqlite_session_factory()
    _seed_rrg_startup_history(session_factory, ["OLD.NS"])
    fetch_calls: list[tuple[tuple[str, ...], str]] = []

    class _FakeFetcher:
        def fetch_prices_in_batches(self, symbols, period="2y", start_batch_size=None, market=None):
            fetch_calls.append((tuple(symbols), period))
            return {
                symbol: {"price_data": _top_up_frame(1.0), "has_error": False}
                for symbol in symbols
            }

    result = _rrg_startup_service(session_factory, _FakeFetcher(), []).refresh(
        as_of_date=date(2026, 6, 4),
        market="IN",
        ensure_static_history=True,
    )

    assert fetch_calls == [
        (("OLD.NS",), STATIC_DAILY_PRICE_REFRESH_PERIOD),
        (tuple(IN_KEY_MARKET_PRICE_SYMBOLS), STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD),
    ]
    assert result["stale_symbols"] == 1
    assert result["rrg_history_incomplete_symbols"] == 0


def test_static_daily_price_refresh_rebootstraps_symbols_whose_history_was_readjusted() -> None:
    session_factory = _sqlite_session_factory()
    _seed_rrg_startup_history(session_factory, ["OLD.NS", "SPLIT.NS"])
    fetch_calls: list[tuple[tuple[str, ...], str]] = []
    stored: list[list[str]] = []

    class _FakeFetcher:
        def fetch_prices_in_batches(self, symbols, period="2y", start_batch_size=None, market=None):
            fetch_calls.append((tuple(symbols), period))
            if period == STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD and symbols == ["SPLIT.NS"]:
                full_history = [*_RRG_STARTUP_SEEDED_DATES, date(2026, 6, 4)]
                return {"SPLIT.NS": {"price_data": _price_frame(full_history, 0.5), "has_error": False}}
            return {
                symbol: {
                    # A 2:1 split halves Yahoo's back-adjusted closes, including
                    # the 2026-06-03 bar the seed already stored at 1.0.
                    "price_data": _top_up_frame(0.5 if symbol == "SPLIT.NS" else 1.0),
                    "has_error": False,
                }
                for symbol in symbols
            }

    result = _rrg_startup_service(session_factory, _FakeFetcher(), stored).refresh(
        as_of_date=date(2026, 6, 4),
        market="IN",
        ensure_static_history=True,
    )

    assert fetch_calls == [
        (("OLD.NS", "SPLIT.NS"), STATIC_DAILY_PRICE_REFRESH_PERIOD),
        (tuple(IN_KEY_MARKET_PRICE_SYMBOLS), STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD),
        (("SPLIT.NS",), STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD),
    ]
    assert stored[0] == ["OLD.NS"]
    # The old-scale history is replaced, not just the latest row.
    assert _adj_closes(session_factory, "SPLIT.NS") == {0.5}
    assert _adj_closes(session_factory, "OLD.NS") == {1.0}
    assert result["readjusted_symbols"] == 1
    assert result["yahoo_fetched_symbols"] == 2 + len(IN_KEY_MARKET_PRICE_SYMBOLS)


def test_static_daily_price_refresh_tops_up_fresh_symbol_with_a_tail_anchor_gap() -> None:
    session_factory = _sqlite_session_factory()
    with session_factory() as db:
        db.add(StockUniverse(symbol="GAP.NS", market="IN", is_active=True, market_cap=100.0))
        # Current as of 2026-06-04, but missing the 2026-06-03 tail anchor.
        db.add_all(
            StockPrice(
                symbol="GAP.NS",
                date=row_date,
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                adj_close=1.0,
                volume=1000,
            )
            for row_date in (
                date(2026, 6, 4),
                *(d for d in _RRG_STARTUP_SEEDED_DATES if d != date(2026, 6, 3)),
            )
        )
        db.commit()
    fetch_calls: list[tuple[tuple[str, ...], str]] = []

    class _FakeFetcher:
        def fetch_prices_in_batches(self, symbols, period="2y", start_batch_size=None, market=None):
            fetch_calls.append((tuple(symbols), period))
            return {
                symbol: {"price_data": _top_up_frame(1.0), "has_error": False}
                for symbol in symbols
            }

    result = _rrg_startup_service(session_factory, _FakeFetcher(), []).refresh(
        as_of_date=date(2026, 6, 4),
        market="IN",
        ensure_static_history=True,
    )

    assert fetch_calls[0] == (("GAP.NS",), STATIC_DAILY_PRICE_REFRESH_PERIOD)
    assert result["db_fresh_symbols"] == 1
    assert result["rrg_history_tail_gap_symbols"] == 1


def test_static_daily_price_refresh_rebootstraps_readjusted_symbols_recovered_by_rate_limit_retry() -> None:
    session_factory = _sqlite_session_factory()
    seeded_dates = (date(2026, 6, 2), date(2026, 6, 3))
    with session_factory() as db:
        db.add(StockUniverse(symbol="SPLIT.NS", market="IN", is_active=True, market_cap=100.0))
        db.add_all(
            StockPrice(
                symbol="SPLIT.NS",
                date=seeded_date,
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                adj_close=1.0,
                volume=1000,
            )
            for seeded_date in seeded_dates
        )
        db.commit()
    fetch_calls: list[tuple[tuple[str, ...], str]] = []
    stored: list[list[str]] = []

    class _FakeFetcher:
        def fetch_prices_in_batches(self, symbols, period="2y", start_batch_size=None, market=None):
            fetch_calls.append((tuple(symbols), period))
            if len(fetch_calls) == 1:
                return {
                    "SPLIT.NS": {
                        "price_data": None,
                        "has_error": True,
                        "error": "Too Many Requests (429)",
                    }
                }
            if symbols == ["SPLIT.NS"] and period == STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD:
                full_history = [*seeded_dates, date(2026, 6, 4)]
                return {"SPLIT.NS": {"price_data": _price_frame(full_history, 0.5), "has_error": False}}
            return {
                symbol: {"price_data": _top_up_frame(0.5), "has_error": False}
                for symbol in symbols
            }

    service = StaticDailyPriceRefreshService(
        session_factory=session_factory,
        price_cache=SimpleNamespace(
            store_batch_in_cache=_persisting_store(session_factory, stored)
        ),
        fetcher=_FakeFetcher(),
        batch_size_for_market=lambda _market: 25,
        sleep=lambda _seconds: None,
    )

    result = service.refresh(as_of_date=date(2026, 6, 4), market="IN")

    assert fetch_calls[-2:] == [
        (("SPLIT.NS",), STATIC_DAILY_PRICE_REFRESH_PERIOD),
        (("SPLIT.NS",), STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD),
    ]
    assert _adj_closes(session_factory, "SPLIT.NS") == {0.5}
    assert result["readjusted_symbols"] == 1
    # The throttled first attempt is not left counted as a failure.
    assert result["yahoo_failed_symbols"] == 0
    assert result["yahoo_fetched_symbols"] == 1 + len(IN_KEY_MARKET_PRICE_SYMBOLS)


def _run_readjusted_split(session_factory, split_history):
    _seed_rrg_startup_history(session_factory, ["SPLIT.NS"])

    class _FakeFetcher:
        def fetch_prices_in_batches(self, symbols, period="2y", start_batch_size=None, market=None):
            if period == STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD and symbols == ["SPLIT.NS"]:
                return {"SPLIT.NS": {"price_data": _price_frame(split_history, 0.5), "has_error": False}}
            return {
                symbol: {"price_data": _top_up_frame(0.5), "has_error": False}
                for symbol in symbols
            }

    return _rrg_startup_service(session_factory, _FakeFetcher(), []).refresh(
        as_of_date=date(2026, 6, 4),
        market="IN",
        ensure_static_history=True,
    )


def test_static_daily_price_refresh_keeps_old_history_when_replacement_write_fails(monkeypatch) -> None:
    import app.services.static_daily_price_refresh_service as module

    def _failing_persist(*_args, **_kwargs):
        raise RuntimeError("database write failed")

    monkeypatch.setattr(module, "persist_stock_price_mappings", _failing_persist)
    session_factory = _sqlite_session_factory()

    result = _run_readjusted_split(
        session_factory, [*_RRG_STARTUP_SEEDED_DATES, date(2026, 6, 4)]
    )

    # The delete rolled back with the failed insert, and the symbol is a failure.
    assert _adj_closes(session_factory, "SPLIT.NS") == {1.0}
    assert result["yahoo_failed_symbols"] == 1


def test_static_daily_price_refresh_rejects_truncated_replacement_history() -> None:
    session_factory = _sqlite_session_factory()

    # A truncated 2y response that stops before the stored 2026-06-03 row.
    truncated = [d for d in _RRG_STARTUP_SEEDED_DATES if d < date(2026, 6, 3)]
    result = _run_readjusted_split(session_factory, truncated)

    with session_factory() as db:
        newest = (
            db.query(StockPrice.adj_close)
            .filter(StockPrice.symbol == "SPLIT.NS", StockPrice.date == date(2026, 6, 3))
            .scalar()
        )
    # Rejected whole: the newer row survives and no mixed-scale series is left.
    assert newest == 1.0
    assert _adj_closes(session_factory, "SPLIT.NS") == {1.0}
    assert result["yahoo_failed_symbols"] == 1


def test_static_daily_price_refresh_rejects_sparse_replacement_history() -> None:
    session_factory = _sqlite_session_factory()

    # Yahoo's 2y response is missing a stored interior date (2025-12-01).
    sparse = [
        *(d for d in _RRG_STARTUP_SEEDED_DATES if d != date(2025, 12, 1)),
        date(2026, 6, 4),
    ]
    result = _run_readjusted_split(session_factory, sparse)

    # Replacing would leave a gap or an old-scale row, so nothing changes.
    assert _adj_closes(session_factory, "SPLIT.NS") == {1.0}
    with session_factory() as db:
        assert db.query(StockPrice).filter(StockPrice.symbol == "SPLIT.NS").count() == len(
            _RRG_STARTUP_SEEDED_DATES
        )
    assert result["yahoo_failed_symbols"] == 1


def test_static_daily_price_refresh_rejects_replacement_missing_the_discarded_top_up_bar() -> None:
    session_factory = _sqlite_session_factory()

    # The drift-triggering 7d frame carried 2026-06-04; the 2y refetch stops at
    # 2026-06-03, the previous stored latest date.
    result = _run_readjusted_split(session_factory, list(_RRG_STARTUP_SEEDED_DATES))

    assert _adj_closes(session_factory, "SPLIT.NS") == {1.0}
    assert result["yahoo_failed_symbols"] == 1


def test_static_daily_price_batch_line_counts_repaired_and_missing_sessions(capsys) -> None:
    as_of = date(2026, 6, 4)
    current = _top_up_frame(1.0)
    stale = _price_frame([date(2026, 6, 3)], 1.0)

    class _FakeFetcher:
        def fetch_prices_in_batches(self, symbols, period="2y", start_batch_size=None, market=None):
            return {
                "FULL": {"price_data": current, "has_error": False},
                "QUOTED": {"price_data": current, "has_error": False, "repaired_by": "yahoo_quote"},
                "BEHIND": {"price_data": stale, "has_error": False},
            }

    service = StaticDailyPriceRefreshService(
        session_factory=_sqlite_session_factory(),
        price_cache=SimpleNamespace(store_batch_in_cache=lambda *args, **kwargs: None),
        fetcher=_FakeFetcher(),
    )

    service._fetch_and_store(
        ["FULL", "QUOTED", "BEHIND"],
        period=STATIC_DAILY_PRICE_REFRESH_PERIOD,
        batch_size=25,
        market="US",
        as_of_date=as_of,
    )

    assert (
        "Batch 1/1 complete: 3/3 processed, 3 refreshed, 0 failed, "
        "1 repaired, 1 missing 2026-06-04."
    ) in capsys.readouterr().out


def test_static_daily_price_fetch_keeps_frames_still_missing_the_session() -> None:
    as_of = date(2026, 6, 4)
    current = _top_up_frame(1.0)
    stale = _price_frame([date(2026, 6, 3)], 1.0)

    class _FakeFetcher:
        def fetch_prices_in_batches(self, symbols, period="2y", start_batch_size=None, market=None):
            return {
                "FULL": {"price_data": current, "has_error": False},
                "BEHIND": {"price_data": stale, "has_error": False},
            }

    service = StaticDailyPriceRefreshService(
        session_factory=_sqlite_session_factory(),
        price_cache=SimpleNamespace(store_batch_in_cache=lambda *args, **kwargs: None),
        fetcher=_FakeFetcher(),
    )
    # A later pass that stored FULL with the session drops it from the map.
    missing = {"FULL": stale}

    service._fetch_and_store(
        ["FULL", "BEHIND"],
        period=STATIC_DAILY_PRICE_REFRESH_PERIOD,
        batch_size=25,
        market="US",
        as_of_date=as_of,
        missing_session_frames=missing,
    )

    assert list(missing) == ["BEHIND"]


def _closing_quote(symbol: str, session_close_utc: datetime) -> dict:
    return {
        "symbol": symbol,
        "marketState": "CLOSED",
        "regularMarketTime": int(session_close_utc.timestamp()),
        "regularMarketOpen": 1.0,
        "regularMarketDayHigh": 1.2,
        "regularMarketDayLow": 0.9,
        "regularMarketPrice": 1.1,
        "regularMarketVolume": 500,
    }


def test_static_daily_price_repairs_missing_sessions_from_quotes_after_a_wait() -> None:
    as_of = date(2026, 6, 4)
    stored: list[dict] = []
    sleeps: list[float] = []
    service = StaticDailyPriceRefreshService(
        session_factory=_sqlite_session_factory(),
        price_cache=SimpleNamespace(
            store_batch_in_cache=lambda frames, **kwargs: stored.append(dict(frames))
        ),
        fetcher=SimpleNamespace(),
        sleep=sleeps.append,
        # 16:00 ET close of the as-of session.
        fetch_quotes=lambda symbols: [
            _closing_quote(symbol, datetime(2026, 6, 4, 20, 0, tzinfo=timezone.utc))
            for symbol in symbols
        ],
    )

    stats = service._repair_missing_sessions(
        market="US",
        as_of_date=as_of,
        frames={"BEHIND": _price_frame([date(2026, 6, 3)], 1.0)},
    )

    assert sleeps == [STATIC_SESSION_REPAIR_WAIT_SECONDS]
    assert stats == {
        "attempted": 1,
        "repaired": 1,
        "wait_seconds": STATIC_SESSION_REPAIR_WAIT_SECONDS,
    }
    [frames] = stored
    assert frames["BEHIND"].index[-1].date() == as_of
    assert float(frames["BEHIND"]["Close"].iloc[-1]) == 1.1


def test_static_daily_price_skips_session_repair_without_a_quote_plan() -> None:
    sleeps: list[float] = []
    service = StaticDailyPriceRefreshService(
        session_factory=_sqlite_session_factory(),
        price_cache=SimpleNamespace(store_batch_in_cache=lambda *args, **kwargs: None),
        fetcher=SimpleNamespace(),
        sleep=sleeps.append,
        fetch_quotes=lambda symbols: pytest.fail("HK prices have no quote repair"),
    )

    stats = service._repair_missing_sessions(
        market="HK",
        as_of_date=date(2026, 6, 4),
        frames={"0700.HK": _price_frame([date(2026, 6, 3)], 1.0)},
    )

    assert sleeps == []
    assert stats == {"attempted": 0, "repaired": 0, "wait_seconds": 0}
