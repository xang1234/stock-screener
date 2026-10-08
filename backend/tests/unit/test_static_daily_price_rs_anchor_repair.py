"""#539: canonical RS anchor gaps are repaired even without group rankings."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.stock import StockPrice
from app.models.stock_universe import StockUniverse
from app.services.price_row_normalization import stock_price_row_from_ohlcv
from app.services.rs_anchor_price_coverage import RsAnchorPriceCoverageService
from app.services.static_daily_price_refresh_service import (
    STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD,
    StaticDailyPriceRefreshService,
)
from app.services.stock_price_persistence import persist_stock_price_mappings

AS_OF = date(2026, 10, 7)
SESSIONS: list[date] = []
_day = AS_OF
while len(SESSIONS) < 320:
    if _day.weekday() < 5:
        SESSIONS.append(_day)
    _day -= timedelta(days=1)
SESSIONS.reverse()
# The 21-session anchor is SESSIONS[-22]; this gap spans it and the next four
# sessions, like AU's 2026-09-07..23 hole.
GAP = set(SESSIONS[-22:-17])


class _WeekdayCalendar:
    @staticmethod
    def session_anchors(market, as_of_date, *, offsets):
        index = SESSIONS.index(as_of_date)
        return {0: as_of_date, **{offset: SESSIONS[index - offset] for offset in offsets}}

    @staticmethod
    def market_timezone(market):
        return "Australia/Sydney"


class _CompleteBreadth:
    @staticmethod
    def classify(db, *, market, through_date, symbols):
        return SimpleNamespace(
            incomplete_symbols=(),
            history_incomplete_symbols=(),
            missing_through_date_symbols=(),
            required_price_date_count=0,
        )


def _session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[StockUniverse.__table__, StockPrice.__table__])
    return sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)


def _frame(dates, adj_close):
    dates = sorted(dates)
    return pd.DataFrame(
        {
            "Open": [adj_close] * len(dates),
            "High": [adj_close] * len(dates),
            "Low": [adj_close] * len(dates),
            "Close": [adj_close] * len(dates),
            "Adj Close": [adj_close] * len(dates),
            "Volume": [1000] * len(dates),
        },
        index=pd.to_datetime(dates),
    )


def _seed(session_factory, histories):
    with session_factory() as db:
        for rank, (symbol, dates) in enumerate(histories.items()):
            db.add(StockUniverse(symbol=symbol, market="AU", is_active=True, market_cap=100.0 - rank))
            db.add_all(
                StockPrice(symbol=symbol, date=day, open=1.0, high=1.0, low=1.0,
                           close=1.0, adj_close=1.0, volume=1000)
                for day in dates
            )
        db.commit()


def _store(session_factory):
    def store(payload, also_store_db=True, market=None):
        with session_factory() as db:
            persist_stock_price_mappings(
                db,
                {
                    symbol: [
                        stock_price_row_from_ohlcv(symbol=symbol, row_date=stamp.date(), row=row)
                        for stamp, row in frame.iterrows()
                    ]
                    for symbol, frame in payload.items()
                },
            )
            db.commit()

    return store


class _Fetcher:
    def __init__(self, history_dates):
        self.history_dates = history_dates
        self.calls: list[tuple[tuple[str, ...], str]] = []

    def fetch_prices_in_batches(self, symbols, period="2y", start_batch_size=None, market=None):
        self.calls.append((tuple(symbols), period))
        dates = self.history_dates if period == STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD else SESSIONS[-5:]
        # A new adjustment basis (2.0): repaired history must not splice bases.
        return {symbol: {"price_data": _frame(dates, 2.0), "has_error": False} for symbol in symbols}


def _service(session_factory, fetcher):
    calendar = _WeekdayCalendar()
    return StaticDailyPriceRefreshService(
        session_factory=session_factory,
        price_cache=SimpleNamespace(store_batch_in_cache=_store(session_factory)),
        fetcher=fetcher,
        batch_size_for_market=lambda _market: 50,
        calendar_service=calendar,
        breadth_history_price_coverage=_CompleteBreadth(),
        rs_anchor_price_coverage=RsAnchorPriceCoverageService(calendar_service=calendar),
        sleep=lambda _seconds: None,
    )


def _stored(session_factory, symbol):
    with session_factory() as db:
        return dict(
            db.query(StockPrice.date, StockPrice.adj_close).filter(StockPrice.symbol == symbol).all()
        )


def _repair_calls(fetcher):
    return [symbols for symbols, period in fetcher.calls
            if period == STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD and "GAP.AX" in symbols]


def test_fresh_long_history_with_an_interior_anchor_gap_is_repaired_on_one_basis():
    session_factory = _session_factory()
    _seed(session_factory, {
        "GAP.AX": [day for day in SESSIONS if day not in GAP],
        "OK.AX": list(SESSIONS),
    })
    fetcher = _Fetcher(SESSIONS)

    result = _service(session_factory, fetcher).refresh(
        as_of_date=AS_OF, market="AU", ensure_static_history=True
    )

    assert _repair_calls(fetcher) == [("GAP.AX",)]
    repair = result["rs_anchor_repair"]
    assert repair["status"] == "verified"
    assert repair["gap_symbols"] == 1
    # The whole multi-session gap is reported, not only the current anchor.
    assert repair["gap_count_by_date"] == {day.isoformat(): 1 for day in sorted(GAP)}
    assert repair["repaired_symbols"] == 1
    assert repair["unresolved_symbols"] == 0
    stored = _stored(session_factory, "GAP.AX")
    assert GAP <= set(stored)
    assert set(stored.values()) == {2.0}
    assert set(_stored(session_factory, "OK.AX").values()) <= {1.0, 2.0}
    assert set(_stored(session_factory, "OK.AX")) == set(SESSIONS)


def test_provider_history_missing_the_anchor_stays_unresolved_and_untouched():
    session_factory = _session_factory()
    history = [day for day in SESSIONS if day not in GAP]
    _seed(session_factory, {"GAP.AX": history})
    fetcher = _Fetcher(history)  # the provider has the same hole

    result = _service(session_factory, fetcher).refresh(
        as_of_date=AS_OF, market="AU", ensure_static_history=True
    )

    repair = result["rs_anchor_repair"]
    assert repair["repaired_symbols"] == 0
    assert repair["unresolved_symbols"] == 1
    assert repair["unresolved_count_by_date"] == {day.isoformat(): 1 for day in sorted(GAP)}
    assert repair["unresolved_samples"] == ["GAP.AX"]
    assert GAP.isdisjoint(_stored(session_factory, "GAP.AX"))


def test_short_history_and_complete_symbols_are_not_scheduled_and_a_rerun_is_idempotent():
    session_factory = _session_factory()
    _seed(session_factory, {
        "NEW.AX": SESSIONS[-15:],  # listed after the older anchors
        "OK.AX": list(SESSIONS),
    })
    fetcher = _Fetcher(SESSIONS)
    service = _service(session_factory, fetcher)

    first = service.refresh(as_of_date=AS_OF, market="AU", ensure_static_history=True)
    second = service.refresh(as_of_date=AS_OF, market="AU", ensure_static_history=True)

    for result in (first, second):
        assert result["rs_anchor_repair"]["gap_symbols"] == 0
    assert not any(
        {"NEW.AX", "OK.AX"} & set(symbols)
        for symbols, period in fetcher.calls
        if period == STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD
    )


def test_required_dates_cover_the_lookahead_rollover_from_the_calendar():
    service = RsAnchorPriceCoverageService(calendar_service=_WeekdayCalendar())

    current_only = service.required_dates(market="AU", through_date=AS_OF, lookahead_sessions=0)
    with_lookahead = service.required_dates(market="AU", through_date=AS_OF, lookahead_sessions=3)

    assert SESSIONS[-22] in current_only  # today's 21-session anchor
    assert SESSIONS[-19] not in current_only
    # Within three sessions the 21-session anchor rolls forward to SESSIONS[-19].
    assert {SESSIONS[-22], SESSIONS[-21], SESSIONS[-20], SESSIONS[-19]} <= with_lookahead
    assert AS_OF not in with_lookahead


def test_provider_errors_leave_the_gap_unresolved_and_history_unchanged():
    session_factory = _session_factory()
    history = [day for day in SESSIONS if day not in GAP]
    _seed(session_factory, {"GAP.AX": history})

    class _FailingRepairFetcher(_Fetcher):
        def fetch_prices_in_batches(self, symbols, period="2y", start_batch_size=None, market=None):
            if period == STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD and "GAP.AX" in symbols:
                self.calls.append((tuple(symbols), period))
                return {"GAP.AX": {"price_data": None, "has_error": True, "error": "timeout"}}
            return super().fetch_prices_in_batches(symbols, period, start_batch_size, market)

    result = _service(session_factory, _FailingRepairFetcher(SESSIONS)).refresh(
        as_of_date=AS_OF, market="AU", ensure_static_history=True
    )

    assert result["rs_anchor_repair"]["unresolved_symbols"] == 1
    assert set(_stored(session_factory, "GAP.AX")) == set(history)


def test_dormant_symbols_are_not_refetched_for_holes_after_their_history_ends():
    session_factory = _session_factory()
    _seed(session_factory, {"DORMANT.AX": SESSIONS[:-40]})  # stopped trading 40 sessions ago
    fetcher = _Fetcher(SESSIONS)

    result = _service(session_factory, fetcher).refresh(
        as_of_date=AS_OF, market="AU", ensure_static_history=True
    )

    assert result["rs_anchor_repair"]["gap_symbols"] == 0


def test_benchmark_anchor_holes_are_repaired_too(monkeypatch):
    from app.services import static_daily_price_refresh_service as module

    session_factory = _session_factory()
    _seed(session_factory, {"OK.AX": list(SESSIONS)})
    with session_factory() as db:
        db.add_all(
            StockPrice(symbol="^AXJO", date=day, open=1.0, high=1.0, low=1.0,
                       close=1.0, adj_close=1.0, volume=1000)
            for day in SESSIONS if day not in GAP
        )
        db.commit()
    monkeypatch.setattr(module, "_key_market_price_symbols", lambda market: ["^AXJO"])
    fetcher = _Fetcher(SESSIONS)

    result = _service(session_factory, fetcher).refresh(
        as_of_date=AS_OF, market="AU", ensure_static_history=True
    )

    assert result["rs_anchor_repair"]["repaired_symbols"] == 1
    assert GAP <= set(_stored(session_factory, "^AXJO"))


def test_rate_limited_repairs_are_retried_once():
    session_factory = _session_factory()
    _seed(session_factory, {"GAP.AX": [day for day in SESSIONS if day not in GAP]})

    class _ThrottledOnce(_Fetcher):
        throttled = False

        def fetch_prices_in_batches(self, symbols, period="2y", start_batch_size=None, market=None):
            if period == STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD and "GAP.AX" in symbols and not self.throttled:
                self.throttled = True
                self.calls.append((tuple(symbols), period))
                return {"GAP.AX": {"price_data": None, "has_error": True, "error": "429 Too Many Requests"}}
            return super().fetch_prices_in_batches(symbols, period, start_batch_size, market)

    fetcher = _ThrottledOnce(SESSIONS)
    result = _service(session_factory, fetcher).refresh(
        as_of_date=AS_OF, market="AU", ensure_static_history=True
    )

    assert _repair_calls(fetcher) == [("GAP.AX",), ("GAP.AX",)]
    assert result["rs_anchor_repair"]["unresolved_symbols"] == 0


def test_real_au_calendar_rolls_the_21_session_anchor_onto_september_7():
    from app.services.market_calendar_service import MarketCalendarService

    calendar = MarketCalendarService()
    service = RsAnchorPriceCoverageService(calendar_service=calendar)
    previous = calendar.trading_days("AU", date(2026, 9, 28), date(2026, 10, 5))[-1]

    on_oct_6 = service.required_dates(market="AU", through_date=date(2026, 10, 6), lookahead_sessions=0)
    before = service.required_dates(market="AU", through_date=previous, lookahead_sessions=0)
    ahead = service.required_dates(market="AU", through_date=previous, lookahead_sessions=10)

    assert date(2026, 9, 7) in on_oct_6 and date(2026, 9, 4) not in on_oct_6
    assert date(2026, 9, 4) in before and date(2026, 9, 7) not in before
    # The lookahead reaches the hole before the anchor does.
    assert date(2026, 9, 7) in ahead


def test_truncated_refetch_ending_before_stored_rows_is_not_spliced():
    session_factory = _session_factory()
    history = [day for day in SESSIONS if day not in GAP]
    _seed(session_factory, {"GAP.AX": history})
    # Covers the gap but stops 3 sessions before the stored history ends.
    fetcher = _Fetcher(SESSIONS[:-3])

    result = _service(session_factory, fetcher).refresh(
        as_of_date=AS_OF, market="AU", ensure_static_history=True
    )

    assert result["rs_anchor_repair"]["unresolved_symbols"] == 1
    assert GAP.isdisjoint(_stored(session_factory, "GAP.AX"))


def test_a_gap_symbol_whose_bootstrap_failed_is_still_repaired():
    class _BootstrapGapBreadth(_CompleteBreadth):
        @staticmethod
        def classify(db, *, market, through_date, symbols):
            return SimpleNamespace(
                incomplete_symbols=("GAP.AX",),
                history_incomplete_symbols=("GAP.AX",),
                missing_through_date_symbols=(),
                required_price_date_count=0,
            )

    session_factory = _session_factory()
    _seed(session_factory, {"GAP.AX": [day for day in SESSIONS if day not in GAP]})

    class _BootstrapThrottled(_Fetcher):
        bootstrap_seen = False

        def fetch_prices_in_batches(self, symbols, period="2y", start_batch_size=None, market=None):
            if period == STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD and "GAP.AX" in symbols and not self.bootstrap_seen:
                self.bootstrap_seen = True
                self.calls.append((tuple(symbols), period))
                return {s: {"price_data": None, "has_error": True, "error": "429"} for s in symbols}
            return super().fetch_prices_in_batches(symbols, period, start_batch_size, market)

    fetcher = _BootstrapThrottled(SESSIONS)
    calendar = _WeekdayCalendar()
    service = StaticDailyPriceRefreshService(
        session_factory=session_factory,
        price_cache=SimpleNamespace(store_batch_in_cache=_store(session_factory)),
        fetcher=fetcher,
        batch_size_for_market=lambda _market: 50,
        calendar_service=calendar,
        breadth_history_price_coverage=_BootstrapGapBreadth(),
        rs_anchor_price_coverage=RsAnchorPriceCoverageService(calendar_service=calendar),
        sleep=lambda _seconds: None,
    )

    result = service.refresh(as_of_date=AS_OF, market="AU", ensure_static_history=True)

    assert result["rs_anchor_repair"]["unresolved_symbols"] == 0
    assert GAP <= set(_stored(session_factory, "GAP.AX"))
