from datetime import date, datetime
from uuid import uuid4
from unittest.mock import Mock

import pandas as pd
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.industry import IBDGroupRank
from app.models.stock_universe import StockUniverse
from app.services.ibd_group_rank_service import (
    IBDGroupRankService,
    IncompleteGroupRankingCacheError,
    MissingIBDIndustryMappingsError,
)
from app.services import ibd_group_rank_service as group_rank_module


def _add_rank(session, group, rank_date, rank):
    session.add(
        IBDGroupRank(
            industry_group=group,
            date=rank_date,
            rank=rank,
            avg_rs_rating=50.0,
            num_stocks=10,
            num_stocks_rs_above_80=2,
            top_symbol="TEST",
            top_rs_rating=90.0,
        )
    )
    session.flush()


def _make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        engine,
        tables=[IBDGroupRank.__table__, StockUniverse.__table__],
    )
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)()


def _make_group_rank_service(price_cache: Mock | None = None, benchmark_cache: Mock | None = None):
    return IBDGroupRankService(
        price_cache=price_cache or Mock(),
        benchmark_cache=benchmark_cache or Mock(),
    )


def test_find_missing_dates_uses_market_calendar(db_session, monkeypatch):
    service = _make_group_rank_service()

    class _FakeCalendarService:
        def market_now(self, market):
            assert market == "HK"
            return datetime(2026, 3, 20, 9, 0, 0)

        def is_trading_day(self, market, day):
            assert market == "HK"
            return day == date(2026, 3, 19)

    monkeypatch.setattr(
        "app.wiring.bootstrap.get_market_calendar_service",
        lambda: _FakeCalendarService(),
    )

    missing = service.find_missing_dates(db_session, lookback_days=2, market="HK")

    assert missing == [date(2026, 3, 19)]


def test_get_historical_rank_picks_closest_date():
    service = _make_group_rank_service()
    db_session = _make_session()
    group = f"TEST_GROUP_UNIT_{uuid4().hex}"
    current_date = date(2024, 1, 22)

    try:
        # target_date = current_date - 7 days = Jan 15
        # Record 2 days before target (Jan 13, rank 10) — closer
        # Record 3 days after target (Jan 18, rank 20) — further
        _add_rank(db_session, group, date(2024, 1, 13), 10)
        _add_rank(db_session, group, date(2024, 1, 18), 20)

        result = service._get_historical_ranks_batch(
            db_session, [group], current_date, {'1w': 7}
        )
        assert result[(group, '1w')] == 10
    finally:
        db_session.rollback()
        db_session.close()


def test_get_historical_rank_prefers_earlier_on_tie():
    service = _make_group_rank_service()
    db_session = _make_session()
    group = f"TEST_GROUP_UNIT_{uuid4().hex}"
    current_date = date(2024, 1, 22)

    try:
        # target_date = current_date - 7 days = Jan 15
        # Record 2 days before target (Jan 13, rank 11) — equidistant, earlier
        # Record 2 days after target (Jan 17, rank 99) — equidistant, later
        _add_rank(db_session, group, date(2024, 1, 13), 11)
        _add_rank(db_session, group, date(2024, 1, 17), 99)

        result = service._get_historical_ranks_batch(
            db_session, [group], current_date, {'1w': 7}
        )
        assert result[(group, '1w')] == 11
    finally:
        db_session.rollback()
        db_session.close()


def test_get_group_history_uses_universe_lookup_for_top_symbol_name(monkeypatch):
    service = _make_group_rank_service()
    db_session = _make_session()
    group = f"TEST_GROUP_UNIT_{uuid4().hex}"
    current_date = date.today()

    try:
        db_session.add(
            IBDGroupRank(
                industry_group=group,
                date=current_date,
                rank=1,
                avg_rs_rating=92.0,
                median_rs_rating=91.0,
                weighted_avg_rs_rating=93.0,
                rs_std_dev=1.5,
                num_stocks=8,
                num_stocks_rs_above_80=6,
                top_symbol="AAPL",
                top_rs_rating=98.0,
            )
        )
        db_session.add(
            StockUniverse(
                symbol="AAPL",
                name="Apple Inc.",
                market="US",
                exchange="NASDAQ",
                is_active=True,
                status="active",
                status_reason="active",
            )
        )
        db_session.commit()

        monkeypatch.setattr(
            service,
            "_get_historical_ranks_batch",
            lambda *_args, **_kwargs: {},
        )
        monkeypatch.setattr(
            service,
            "_get_constituent_stocks",
            lambda *_args, **_kwargs: [{"symbol": "MSFT", "company_name": "Microsoft"}],
        )

        result = service.get_group_history(db_session, group, days=30)

        assert result["top_symbol"] == "AAPL"
        assert result["top_symbol_name"] == "Apple Inc."
    finally:
        db_session.rollback()
        db_session.close()


def _price_frame() -> pd.DataFrame:
    dates = pd.date_range(end="2026-03-20", periods=260, freq="B")
    return pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Volume": 1_000_000,
        },
        index=dates,
    )


def _trend_price_frame(
    *,
    end: str = "2026-03-20",
    periods: int = 260,
    start_close: float = 100.0,
    daily_return: float = 0.001,
) -> pd.DataFrame:
    dates = pd.date_range(end=end, periods=periods, freq="B")
    closes = [start_close * ((1 + daily_return) ** idx) for idx in range(periods)]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": 1_000_000,
        },
        index=dates,
    )


def test_prefetch_all_data_uses_cached_only_prices_for_same_day(db_session, monkeypatch):
    service = _make_group_rank_service()
    spy_data = _price_frame()
    aapl_data = _price_frame()

    price_cache = Mock()
    price_cache.get_cached_only_fresh.return_value = spy_data
    price_cache.get_many_cached_only_fresh.return_value = {"AAPL": aapl_data}
    price_cache.get_many.side_effect = AssertionError("fetch-capable path should not be used")
    service.price_cache = price_cache

    benchmark_cache = Mock()
    benchmark_cache.get_benchmark_symbol.return_value = "SPY"
    benchmark_cache.get_benchmark_data.side_effect = AssertionError("benchmark fetch should not be used")
    service.benchmark_cache = benchmark_cache

    stock_universe_service = Mock()
    stock_universe_service.get_active_symbols.return_value = ["AAPL"]
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_stock_universe_service",
        lambda: stock_universe_service,
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db, **kw: ["Software"],
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_group_symbols",
        lambda db, group, **kw: ["AAPL"],
    )
    monkeypatch.setattr(
        service,
        "_get_market_caps_for_symbols",
        lambda db, symbols: {"AAPL": 1_000_000_000},
    )

    prefetch = service._prefetch_all_data(
        db_session,
        cache_only=True,
    )

    assert prefetch.benchmark_prices is spy_data
    assert prefetch.prices_by_symbol == {"AAPL": aapl_data}
    assert prefetch.active_symbols == {"AAPL"}
    assert prefetch.market_caps == {"AAPL": 1_000_000_000}
    assert prefetch.symbols_by_group == {"Software": ["AAPL"]}
    assert prefetch.stats == {
        "target_symbols": 1,
        "symbols_with_prices": 1,
        "cache_miss_symbols": 0,
        "spy_cached": True,
        "skipped_unsupported_symbols": 0,
    }
    price_cache.get_cached_only_fresh.assert_called_once_with("SPY", period="2y")
    price_cache.get_many_cached_only_fresh.assert_called_once_with(["AAPL"], period="2y")


def test_prefetch_all_data_treats_stale_same_day_cache_as_missing(db_session, monkeypatch):
    service = _make_group_rank_service()
    spy_data = _price_frame()

    price_cache = Mock()
    price_cache.get_cached_only_fresh.return_value = spy_data
    price_cache.get_many_cached_only_fresh.return_value = {"AAPL": None}
    price_cache.get_many.side_effect = AssertionError("fetch-capable path should not be used")
    service.price_cache = price_cache

    benchmark_cache = Mock()
    benchmark_cache.get_benchmark_symbol.return_value = "SPY"
    benchmark_cache.get_benchmark_data.side_effect = AssertionError("benchmark fetch should not be used")
    service.benchmark_cache = benchmark_cache

    stock_universe_service = Mock()
    stock_universe_service.get_active_symbols.return_value = ["AAPL"]
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_stock_universe_service",
        lambda: stock_universe_service,
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db, **kw: ["Software"],
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_group_symbols",
        lambda db, group, **kw: ["AAPL"],
    )
    monkeypatch.setattr(
        service,
        "_get_market_caps_for_symbols",
        lambda db, symbols: {"AAPL": 1_000_000_000},
    )

    prefetch = service._prefetch_all_data(
        db_session,
        cache_only=True,
    )

    assert prefetch.benchmark_prices is spy_data
    assert prefetch.prices_by_symbol == {"AAPL": None}
    assert prefetch.active_symbols == {"AAPL"}
    assert prefetch.market_caps == {"AAPL": 1_000_000_000}
    assert prefetch.symbols_by_group == {"Software": ["AAPL"]}
    assert prefetch.stats == {
        "target_symbols": 1,
        "symbols_with_prices": 0,
        "cache_miss_symbols": 1,
        "spy_cached": True,
        "skipped_unsupported_symbols": 0,
    }
    price_cache.get_many_cached_only_fresh.assert_called_once_with(["AAPL"], period="2y")


def test_prefetch_all_data_uses_fetch_capable_prices_for_historical(db_session, monkeypatch):
    service = _make_group_rank_service()
    spy_data = _price_frame()
    aapl_data = _price_frame()

    price_cache = Mock()
    price_cache.get_many.return_value = {"AAPL": aapl_data}
    price_cache.get_many_cached_only.side_effect = AssertionError("cache-only path should not be used")
    service.price_cache = price_cache

    benchmark_cache = Mock()
    benchmark_cache.get_benchmark_symbol.return_value = "SPY"
    benchmark_cache.get_benchmark_data.return_value = spy_data
    service.benchmark_cache = benchmark_cache

    stock_universe_service = Mock()
    stock_universe_service.get_active_symbols.return_value = ["AAPL"]
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_stock_universe_service",
        lambda: stock_universe_service,
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db, **kw: ["Software"],
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_group_symbols",
        lambda db, group, **kw: ["AAPL"],
    )
    monkeypatch.setattr(
        service,
        "_get_market_caps_for_symbols",
        lambda db, symbols: {"AAPL": 1_000_000_000},
    )

    prefetch = service._prefetch_all_data(
        db_session,
        cache_only=False,
    )

    assert prefetch.benchmark_prices is spy_data
    assert prefetch.prices_by_symbol == {"AAPL": aapl_data}
    assert prefetch.active_symbols == {"AAPL"}
    assert prefetch.market_caps == {"AAPL": 1_000_000_000}
    assert prefetch.symbols_by_group == {"Software": ["AAPL"]}
    assert prefetch.stats == {
        "target_symbols": 1,
        "symbols_with_prices": 1,
        "cache_miss_symbols": 0,
        "spy_cached": True,
        "skipped_unsupported_symbols": 0,
    }
    benchmark_cache.get_benchmark_data.assert_called_once_with(market="US", period="2y")
    price_cache.get_many.assert_called_once_with(["AAPL"], period="2y")


def test_prefetch_all_data_skips_unsupported_suffix_symbols(db_session, monkeypatch):
    service = _make_group_rank_service()
    spy_data = _price_frame()
    aapl_data = _price_frame()

    price_cache = Mock()
    price_cache.get_many.return_value = {"AAPL": aapl_data}
    service.price_cache = price_cache

    benchmark_cache = Mock()
    benchmark_cache.get_benchmark_symbol.return_value = "SPY"
    benchmark_cache.get_benchmark_data.return_value = spy_data
    service.benchmark_cache = benchmark_cache

    stock_universe_service = Mock()
    stock_universe_service.get_active_symbols.return_value = ["AAPL", "MZYX-U"]
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_stock_universe_service",
        lambda: stock_universe_service,
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db, **kw: ["Software"],
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_group_symbols",
        lambda db, group, **kw: ["AAPL", "MZYX-U"],
    )
    monkeypatch.setattr(
        service,
        "_get_market_caps_for_symbols",
        lambda db, symbols: {"AAPL": 1_000_000_000},
    )

    prefetch = service._prefetch_all_data(
        db_session,
        cache_only=False,
    )

    assert prefetch.active_symbols == {"AAPL", "MZYX-U"}
    assert prefetch.prices_by_symbol == {"AAPL": aapl_data}
    assert prefetch.symbols_by_group == {"Software": ["AAPL"]}
    assert prefetch.stats["target_symbols"] == 1
    assert prefetch.stats["skipped_unsupported_symbols"] == 1
    price_cache.get_many.assert_called_once_with(["AAPL"], period="2y")


def test_calculate_group_rankings_rejects_incomplete_cache_only_inputs(db_session, monkeypatch):
    service = _make_group_rank_service()
    price_data = _price_frame()

    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db, **kw: ["Software"],
    )
    monkeypatch.setattr(
        service,
        "_prefetch_all_data",
        lambda db, cache_only=False, **kw: (
            price_data,
            {"AAPL": price_data},
            {"AAPL"},
            {"AAPL": 1_000_000_000},
            {
                "target_symbols": 2,
                "symbols_with_prices": 1,
                "cache_miss_symbols": 1,
                "spy_cached": True,
            },
        ),
    )
    store_rankings = Mock()
    monkeypatch.setattr(service, "_store_rankings", store_rankings)

    with pytest.raises(IncompleteGroupRankingCacheError) as excinfo:
        service.calculate_group_rankings(
            db_session,
            date(2026, 3, 20),
            cache_only=True,
            require_complete_cache=True,
        )

    assert excinfo.value.stats["cache_miss_symbols"] == 1
    store_rankings.assert_not_called()


def test_store_rankings_bulk_loads_existing_rows_once_for_sqlite_fallback():
    service = _make_group_rank_service()
    db_session = _make_session()
    engine = db_session.get_bind()
    query_counts = {"select": 0}

    def count_selects(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            query_counts["select"] += 1

    try:
        db_session.add(
            IBDGroupRank(
                industry_group="Software",
                date=date(2026, 3, 20),
                rank=9,
                avg_rs_rating=70.0,
                num_stocks=3,
                num_stocks_rs_above_80=1,
                top_symbol="OLD",
                top_rs_rating=80.0,
            )
        )
        db_session.commit()

        event.listen(engine, "before_cursor_execute", count_selects)
        service._store_rankings(
            db_session,
            date(2026, 3, 20),
            [
                {
                    "industry_group": "Software",
                    "rank": 1,
                    "avg_rs_rating": 91.0,
                    "median_rs_rating": 90.0,
                    "weighted_avg_rs_rating": 92.0,
                    "rs_std_dev": 1.5,
                    "num_stocks": 4,
                    "num_stocks_rs_above_80": 3,
                    "top_symbol": "AAPL",
                    "top_rs_rating": 99.0,
                },
                {
                    "industry_group": "Semiconductors",
                    "rank": 2,
                    "avg_rs_rating": 88.0,
                    "median_rs_rating": 87.0,
                    "weighted_avg_rs_rating": 89.0,
                    "rs_std_dev": 2.5,
                    "num_stocks": 5,
                    "num_stocks_rs_above_80": 4,
                    "top_symbol": "NVDA",
                    "top_rs_rating": 98.0,
                },
            ],
        )
        event.remove(engine, "before_cursor_execute", count_selects)

        rows = db_session.query(IBDGroupRank).order_by(IBDGroupRank.rank).all()

        assert query_counts["select"] == 1
        assert len(rows) == 2
        assert rows[0].industry_group == "Software"
        assert rows[0].top_symbol == "AAPL"
        assert rows[1].industry_group == "Semiconductors"
    finally:
        try:
            event.remove(engine, "before_cursor_execute", count_selects)
        except Exception:
            pass
        db_session.rollback()
        db_session.close()


def test_calculate_group_rankings_fails_explicitly_when_ibd_mappings_missing(db_session, monkeypatch):
    service = _make_group_rank_service()

    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db, **kw: [],
    )

    with pytest.raises(MissingIBDIndustryMappingsError, match="IBD industry mappings are not loaded"):
        service.calculate_group_rankings(db_session, date(2026, 3, 20), market="US")


def test_calculate_group_rankings_propagates_group_lookup_failures(db_session, monkeypatch):
    service = _make_group_rank_service()

    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db, **kw: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )

    with pytest.raises(RuntimeError, match="database unavailable"):
        service.calculate_group_rankings(db_session, date(2026, 3, 20), market="US")


def test_backfill_rankings_optimized_accepts_prefetch_stats_tuple(db_session, monkeypatch):
    service = _make_group_rank_service()
    price_data = _price_frame()
    delete_kwargs: dict = {}
    prefetch_kwargs: dict = {}
    group_kwargs: dict = {}

    class _FakeCalendarService:
        def is_trading_day(self, market, day):
            assert market == "HK"
            return day == date(2026, 3, 17)

    monkeypatch.setattr(
        "app.wiring.bootstrap.get_market_calendar_service",
        lambda: _FakeCalendarService(),
    )

    monkeypatch.setattr(
        service,
        "_delete_rankings_for_range",
        lambda db, start_date, end_date, **kw: delete_kwargs.update(kw) or 0,
    )
    monkeypatch.setattr(
        service,
        "_prefetch_all_data",
        lambda db, **kw: prefetch_kwargs.update(kw) or (
            price_data,
            {"AAPL": price_data},
            {"AAPL"},
            {"AAPL": 1_000_000_000},
            {"target_symbols": 1, "symbols_with_prices": 1, "cache_miss_symbols": 0, "spy_cached": True},
        ),
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db, **kw: group_kwargs.update(kw) or [],
    )

    stats = service.backfill_rankings_optimized(
        db_session,
        date(2026, 3, 17),
        date(2026, 3, 17),
        market="HK",
    )

    assert stats["processed"] == 0
    assert stats["errors"] == 1
    assert delete_kwargs["market"] == "HK"
    assert prefetch_kwargs["market"] == "HK"
    assert group_kwargs["market"] == "HK"


def test_backfill_rankings_optimized_uses_market_calendar(db_session, monkeypatch):
    service = _make_group_rank_service()
    price_data = _price_frame()
    calendar_calls: list[tuple[str, date]] = []

    class _FakeCalendarService:
        def is_trading_day(self, market, day):
            calendar_calls.append((market, day))
            return market == "HK" and day == date(2026, 3, 18)

    monkeypatch.setattr(
        "app.wiring.bootstrap.get_market_calendar_service",
        lambda: _FakeCalendarService(),
    )
    monkeypatch.setattr(service, "_delete_rankings_for_range", lambda *args, **kw: 0)
    monkeypatch.setattr(
        service,
        "_prefetch_all_data",
        lambda db, **kw: (
            price_data,
            {"0700.HK": price_data},
            {"0700.HK"},
            {"0700.HK": 1_000_000_000},
            {"target_symbols": 1, "symbols_with_prices": 1, "cache_miss_symbols": 0, "spy_cached": True},
        ),
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db, **kw: [],
    )

    stats = service.backfill_rankings_optimized(
        db_session,
        date(2026, 3, 17),
        date(2026, 3, 19),
        market="HK",
    )

    assert stats["total_dates"] == 1
    assert stats["errors"] == 1
    assert calendar_calls == [
        ("HK", date(2026, 3, 19)),
        ("HK", date(2026, 3, 18)),
        ("HK", date(2026, 3, 17)),
    ]


def test_backfill_rankings_checks_existing_rows_by_market(db_session, monkeypatch):
    service = _make_group_rank_service()

    _add_rank(db_session, "Software", date(2026, 3, 17), 5)
    db_session.commit()

    class _FakeCalendarService:
        def is_trading_day(self, market, day):
            assert market == "HK"
            return day == date(2026, 3, 17)

    monkeypatch.setattr(
        "app.wiring.bootstrap.get_market_calendar_service",
        lambda: _FakeCalendarService(),
    )

    calculate_calls: list[date] = []
    monkeypatch.setattr(
        service,
        "calculate_group_rankings",
        lambda db, calc_date, **kw: calculate_calls.append(calc_date) or [{"rank": 1}],
    )

    stats = service.backfill_rankings(
        db_session,
        date(2026, 3, 17),
        date(2026, 3, 17),
        market="HK",
    )

    assert stats["processed"] == 1
    assert stats["skipped"] == 0
    assert calculate_calls == [date(2026, 3, 17)]


def test_fill_gaps_optimized_accepts_prefetch_stats_tuple(db_session, monkeypatch):
    service = _make_group_rank_service()
    price_data = _price_frame()
    prefetch_kwargs: dict = {}
    group_kwargs: dict = {}

    monkeypatch.setattr(
        service,
        "_prefetch_all_data",
        lambda db, **kw: prefetch_kwargs.update(kw) or (
            price_data,
            {"AAPL": price_data},
            {"AAPL"},
            {"AAPL": 1_000_000_000},
            {"target_symbols": 1, "symbols_with_prices": 1, "cache_miss_symbols": 0, "spy_cached": True},
        ),
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db, **kw: group_kwargs.update(kw) or [],
    )

    stats = service.fill_gaps_optimized(db_session, [date(2026, 3, 17)], market="HK")

    assert stats["processed"] == 0
    assert stats["errors"] == 1
    assert prefetch_kwargs["market"] == "HK"
    assert group_kwargs["market"] == "HK"


def test_fill_gaps_optimized_uses_prefetched_group_symbols_without_inner_lookup(db_session, monkeypatch):
    service = _make_group_rank_service()
    price_data = _price_frame()
    symbols = ["AAA", "BBB", "CCC"]
    prefetch = group_rank_module.GroupRankPrefetchData(
        benchmark_prices=price_data,
        prices_by_symbol={symbol: price_data for symbol in symbols},
        active_symbols=set(symbols),
        market_caps={symbol: 1_000_000_000 for symbol in symbols},
        stats={
            "target_symbols": 3,
            "symbols_with_prices": 3,
            "cache_miss_symbols": 0,
            "spy_cached": True,
            "skipped_unsupported_symbols": 0,
        },
        symbols_by_group={"Software": symbols},
    )
    monkeypatch.setattr(service, "_prefetch_all_data", lambda db, **kw: prefetch)
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db, **kw: ["Software"],
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_group_symbols",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("optimized gapfill should reuse prefetched symbols_by_group")
        ),
    )

    stats = service.fill_gaps_optimized(db_session, [date(2026, 3, 20)], market="US")

    assert stats["processed"] == 1
    row = db_session.query(IBDGroupRank).filter(
        IBDGroupRank.date == date(2026, 3, 20),
        IBDGroupRank.industry_group == "Software",
    ).one()
    assert row.rank == 1
    assert row.num_stocks == 3
    assert row.avg_rs_rating == 50.0


def test_vectorized_group_rs_matches_legacy_cache_path_and_excludes_short_history(
    db_session,
    monkeypatch,
):
    service = _make_group_rank_service()
    calculation_date = date(2026, 3, 20)
    symbols = ["AAA", "BBB", "CCC", "NEW"]
    benchmark_prices = _trend_price_frame(daily_return=0.0004)
    prices_by_symbol = {
        "AAA": _trend_price_frame(start_close=100.0, daily_return=0.0020),
        "BBB": _trend_price_frame(start_close=80.0, daily_return=0.0011),
        "CCC": _trend_price_frame(start_close=120.0, daily_return=-0.0002),
        "NEW": _trend_price_frame(start_close=50.0, daily_return=0.0040, periods=200),
    }
    market_caps = {
        "AAA": 10_000_000_000,
        "BBB": 5_000_000_000,
        "CCC": 1_000_000_000,
        "NEW": 50_000_000_000,
    }
    active_symbols = set(symbols)
    monkeypatch.setattr(
        service,
        "_get_validated_group_symbols",
        lambda *_args, **_kwargs: symbols,
    )
    legacy_metrics = service._calculate_group_rs_from_cache(
        db_session,
        "Software",
        benchmark_prices["Close"].sort_index(ascending=False),
        prices_by_symbol,
        active_symbols,
        market_caps,
        calculation_date,
    )
    prefetch = group_rank_module.GroupRankPrefetchData(
        benchmark_prices=benchmark_prices,
        prices_by_symbol=prices_by_symbol,
        active_symbols=active_symbols,
        market_caps=market_caps,
        stats={},
        symbols_by_group={"Software": symbols},
    )

    rs_by_date = service._calculate_rs_by_symbol_for_dates(prefetch, [calculation_date])
    vectorized_metrics = service._calculate_group_metrics_from_rs(
        "Software",
        symbols,
        rs_by_date[calculation_date],
        market_caps,
        calculation_date,
    )

    assert set(rs_by_date[calculation_date]) == {"AAA", "BBB", "CCC"}
    assert vectorized_metrics == legacy_metrics


def test_vectorized_group_rs_preserves_invalid_period_return_semantics(
    db_session,
    monkeypatch,
):
    service = _make_group_rank_service()
    calculation_date = date(2026, 3, 20)
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    benchmark_prices = _trend_price_frame(daily_return=0.0004)
    prices_by_symbol = {
        "AAA": _trend_price_frame(start_close=100.0, daily_return=0.0020),
        "BBB": _trend_price_frame(start_close=80.0, daily_return=0.0011),
        "CCC": _trend_price_frame(start_close=120.0, daily_return=-0.0002),
        "DDD": _trend_price_frame(start_close=90.0, daily_return=0.0017),
        "EEE": _trend_price_frame(start_close=70.0, daily_return=0.0009),
    }
    prices_by_symbol["BBB"].iloc[-63, prices_by_symbol["BBB"].columns.get_loc("Close")] = 0.0
    prices_by_symbol["DDD"].iloc[-1, prices_by_symbol["DDD"].columns.get_loc("Close")] = float("nan")
    prices_by_symbol["EEE"].iloc[-63, prices_by_symbol["EEE"].columns.get_loc("Close")] = float("nan")
    market_caps = {
        "AAA": 10_000_000_000,
        "BBB": 5_000_000_000,
        "CCC": 1_000_000_000,
        "DDD": 7_000_000_000,
        "EEE": 2_000_000_000,
    }
    active_symbols = set(symbols)
    monkeypatch.setattr(
        service,
        "_get_validated_group_symbols",
        lambda *_args, **_kwargs: symbols,
    )
    legacy_metrics = service._calculate_group_rs_from_cache(
        db_session,
        "Software",
        benchmark_prices["Close"].sort_index(ascending=False),
        prices_by_symbol,
        active_symbols,
        market_caps,
        calculation_date,
    )
    prefetch = group_rank_module.GroupRankPrefetchData(
        benchmark_prices=benchmark_prices,
        prices_by_symbol=prices_by_symbol,
        active_symbols=active_symbols,
        market_caps=market_caps,
        stats={},
        symbols_by_group={"Software": symbols},
    )

    rs_by_date = service._calculate_rs_by_symbol_for_dates(prefetch, [calculation_date])
    vectorized_metrics = service._calculate_group_metrics_from_rs(
        "Software",
        symbols,
        rs_by_date[calculation_date],
        market_caps,
        calculation_date,
    )

    assert set(rs_by_date[calculation_date]) == {"AAA", "BBB", "CCC", "EEE"}
    assert vectorized_metrics == legacy_metrics


def test_get_current_rankings_can_target_explicit_date():
    service = _make_group_rank_service()
    db_session = _make_session()
    group = f"TEST_GROUP_UNIT_{uuid4().hex}"

    try:
        _add_rank(db_session, group, date(2024, 1, 10), 5)
        _add_rank(db_session, group, date(2024, 1, 17), 3)
        db_session.commit()

        rankings = service.get_current_rankings(
            db_session,
            limit=10,
            calculation_date=date(2024, 1, 10),
        )

        assert len(rankings) == 1
        assert rankings[0]["date"] == "2024-01-10"
        assert rankings[0]["rank"] == 5
    finally:
        db_session.rollback()
        db_session.close()
