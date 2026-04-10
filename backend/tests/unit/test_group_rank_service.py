from datetime import date
from uuid import uuid4
from unittest.mock import Mock

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.industry import IBDGroupRank
from app.services.ibd_group_rank_service import (
    IBDGroupRankService,
    IncompleteGroupRankingCacheError,
)


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
    Base.metadata.create_all(engine, tables=[IBDGroupRank.__table__])
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)()


def _make_group_rank_service(price_cache: Mock | None = None, benchmark_cache: Mock | None = None):
    return IBDGroupRankService(
        price_cache=price_cache or Mock(),
        benchmark_cache=benchmark_cache or Mock(),
    )


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
    benchmark_cache.get_spy_data.side_effect = AssertionError("benchmark fetch should not be used")
    service.benchmark_cache = benchmark_cache

    monkeypatch.setattr(
        "app.services.stock_universe_service.stock_universe_service.get_active_symbols",
        lambda db: ["AAPL"],
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db: ["Software"],
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_group_symbols",
        lambda db, group: ["AAPL"],
    )
    monkeypatch.setattr(
        service,
        "_get_market_caps_for_symbols",
        lambda db, symbols: {"AAPL": 1_000_000_000},
    )

    spy, all_prices, active_symbols, market_caps, stats = service._prefetch_all_data(
        db_session,
        cache_only=True,
    )

    assert spy is spy_data
    assert all_prices == {"AAPL": aapl_data}
    assert active_symbols == {"AAPL"}
    assert market_caps == {"AAPL": 1_000_000_000}
    assert stats == {
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
    benchmark_cache.get_spy_data.side_effect = AssertionError("benchmark fetch should not be used")
    service.benchmark_cache = benchmark_cache

    monkeypatch.setattr(
        "app.services.stock_universe_service.stock_universe_service.get_active_symbols",
        lambda db: ["AAPL"],
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db: ["Software"],
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_group_symbols",
        lambda db, group: ["AAPL"],
    )
    monkeypatch.setattr(
        service,
        "_get_market_caps_for_symbols",
        lambda db, symbols: {"AAPL": 1_000_000_000},
    )

    spy, all_prices, active_symbols, market_caps, stats = service._prefetch_all_data(
        db_session,
        cache_only=True,
    )

    assert spy is spy_data
    assert all_prices == {"AAPL": None}
    assert active_symbols == {"AAPL"}
    assert market_caps == {"AAPL": 1_000_000_000}
    assert stats == {
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
    benchmark_cache.get_spy_data.return_value = spy_data
    service.benchmark_cache = benchmark_cache

    monkeypatch.setattr(
        "app.services.stock_universe_service.stock_universe_service.get_active_symbols",
        lambda db: ["AAPL"],
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db: ["Software"],
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_group_symbols",
        lambda db, group: ["AAPL"],
    )
    monkeypatch.setattr(
        service,
        "_get_market_caps_for_symbols",
        lambda db, symbols: {"AAPL": 1_000_000_000},
    )

    spy, all_prices, active_symbols, market_caps, stats = service._prefetch_all_data(
        db_session,
        cache_only=False,
    )

    assert spy is spy_data
    assert all_prices == {"AAPL": aapl_data}
    assert active_symbols == {"AAPL"}
    assert market_caps == {"AAPL": 1_000_000_000}
    assert stats == {
        "target_symbols": 1,
        "symbols_with_prices": 1,
        "cache_miss_symbols": 0,
        "spy_cached": True,
        "skipped_unsupported_symbols": 0,
    }
    benchmark_cache.get_spy_data.assert_called_once_with(period="2y")
    price_cache.get_many.assert_called_once_with(["AAPL"], period="2y")


def test_prefetch_all_data_skips_unsupported_suffix_symbols(db_session, monkeypatch):
    service = _make_group_rank_service()
    spy_data = _price_frame()
    aapl_data = _price_frame()

    price_cache = Mock()
    price_cache.get_many.return_value = {"AAPL": aapl_data}
    service.price_cache = price_cache

    benchmark_cache = Mock()
    benchmark_cache.get_spy_data.return_value = spy_data
    service.benchmark_cache = benchmark_cache

    monkeypatch.setattr(
        "app.services.stock_universe_service.stock_universe_service.get_active_symbols",
        lambda db: ["AAPL", "MZYX-U"],
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db: ["Software"],
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_group_symbols",
        lambda db, group: ["AAPL", "MZYX-U"],
    )
    monkeypatch.setattr(
        service,
        "_get_market_caps_for_symbols",
        lambda db, symbols: {"AAPL": 1_000_000_000},
    )

    _spy, all_prices, active_symbols, _market_caps, stats = service._prefetch_all_data(
        db_session,
        cache_only=False,
    )

    assert active_symbols == {"AAPL", "MZYX-U"}
    assert all_prices == {"AAPL": aapl_data}
    assert stats["target_symbols"] == 1
    assert stats["skipped_unsupported_symbols"] == 1
    price_cache.get_many.assert_called_once_with(["AAPL"], period="2y")


def test_calculate_group_rankings_rejects_incomplete_cache_only_inputs(db_session, monkeypatch):
    service = _make_group_rank_service()
    price_data = _price_frame()

    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db: ["Software"],
    )
    monkeypatch.setattr(
        service,
        "_prefetch_all_data",
        lambda db, cache_only=False: (
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


def test_backfill_rankings_optimized_accepts_prefetch_stats_tuple(db_session, monkeypatch):
    service = _make_group_rank_service()
    price_data = _price_frame()

    monkeypatch.setattr(
        service,
        "_delete_rankings_for_range",
        lambda db, start_date, end_date: 0,
    )
    monkeypatch.setattr(
        service,
        "_prefetch_all_data",
        lambda db: (
            price_data,
            {"AAPL": price_data},
            {"AAPL"},
            {"AAPL": 1_000_000_000},
            {"target_symbols": 1, "symbols_with_prices": 1, "cache_miss_symbols": 0, "spy_cached": True},
        ),
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db: [],
    )

    stats = service.backfill_rankings_optimized(
        db_session,
        date(2026, 3, 17),
        date(2026, 3, 17),
    )

    assert stats["processed"] == 0
    assert stats["errors"] == 1


def test_fill_gaps_optimized_accepts_prefetch_stats_tuple(db_session, monkeypatch):
    service = _make_group_rank_service()
    price_data = _price_frame()

    monkeypatch.setattr(
        service,
        "_prefetch_all_data",
        lambda db: (
            price_data,
            {"AAPL": price_data},
            {"AAPL"},
            {"AAPL": 1_000_000_000},
            {"target_symbols": 1, "symbols_with_prices": 1, "cache_miss_symbols": 0, "spy_cached": True},
        ),
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db: [],
    )

    stats = service.fill_gaps_optimized(db_session, [date(2026, 3, 17)])

    assert stats["processed"] == 0
    assert stats["errors"] == 1


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
