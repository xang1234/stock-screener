from datetime import date, datetime
import inspect
from pathlib import Path
from uuid import uuid4
from unittest.mock import Mock

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.database import Base
from app.models.industry import IBDGroupRank
from app.models.stock_universe import StockUniverse
from app.services.derived_data_execution_policy import (
    resolve_derived_data_execution_policy,
)
from app.services.group_rank_cache_policy import GroupRankCacheRequirement
from app.services.group_rank_historical_calculator import (
    GroupRankHistoricalCalculator,
)
from app.services.group_rank_input_loader import GroupRankInputLoader
from app.services.group_rank_input_sources import (
    IBDIndustryTaxonomySource,
    SqlGroupRankMarketCapSource,
    StockUniverseGroupRankSource,
)
from app.services.group_rank_models import (
    GroupRankCalculationResult,
    GroupRankPrefetchData,
    GroupRankPrefetchStats,
)
from app.services.group_rank_legacy_adapter import (
    LegacyGroupRankPrefetchAdapter,
)
from app.services.group_ranking_calculator import (
    GroupRankingCalculator,
)
from app.services.group_ranking_repository import (
    GroupRankingRepository,
)
from app.services.stock_universe_service import StockUniverseService
from app.services.market_calendar_service import MarketCalendarService
from app.scanners.criteria.relative_strength import (
    RelativeStrengthCalculator,
)
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


def _input_loader(price_cache, benchmark_cache):
    return GroupRankInputLoader(
        price_cache=price_cache,
        benchmark_cache=benchmark_cache,
        universe_source=StockUniverseGroupRankSource(
            StockUniverseService()
        ),
        taxonomy_source=IBDIndustryTaxonomySource(),
        market_cap_source=SqlGroupRankMarketCapSource(),
    )


def _make_group_rank_service(
    price_cache: Mock | None = None,
    benchmark_cache: Mock | None = None,
    group_constituent_source=None,
):
    price_cache = price_cache or Mock()
    benchmark_cache = benchmark_cache or Mock()
    input_loader = _input_loader(price_cache, benchmark_cache)
    rs_calculator = RelativeStrengthCalculator()
    ranking_calculator = GroupRankingCalculator(rs_calculator)
    ranking_repository = GroupRankingRepository()
    legacy_adapter = LegacyGroupRankPrefetchAdapter()
    historical_calculator = GroupRankHistoricalCalculator(
        input_loader=input_loader,
        ranking_calculator=ranking_calculator,
        repository=ranking_repository,
        calendar_service=MarketCalendarService(),
        legacy_adapter=legacy_adapter,
    )
    return IBDGroupRankService(
        price_cache=price_cache,
        benchmark_cache=benchmark_cache,
        rs_calculator=rs_calculator,
        group_constituent_source=group_constituent_source,
        input_loader=input_loader,
        ranking_calculator=ranking_calculator,
        ranking_repository=ranking_repository,
        historical_calculator=historical_calculator,
        legacy_prefetch_adapter=legacy_adapter,
    )


def _policy(mode: str, target: date):
    return resolve_derived_data_execution_policy(
        execution_policy=mode,
        target_date=target,
        current_date=date(2026, 3, 20),
    )


def _prefetch_stats(
    target_symbols: int,
    *,
    symbols_with_prices: int | None = None,
    cache_miss_symbols: int = 0,
    cache_miss_symbols_sample: tuple[str, ...] = (),
    cache_only: bool = True,
) -> GroupRankPrefetchStats:
    available = (
        target_symbols
        if symbols_with_prices is None
        else symbols_with_prices
    )
    return GroupRankPrefetchStats(
        target_symbols=target_symbols,
        symbols_with_prices=available,
        cache_miss_symbols=cache_miss_symbols,
        cache_miss_symbols_sample=cache_miss_symbols_sample,
        cache_coverage_ratio=(
            available / target_symbols
            if target_symbols
            else 1.0
        ),
        benchmark_available=True,
        benchmark_cached=cache_only,
        benchmark_symbol="SPY",
        benchmark_role="primary",
        market="US",
        cache_only=cache_only,
        skipped_unsupported_symbols=0,
    )


def _ported_find_missing_dates_uses_market_calendar(db_session, monkeypatch):
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
            service.ranking_repository,
            "historical_ranks_batch",
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


def test_get_group_history_uses_calendar_rank_change_offsets(monkeypatch):
    service = _make_group_rank_service()
    db_session = _make_session()
    group = f"TEST_GROUP_UNIT_{uuid4().hex}"
    current_date = date.today()
    captured_period_days: list[dict[str, int]] = []

    try:
        db_session.add(
            IBDGroupRank(
                market="US",
                industry_group=group,
                date=current_date,
                rank=4,
                avg_rs_rating=88.0,
                median_rs_rating=87.0,
                weighted_avg_rs_rating=89.0,
                rs_std_dev=2.0,
                num_stocks=5,
                num_stocks_rs_above_80=3,
                top_symbol="AAPL",
                top_rs_rating=96.0,
            )
        )
        db_session.commit()

        expected_period_days = dict(group_rank_module.GROUP_RANK_CHANGE_CALENDAR_DAYS)

        def fake_historical_batch(db, *, group_names, current_date, period_days, market):  # noqa: ANN001
            captured_period_days.append(dict(period_days))
            return {(group, "1w"): 6}

        monkeypatch.setattr(
            service.ranking_repository,
            "historical_ranks_batch",
            fake_historical_batch,
        )
        monkeypatch.setattr(service, "_get_constituent_stocks", lambda *_args, **_kwargs: [])

        result = service.get_group_history(db_session, group, days=30, market="US")

        assert captured_period_days == [expected_period_days]
        assert result["rank_change_1w"] == 2
        assert result["rank_change_1m"] is None
    finally:
        db_session.rollback()
        db_session.close()


def test_get_current_rankings_uses_calendar_rank_change_offsets(monkeypatch):
    service = _make_group_rank_service()
    db_session = _make_session()
    group = f"TEST_GROUP_UNIT_{uuid4().hex}"
    current_date = date(2026, 4, 18)
    captured_period_days: list[dict[str, int]] = []

    try:
        _add_rank(db_session, group, current_date, 4)
        db_session.commit()

        expected_period_days = dict(group_rank_module.GROUP_RANK_CHANGE_CALENDAR_DAYS)

        def fake_historical_batch(db, *, group_names, current_date, period_days, market):  # noqa: ANN001
            captured_period_days.append(dict(period_days))
            return {(group, "1w"): 6}

        monkeypatch.setattr(
            service.ranking_repository,
            "historical_ranks_batch",
            fake_historical_batch,
        )

        rankings = service.get_current_rankings(
            db_session,
            limit=10,
            calculation_date=current_date,
            market="US",
        )

        assert captured_period_days == [expected_period_days]
        assert rankings[0]["rank_change_1w"] == 2
        assert rankings[0]["rank_change_1m"] is None
    finally:
        db_session.rollback()
        db_session.close()


def test_get_group_history_propagates_constituent_source_failures(monkeypatch):
    service = _make_group_rank_service(
        group_constituent_source=Mock(),
    )
    service.group_constituent_source.get_constituent_items.side_effect = RuntimeError(
        "source failed"
    )
    db_session = _make_session()
    group = f"TEST_GROUP_UNIT_{uuid4().hex}"
    current_date = date.today()

    try:
        db_session.add(
            IBDGroupRank(
                market="US",
                industry_group=group,
                date=current_date,
                rank=1,
                avg_rs_rating=91.0,
                num_stocks=1,
                num_stocks_rs_above_80=1,
                top_symbol="AAPL",
                top_rs_rating=96.0,
            )
        )
        db_session.commit()
        monkeypatch.setattr(
            service.ranking_repository,
            "historical_ranks_batch",
            lambda *_args, **_kwargs: {},
        )

        with pytest.raises(RuntimeError, match="source failed"):
            service.get_group_history(db_session, group, days=30, market="US")
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


def test_cache_only_missing_market_benchmark_names_market_symbol(db_session, monkeypatch):
    price_cache = Mock()
    price_cache.get_cached_only_fresh.return_value = None
    price_cache.get_many_cached_only_fresh.return_value = {}

    benchmark_cache = Mock()
    benchmark_cache.get_benchmark_symbol.return_value = "^N225"
    benchmark_cache.get_benchmark_candidates.return_value = ["^N225"]
    service = _make_group_rank_service(
        price_cache=price_cache,
        benchmark_cache=benchmark_cache,
    )

    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db, **kw: ["JP_Software"],
    )

    with pytest.raises(IncompleteGroupRankingCacheError) as excinfo:
        service.calculate_group_rankings(
            db_session,
            date(2026, 5, 1),
            market="JP",
            policy=_policy("strict_cache_only", date(2026, 5, 1)),
            cache_requirement=GroupRankCacheRequirement.strict(),
        )

    assert str(excinfo.value) == "^N225 benchmark data is missing from cache for JP"
    assert excinfo.value.stats.benchmark_symbol == "^N225"
    assert excinfo.value.stats.market == "JP"


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
    monkeypatch.setattr(
        service.ranking_repository,
        "store_rankings",
        store_rankings,
    )

    with pytest.raises(IncompleteGroupRankingCacheError) as excinfo:
        service.calculate_group_rankings(
            db_session,
            date(2026, 3, 20),
            policy=_policy("strict_cache_only", date(2026, 3, 20)),
            cache_requirement=GroupRankCacheRequirement.strict(),
        )

    assert excinfo.value.stats.cache_miss_symbols == 1
    store_rankings.assert_not_called()


def test_calculate_group_rankings_rejects_cache_coverage_below_minimum(db_session, monkeypatch):
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
            {"AAPL": price_data, "MSFT": None},
            {"AAPL", "MSFT"},
            {"AAPL": 1_000_000_000, "MSFT": 1_000_000_000},
            {
                "target_symbols": 2,
                "symbols_with_prices": 1,
                "cache_miss_symbols": 1,
                "spy_cached": True,
            },
        ),
    )
    store_rankings = Mock()
    monkeypatch.setattr(
        service.ranking_repository,
        "store_rankings",
        store_rankings,
    )

    with pytest.raises(IncompleteGroupRankingCacheError) as excinfo:
        service.calculate_group_rankings(
            db_session,
            date(2026, 6, 10),
            market="TW",
            policy=_policy("strict_cache_only", date(2026, 6, 10)),
            cache_requirement=GroupRankCacheRequirement.minimum(0.55, reason="test"),
        )

    assert excinfo.value.stats.cache_coverage_ratio == 0.5
    assert excinfo.value.stats.cache_coverage_min == 0.55
    store_rankings.assert_not_called()


def test_calculate_group_rankings_tolerates_partial_cache_when_requirement_disabled(
    db_session,
    monkeypatch,
):
    service = _make_group_rank_service()
    price_data = _price_frame()
    symbols = ["AAA", "BBB", "CCC", "MISS"]
    stats = GroupRankPrefetchStats(
        target_symbols=4,
        symbols_with_prices=3,
        cache_miss_symbols=1,
        cache_miss_symbols_sample=("MISS",),
        cache_coverage_ratio=0.75,
        benchmark_available=True,
        benchmark_cached=True,
        benchmark_symbol="SPY",
        benchmark_role="primary",
        market="US",
        cache_only=True,
        skipped_unsupported_symbols=0,
    )
    prefetch = GroupRankPrefetchData(
        benchmark_prices=price_data,
        prices_by_symbol={
            "AAA": price_data,
            "BBB": price_data,
            "CCC": price_data,
            "MISS": None,
        },
        active_symbols=frozenset(symbols),
        market_caps={symbol: 1_000_000_000 for symbol in symbols},
        stats=stats,
        symbols_by_group={"Software": tuple(symbols)},
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db, **kwargs: ["Software"],
    )
    monkeypatch.setattr(service, "_prefetch_all_data", lambda db, **kwargs: prefetch)
    service.ranking_calculator = Mock()
    service.ranking_calculator.calculate_for_date.return_value = (
        {
            "industry_group": "Software",
            "rank": 1,
            "avg_rs_rating": 85.0,
            "num_stocks": 3,
        },
    )
    store_rankings = Mock()
    monkeypatch.setattr(
        service.ranking_repository,
        "store_rankings",
        store_rankings,
    )
    calculation = service.calculate_group_rankings(
        db_session,
        date(2026, 3, 20),
        market="US",
        policy=resolve_derived_data_execution_policy(
            execution_policy="refresh_guarded",
            target_date=date(2026, 3, 20),
            current_date=date(2026, 3, 20),
        ),
        cache_requirement=GroupRankCacheRequirement.disabled(),
    )

    assert len(calculation.rankings) == 1
    assert calculation.rankings[0]["industry_group"] == "Software"
    assert calculation.rankings[0]["num_stocks"] == 3
    assert calculation.prefetch_stats.cache_miss_symbols == 1
    assert calculation.prefetch_stats.cache_miss_symbols_sample == ("MISS",)
    store_rankings.assert_called_once()


def test_calculate_group_rankings_has_no_diagnostics_output_parameter():
    signature = inspect.signature(
        IBDGroupRankService.calculate_group_rankings
    )

    assert "diagnostics" not in signature.parameters


def test_group_rank_service_is_a_compatibility_facade():
    source = Path(group_rank_module.__file__).read_text()
    assert "gc.collect" not in source
    assert "get_market_calendar_service" not in source
    assert "pg_insert" not in source
    assert "StockUniverse" not in source
    assert source.count("\n") < 900

    for method_name in (
        "calculate_group_rankings",
        "get_current_rankings",
        "get_historical_ranks_batch",
        "get_group_history",
        "get_rank_movers",
        "backfill_rankings_optimized",
        "backfill_rankings",
        "find_missing_dates",
        "fill_gaps",
        "fill_gaps_optimized",
        "backfill_rankings_chunked",
    ):
        assert hasattr(IBDGroupRankService, method_name)


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


def _ported_backfill_rankings_optimized_accepts_prefetch_stats_tuple(db_session, monkeypatch):
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
        service.ranking_repository,
        "delete_range",
        lambda db, **kw: delete_kwargs.update(kw) or 0,
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


def _ported_backfill_rankings_optimized_uses_market_calendar(db_session, monkeypatch):
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
    monkeypatch.setattr(
        service.ranking_repository,
        "delete_range",
        lambda *args, **kw: 0,
    )
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


def _ported_backfill_rankings_optimized_chunks_rs_date_calculation(db_session, monkeypatch):
    service = _make_group_rank_service()
    price_data = _price_frame()
    symbols = ["AAA", "BBB", "CCC"]
    prefetch = group_rank_module.GroupRankPrefetchData(
        benchmark_prices=price_data,
        prices_by_symbol={symbol: price_data for symbol in symbols},
        active_symbols=frozenset(symbols),
        market_caps={symbol: 1_000_000_000 for symbol in symbols},
        stats=_prefetch_stats(3),
        symbols_by_group={"Software": tuple(symbols)},
    )
    chunked_dates: list[list[date]] = []

    class _FakeCalendarService:
        def is_trading_day(self, market, day):
            assert market == "US"
            return True

    monkeypatch.setattr(settings, "group_rank_gapfill_chunk_size", 3)
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_market_calendar_service",
        lambda: _FakeCalendarService(),
    )
    monkeypatch.setattr(
        service.ranking_repository,
        "delete_range",
        lambda *args, **kw: 0,
    )
    monkeypatch.setattr(service, "_prefetch_all_data", lambda db, **kw: prefetch)
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db, **kw: ["Software"],
    )

    def fake_rankings_for_dates(*, calculation_dates, **kwargs):
        chunked_dates.append(list(calculation_dates))
        return {
            calc_date: (
                {
                    "industry_group": "Software",
                    "rank": 1,
                    "avg_rs_rating": 50.0,
                    "median_rs_rating": 50.0,
                    "weighted_avg_rs_rating": 50.0,
                    "rs_std_dev": 0.0,
                    "num_stocks": 3,
                    "num_stocks_rs_above_80": 0,
                    "top_symbol": "AAA",
                    "top_rs_rating": 50.0,
                    "date": calc_date,
                },
            )
            for calc_date in calculation_dates
        }

    service.ranking_calculator = Mock()
    service.ranking_calculator.calculate_for_dates.side_effect = (
        fake_rankings_for_dates
    )

    stats = service.backfill_rankings_optimized(
        db_session,
        date(2026, 1, 1),
        date(2026, 1, 7),
        market="US",
    )

    assert [len(chunk) for chunk in chunked_dates] == [3, 3, 1]
    assert stats["processed"] == 7


def _ported_backfill_rankings_checks_existing_rows_by_market(db_session, monkeypatch):
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
        lambda db, calc_date, **kw: (
            calculate_calls.append(calc_date)
            or GroupRankCalculationResult(
                rankings=({"rank": 1},),
                prefetch_stats=_prefetch_stats(1),
            )
        ),
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


def _ported_backfill_optimized_legacy_prefetch_tuple_falls_back_to_validated_group_symbols(
    db_session,
    monkeypatch,
):
    service = _make_group_rank_service()
    price_data = _price_frame()
    symbols = ["AAA", "BBB", "CCC"]

    class _FakeCalendarService:
        def is_trading_day(self, market, day):
            assert market == "US"
            return day == date(2026, 3, 20)

    monkeypatch.setattr(
        "app.wiring.bootstrap.get_market_calendar_service",
        lambda: _FakeCalendarService(),
    )
    monkeypatch.setattr(
        service,
        "_prefetch_all_data",
        lambda db, **kw: (
            price_data,
            {symbol: price_data for symbol in symbols},
            set(symbols),
            {symbol: 1_000_000_000 for symbol in symbols},
            {"target_symbols": 3, "symbols_with_prices": 3, "cache_miss_symbols": 0, "spy_cached": True},
        ),
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db, **kw: ["Software"],
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_group_symbols",
        lambda db, group, **kw: symbols if group == "Software" else [],
    )

    stats = service.backfill_rankings_optimized(
        db_session,
        date(2026, 3, 20),
        date(2026, 3, 20),
        market="US",
    )

    assert stats["processed"] == 1
    row = db_session.query(IBDGroupRank).filter(
        IBDGroupRank.date == date(2026, 3, 20),
        IBDGroupRank.industry_group == "Software",
    ).one()
    assert row.rank == 1
    assert row.num_stocks == 3


def _ported_fill_gaps_optimized_accepts_prefetch_stats_tuple(db_session, monkeypatch):
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


def _ported_fill_gaps_optimized_propagates_policy_and_returns_prefetch_stats(
    db_session,
    monkeypatch,
):
    service = _make_group_rank_service()
    price_data = _price_frame()
    captured: dict = {}
    prefetch = group_rank_module.GroupRankPrefetchData(
        benchmark_prices=price_data,
        prices_by_symbol={"AAA": price_data, "BBB": None},
        active_symbols=frozenset({"AAA", "BBB"}),
        market_caps={"AAA": 1_000_000_000},
        stats=_prefetch_stats(
            2,
            symbols_with_prices=1,
            cache_miss_symbols=1,
            cache_miss_symbols_sample=("BBB",),
        ),
        symbols_by_group={"Software": ("AAA", "BBB")},
    )
    monkeypatch.setattr(
        service,
        "_prefetch_all_data",
        lambda db, **kwargs: captured.update(kwargs) or prefetch,
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db, **kwargs: ["Software"],
    )

    result = service.fill_gaps_optimized(
        db_session,
        [date(2026, 3, 20)],
        market="US",
        policy=_policy("strict_cache_only", date(2026, 3, 20)),
    )

    assert captured["market"] == "US"
    assert captured["policy"].mode.value == "strict_cache_only"
    assert result["prefetch_stats"]["cache_miss_symbols"] == 1
    assert result["prefetch_stats"]["cache_miss_symbols_sample"] == ["BBB"]


def _ported_fill_gaps_optimized_uses_prefetched_group_symbols_without_inner_lookup(db_session, monkeypatch):
    service = _make_group_rank_service()
    price_data = _price_frame()
    symbols = ["AAA", "BBB", "CCC"]
    prefetch = group_rank_module.GroupRankPrefetchData(
        benchmark_prices=price_data,
        prices_by_symbol={symbol: price_data for symbol in symbols},
        active_symbols=frozenset(symbols),
        market_caps={symbol: 1_000_000_000 for symbol in symbols},
        stats=_prefetch_stats(3),
        symbols_by_group={"Software": tuple(symbols)},
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


def _ported_fill_gaps_optimized_chunks_rs_date_calculation(db_session, monkeypatch):
    service = _make_group_rank_service()
    price_data = _price_frame()
    symbols = ["AAA", "BBB", "CCC"]
    prefetch = group_rank_module.GroupRankPrefetchData(
        benchmark_prices=price_data,
        prices_by_symbol={symbol: price_data for symbol in symbols},
        active_symbols=frozenset(symbols),
        market_caps={symbol: 1_000_000_000 for symbol in symbols},
        stats=_prefetch_stats(3),
        symbols_by_group={"Software": tuple(symbols)},
    )
    missing_dates = [date(2026, 1, day) for day in range(1, 8)]
    chunked_dates: list[list[date]] = []

    monkeypatch.setattr(settings, "group_rank_gapfill_chunk_size", 3)
    monkeypatch.setattr(service, "_prefetch_all_data", lambda db, **kw: prefetch)
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db, **kw: ["Software"],
    )

    def fake_rankings_for_dates(*, calculation_dates, **kwargs):
        chunked_dates.append(list(calculation_dates))
        return {
            calc_date: (
                {
                    "industry_group": "Software",
                    "rank": 1,
                    "avg_rs_rating": 50.0,
                    "median_rs_rating": 50.0,
                    "weighted_avg_rs_rating": 50.0,
                    "rs_std_dev": 0.0,
                    "num_stocks": 3,
                    "num_stocks_rs_above_80": 0,
                    "top_symbol": "AAA",
                    "top_rs_rating": 50.0,
                    "date": calc_date,
                },
            )
            for calc_date in calculation_dates
        }

    service.ranking_calculator = Mock()
    service.ranking_calculator.calculate_for_dates.side_effect = (
        fake_rankings_for_dates
    )

    stats = service.fill_gaps_optimized(db_session, missing_dates, market="US")

    assert [len(chunk) for chunk in chunked_dates] == [3, 3, 1]
    assert stats["processed"] == 7
    assert db_session.query(IBDGroupRank).count() == 7


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


def test_get_rank_movers_filters_gainers_and_losers_by_sign(monkeypatch):
    service = _make_group_rank_service()
    rankings = [
        {"industry_group": "Up Big", "rank_change_1w": 8},
        {"industry_group": "Up Small", "rank_change_1w": 1},
        {"industry_group": "Flat", "rank_change_1w": 0},
        {"industry_group": "Down Small", "rank_change_1w": -1},
        {"industry_group": "Down Big", "rank_change_1w": -7},
    ]
    monkeypatch.setattr(
        service,
        "get_current_rankings",
        lambda *args, **kwargs: rankings,
    )

    movers = service.get_rank_movers(Mock(), period="1w", limit=10, market="HK")

    assert [g["industry_group"] for g in movers["gainers"]] == ["Up Big", "Up Small"]
    assert [l["industry_group"] for l in movers["losers"]] == ["Down Big", "Down Small"]


def test_get_rank_movers_omits_losers_when_only_gainers_exist(monkeypatch):
    service = _make_group_rank_service()
    rankings = [
        {"industry_group": "A", "rank_change_1w": 5},
        {"industry_group": "B", "rank_change_1w": 2},
        {"industry_group": "C", "rank_change_1w": 1},
    ]
    monkeypatch.setattr(
        service,
        "get_current_rankings",
        lambda *args, **kwargs: rankings,
    )

    movers = service.get_rank_movers(Mock(), period="1w", limit=10, market="HK")

    assert [g["industry_group"] for g in movers["gainers"]] == ["A", "B", "C"]
    assert movers["losers"] == []
