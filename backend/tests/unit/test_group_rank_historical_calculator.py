from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pandas as pd

import app.services.group_rank_historical_calculator as historical_module
from app.services.derived_data_execution_policy import (
    resolve_derived_data_execution_policy,
)
from app.services.group_rank_historical_calculator import (
    GroupRankHistoricalCalculator,
)
from app.services.group_rank_legacy_adapter import (
    LegacyGroupRankPrefetchAdapter,
)
from app.services.group_rank_models import (
    GroupRankPrefetchData,
    GroupRankPrefetchStats,
    GroupRanking,
)


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": [100.0, 101.0, 102.0]},
        index=pd.bdate_range(end="2026-03-20", periods=3),
    )


def _prefetch(
    *,
    symbols_by_group: dict[str, tuple[str, ...]] | None = None,
) -> GroupRankPrefetchData:
    prices = _prices()
    return GroupRankPrefetchData(
        benchmark_prices=prices,
        prices_by_symbol={"AAA": prices},
        active_symbols=frozenset({"AAA"}),
        market_caps={"AAA": 1_000_000},
        stats=GroupRankPrefetchStats(
            target_symbols=1,
            symbols_with_prices=1,
            cache_miss_symbols=0,
            cache_miss_symbols_sample=(),
            cache_coverage_ratio=1.0,
            benchmark_available=True,
            benchmark_cached=True,
            benchmark_symbol="SPY",
            benchmark_role="primary",
            market="US",
            cache_only=True,
            skipped_unsupported_symbols=0,
        ),
        symbols_by_group=(
            symbols_by_group
            if symbols_by_group is not None
            else {"Software": ("AAA",)}
        ),
        group_names=("Software",),
    )


def _ranking(calculation_date: date) -> GroupRanking:
    return GroupRanking(
        industry_group="Software",
        date=calculation_date,
        rank=1,
        avg_rs_rating=80.0,
        median_rs_rating=80.0,
        weighted_avg_rs_rating=80.0,
        rs_std_dev=0.0,
        num_stocks=3,
        num_stocks_rs_above_80=1,
        top_symbol="AAA",
        top_rs_rating=90.0,
    )


def _historical(*, prefetch=None):
    loader = Mock()
    loader.load.return_value = prefetch or _prefetch()
    loader.taxonomy_source.groups.return_value = ("Software",)
    loader.complete_legacy_symbols.side_effect = (
        lambda db, *, market, group_names, prefetch: (
            prefetch
            if prefetch.symbols_by_group
            else replace(
                prefetch,
                symbols_by_group={"Software": ("AAA",)},
            )
        )
    )
    calculator = Mock()
    calculator.calculate_for_dates.side_effect = (
        lambda *, calculation_dates, **kwargs: {
            item_date: (_ranking(item_date),)
            for item_date in calculation_dates
        }
    )
    repository = Mock()
    repository.delete_range.return_value = 0
    repository.current_rank_rows.return_value = []
    calendar = Mock()
    calendar.is_trading_day.return_value = True
    calendar.market_now.return_value = datetime(2026, 3, 20)
    historical = GroupRankHistoricalCalculator(
        input_loader=loader,
        ranking_calculator=calculator,
        repository=repository,
        calendar_service=calendar,
        legacy_adapter=LegacyGroupRankPrefetchAdapter(),
    )
    return historical, loader, calculator, repository, calendar


def test_optimized_backfill_uses_injected_calendar_and_chunks_dates(
    monkeypatch,
):
    historical, _, calculator, repository, calendar = _historical()
    monkeypatch.setattr(
        historical_module.settings,
        "group_rank_gapfill_chunk_size",
        2,
    )
    db = MagicMock()

    result = historical.backfill_rankings_optimized(
        db,
        date(2026, 3, 18),
        date(2026, 3, 20),
        market="JP",
    )

    assert result["processed"] == 3
    assert [
        len(call.kwargs["calculation_dates"])
        for call in calculator.calculate_for_dates.call_args_list
    ] == [2, 1]
    assert all(
        call.args[0] == "JP"
        for call in calendar.is_trading_day.call_args_list
    )
    assert repository.delete_range.call_args.kwargs["market"] == "JP"


def test_optimized_backfill_adapts_legacy_prefetch_and_completes_symbols():
    prices = _prices()
    legacy = (
        prices,
        {"AAA": prices},
        {"AAA"},
        {"AAA": 1_000_000},
        {
            "target_symbols": 1,
            "symbols_with_prices": 1,
            "cache_miss_symbols": 0,
            "spy_cached": True,
        },
    )
    historical, loader, calculator, _, _ = _historical(
        prefetch=legacy
    )

    result = historical.backfill_rankings_optimized(
        MagicMock(),
        date(2026, 3, 20),
        date(2026, 3, 20),
        market="US",
    )

    assert result["processed"] == 1
    loader.complete_legacy_symbols.assert_called_once()
    used_prefetch = calculator.calculate_for_dates.call_args.kwargs[
        "prefetch"
    ]
    assert used_prefetch.symbols_by_group == {
        "Software": ("AAA",)
    }


def test_fill_gaps_optimized_propagates_policy_and_returns_prefetch_stats():
    historical, loader, _, _, _ = _historical()
    policy = resolve_derived_data_execution_policy(
        execution_policy="refresh_guarded",
        target_date=date(2026, 3, 20),
        current_date=date(2026, 3, 20),
    )

    result = historical.fill_gaps_optimized(
        MagicMock(),
        [date(2026, 3, 20)],
        market="US",
        policy=policy,
    )

    assert result["processed"] == 1
    assert result["prefetch_stats"]["target_symbols"] == 1
    assert loader.load.call_args.kwargs["policy"] is policy


def test_nonoptimized_backfill_checks_existing_rows_by_market():
    historical, _, _, repository, _ = _historical()
    repository.current_rank_rows.side_effect = [
        [Mock()],
        [],
    ]

    result = historical.backfill_rankings(
        MagicMock(),
        date(2026, 3, 19),
        date(2026, 3, 20),
        market="HK",
    )

    assert result["skipped"] == 1
    assert result["processed"] == 1
    assert {
        call.kwargs["market"]
        for call in repository.current_rank_rows.call_args_list
    } == {"HK"}


def test_find_missing_dates_reads_existing_dates_from_repository():
    historical, _, _, repository, calendar = _historical()
    repository.existing_dates.return_value = frozenset(
        {date(2026, 3, 19)}
    )

    result = historical.find_missing_dates(
        MagicMock(),
        lookback_days=2,
        market="US",
        end_date=date(2026, 3, 20),
    )

    assert result == [date(2026, 3, 18)]
    repository.existing_dates.assert_called_once()
    assert repository.existing_dates.call_args.kwargs == {
        "start_date": date(2026, 3, 18),
        "end_date": date(2026, 3, 20),
        "market": "US",
    }
    assert calendar.is_trading_day.call_count == 2


def test_chunked_backfill_aggregates_delegate_results(monkeypatch):
    historical, _, _, _, _ = _historical()
    backfill = Mock(
        side_effect=[
            {
                "total_dates": 2,
                "processed": 1,
                "skipped": 1,
                "errors": 0,
            },
            {
                "total_dates": 1,
                "processed": 1,
                "skipped": 0,
                "errors": 0,
            },
        ]
    )
    monkeypatch.setattr(
        GroupRankHistoricalCalculator,
        "backfill_rankings",
        lambda self, *args, **kwargs: backfill(
            *args,
            **kwargs,
        ),
    )

    result = historical.backfill_rankings_chunked(
        MagicMock(),
        date(2026, 3, 18),
        date(2026, 3, 20),
        chunk_size_days=2,
        market="US",
    )

    assert result["total_dates"] == 3
    assert result["processed"] == 2
    assert result["skipped"] == 1


def test_historical_calculator_does_not_locate_calendar_or_services():
    source = Path(historical_module.__file__).read_text()
    assert "wiring.bootstrap" not in source
    assert "get_market_calendar_service" not in source
    assert "IBDGroupRankService" not in source
    assert "IBDGroupRank" not in source
