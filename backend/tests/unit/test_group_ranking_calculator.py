from datetime import date
import inspect

import pandas as pd

from app.scanners.criteria.relative_strength import (
    RelativeStrengthCalculator,
)
from app.services.group_rank_models import (
    GroupRankPrefetchData,
    GroupRankPrefetchStats,
)
from app.services.group_ranking_calculator import GroupRankingCalculator


def _stats(symbol_count: int) -> GroupRankPrefetchStats:
    return GroupRankPrefetchStats(
        target_symbols=symbol_count,
        symbols_with_prices=symbol_count,
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
    )


def _trend_frame(daily_step: float) -> pd.DataFrame:
    index = pd.bdate_range(end="2026-03-20", periods=260)
    closes = [
        100.0 + daily_step * item
        for item in range(len(index))
    ]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [value + 1 for value in closes],
            "Low": [value - 1 for value in closes],
            "Close": closes,
            "Volume": 1_000_000,
        },
        index=index,
    )


def _trend_price_frame(
    *,
    periods: int = 260,
    start_close: float = 100.0,
    daily_return: float = 0.001,
) -> pd.DataFrame:
    dates = pd.date_range(
        end="2026-03-20",
        periods=periods,
        freq="B",
    )
    closes = [
        start_close * ((1 + daily_return) ** idx)
        for idx in range(periods)
    ]
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


def _prefetch_for_two_groups() -> GroupRankPrefetchData:
    benchmark = _trend_frame(0.2)
    prices = {
        "AAA": _trend_frame(0.6),
        "BBB": _trend_frame(0.55),
        "CCC": _trend_frame(0.5),
        "DDD": _trend_frame(0.15),
        "EEE": _trend_frame(0.1),
        "FFF": _trend_frame(0.05),
    }
    return GroupRankPrefetchData(
        benchmark_prices=benchmark,
        prices_by_symbol=prices,
        active_symbols=frozenset(prices),
        market_caps={symbol: 1_000_000 for symbol in prices},
        stats=_stats(len(prices)),
        symbols_by_group={
            "Software": ("AAA", "BBB", "CCC"),
            "Retail": ("DDD", "EEE", "FFF"),
        },
    )


def test_calculate_for_date_returns_ranked_groups_without_mutating_prefetch():
    prefetch = _prefetch_for_two_groups()
    calculator = GroupRankingCalculator(
        rs_calculator=RelativeStrengthCalculator()
    )

    rankings = calculator.calculate_for_date(
        prefetch=prefetch,
        group_names=("Software", "Retail"),
        calculation_date=date(2026, 3, 20),
    )

    assert [row.rank for row in rankings] == [1, 2]
    assert [row.industry_group for row in rankings] == [
        "Software",
        "Retail",
    ]
    assert rankings[0].avg_rs_rating >= rankings[1].avg_rs_rating
    assert prefetch.symbols_by_group == {
        "Software": ("AAA", "BBB", "CCC"),
        "Retail": ("DDD", "EEE", "FFF"),
    }


def test_calculator_performs_no_database_or_cache_reads():
    signature = inspect.signature(
        GroupRankingCalculator.calculate_for_date
    )
    assert "db" not in signature.parameters


def test_calculate_for_date_excludes_short_history():
    calculator = GroupRankingCalculator(
        rs_calculator=RelativeStrengthCalculator()
    )
    calculation_date = date(2026, 3, 20)
    symbols = ("AAA", "BBB", "CCC", "NEW")
    benchmark_prices = _trend_price_frame(daily_return=0.0004)
    prices_by_symbol = {
        "AAA": _trend_price_frame(
            start_close=100.0,
            daily_return=0.0020,
        ),
        "BBB": _trend_price_frame(
            start_close=80.0,
            daily_return=0.0011,
        ),
        "CCC": _trend_price_frame(
            start_close=120.0,
            daily_return=-0.0002,
        ),
        "NEW": _trend_price_frame(
            start_close=50.0,
            daily_return=0.0040,
            periods=200,
        ),
    }
    market_caps = {
        "AAA": 10_000_000_000,
        "BBB": 5_000_000_000,
        "CCC": 1_000_000_000,
        "NEW": 50_000_000_000,
    }
    prefetch = GroupRankPrefetchData(
        benchmark_prices=benchmark_prices,
        prices_by_symbol=prices_by_symbol,
        active_symbols=frozenset(symbols),
        market_caps=market_caps,
        stats=_stats(len(symbols)),
        symbols_by_group={"Software": symbols},
    )

    rankings = calculator.calculate_for_date(
        prefetch=prefetch,
        group_names=("Software",),
        calculation_date=calculation_date,
    )

    assert len(rankings) == 1
    assert rankings[0].industry_group == "Software"
    assert rankings[0].num_stocks == 3
    assert rankings[0].top_symbol == "AAA"


def test_calculate_for_date_preserves_invalid_period_return_semantics():
    calculator = GroupRankingCalculator(
        rs_calculator=RelativeStrengthCalculator()
    )
    calculation_date = date(2026, 3, 20)
    symbols = ("AAA", "BBB", "CCC", "DDD", "EEE")
    benchmark_prices = _trend_price_frame(daily_return=0.0004)
    prices_by_symbol = {
        "AAA": _trend_price_frame(
            start_close=100.0,
            daily_return=0.0020,
        ),
        "BBB": _trend_price_frame(
            start_close=80.0,
            daily_return=0.0011,
        ),
        "CCC": _trend_price_frame(
            start_close=120.0,
            daily_return=-0.0002,
        ),
        "DDD": _trend_price_frame(
            start_close=90.0,
            daily_return=0.0017,
        ),
        "EEE": _trend_price_frame(
            start_close=70.0,
            daily_return=0.0009,
        ),
    }
    prices_by_symbol["BBB"].iloc[
        -63,
        prices_by_symbol["BBB"].columns.get_loc("Close"),
    ] = 0.0
    prices_by_symbol["DDD"].iloc[
        -1,
        prices_by_symbol["DDD"].columns.get_loc("Close"),
    ] = float("nan")
    prices_by_symbol["EEE"].iloc[
        -63,
        prices_by_symbol["EEE"].columns.get_loc("Close"),
    ] = float("nan")
    market_caps = {
        "AAA": 10_000_000_000,
        "BBB": 5_000_000_000,
        "CCC": 1_000_000_000,
        "DDD": 7_000_000_000,
        "EEE": 2_000_000_000,
    }
    prefetch = GroupRankPrefetchData(
        benchmark_prices=benchmark_prices,
        prices_by_symbol=prices_by_symbol,
        active_symbols=frozenset(symbols),
        market_caps=market_caps,
        stats=_stats(len(symbols)),
        symbols_by_group={"Software": symbols},
    )

    rankings = calculator.calculate_for_date(
        prefetch=prefetch,
        group_names=("Software",),
        calculation_date=calculation_date,
    )

    assert len(rankings) == 1
    assert rankings[0].num_stocks == 4
    assert rankings[0].top_symbol == "AAA"


def test_calculator_has_no_second_group_ranking_implementation():
    assert not hasattr(
        GroupRankingCalculator,
        "_calculate_group_rs_from_cache",
    )
