from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from app.services.group_rank_cache_policy import GroupRankCacheRequirement
from app.services.group_rank_models import (
    GroupRankCalculationResult,
    GroupRankPrefetchData,
    GroupRankPrefetchStats,
    GroupRanking,
)


def _stats():
    return GroupRankPrefetchStats(
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


def test_prefetch_stats_are_immutable():
    stats = _stats()

    with pytest.raises(FrozenInstanceError):
        stats.cache_miss_symbols = 2


def test_cache_requirement_returns_replaced_stats():
    stats = _stats()
    replaced = stats.with_cache_requirement(
        GroupRankCacheRequirement.minimum(0.8, reason="test"),
    )

    assert stats.cache_coverage_min is None
    assert replaced.cache_coverage_min == 0.8
    assert replaced.cache_requirement_reason == "test"


def test_prefetch_stats_preserve_external_dictionary_keys():
    assert _stats().to_dict() == {
        "target_symbols": 4,
        "symbols_with_prices": 3,
        "cache_miss_symbols": 1,
        "cache_miss_symbols_sample": ["MISS"],
        "cache_coverage_ratio": 0.75,
        "spy_cached": True,
        "benchmark_cached": True,
        "benchmark_symbol": "SPY",
        "benchmark_role": "primary",
        "market": "US",
        "cache_only": True,
        "skipped_unsupported_symbols": 0,
    }


def test_prefetch_data_rejects_legacy_stats_mapping():
    with pytest.raises(TypeError, match="GroupRankPrefetchStats"):
        GroupRankPrefetchData(
            benchmark_prices=pd.DataFrame({"Close": [100.0]}),
            prices_by_symbol={},
            active_symbols=frozenset(),
            market_caps={},
            stats={"spy_cached": True},
            symbols_by_group={},
        )


def test_prefetch_data_rejects_mutable_group_symbol_lists():
    with pytest.raises(TypeError, match="tuple"):
        GroupRankPrefetchData(
            benchmark_prices=None,
            prices_by_symbol={},
            active_symbols=frozenset(),
            market_caps={},
            stats=_stats(),
            symbols_by_group={"Software": ["AAA"]},
        )


def test_calculation_result_contains_immutable_group_rankings():
    ranking = GroupRanking(
        industry_group="Software",
        date=pd.Timestamp("2026-03-20").date(),
        rank=1,
        avg_rs_rating=88.0,
        median_rs_rating=87.0,
        weighted_avg_rs_rating=89.0,
        rs_std_dev=3.0,
        num_stocks=12,
        num_stocks_rs_above_80=8,
        top_symbol="AAA",
        top_rs_rating=96.0,
    )

    result = GroupRankCalculationResult(
        rankings=(ranking,),
        prefetch_stats=_stats(),
    )

    assert result.rankings == (ranking,)
    with pytest.raises(FrozenInstanceError):
        ranking.rank = 2
