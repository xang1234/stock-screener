from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from app.services.group_rank_cache_policy import GroupRankCacheRequirement
from app.services.group_rank_models import (
    GroupRankPrefetchData,
    GroupRankPrefetchStats,
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


def test_prefetch_stats_can_coerce_legacy_test_mapping():
    stats = GroupRankPrefetchStats.from_mapping({
        "target_symbols": 2,
        "symbols_with_prices": 1,
        "cache_miss_symbols": 1,
        "spy_cached": True,
    })

    assert stats.cache_coverage_ratio == 0.5
    assert stats.benchmark_available is True
    assert stats.cache_miss_symbols_sample == ()


def test_prefetch_data_normalizes_legacy_container_values():
    prices = pd.DataFrame({"Close": [100.0]})

    prefetch = GroupRankPrefetchData(
        benchmark_prices=prices,
        prices_by_symbol={"AAA": prices},
        active_symbols={"AAA"},
        market_caps={"AAA": 1_000_000},
        stats={
            "target_symbols": 1,
            "symbols_with_prices": 1,
            "cache_miss_symbols": 0,
            "spy_cached": True,
        },
        symbols_by_group={"Software": ["AAA"]},
    )

    assert prefetch.active_symbols == frozenset({"AAA"})
    assert prefetch.symbols_by_group == {"Software": ("AAA",)}
    assert isinstance(prefetch.stats, GroupRankPrefetchStats)
