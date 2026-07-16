import pandas as pd
import pytest

from app.services.group_rank_legacy_adapter import (
    LegacyGroupRankPrefetchAdapter,
)
from app.services.group_rank_models import (
    GroupRankPrefetchData,
    GroupRankPrefetchStats,
)


def _adapter_price_frame() -> pd.DataFrame:
    index = pd.bdate_range(end="2026-03-20", periods=260)
    return pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Volume": 1_000_000,
        },
        index=index,
    )


def _adapter_stats() -> GroupRankPrefetchStats:
    return GroupRankPrefetchStats(
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
    )


def _typed_prefetch() -> GroupRankPrefetchData:
    prices = _adapter_price_frame()
    return GroupRankPrefetchData(
        benchmark_prices=prices,
        prices_by_symbol={"AAA": prices},
        active_symbols=frozenset({"AAA"}),
        market_caps={"AAA": 1_000_000},
        stats=_adapter_stats(),
        symbols_by_group={"Software": ("AAA",)},
    )


def test_adapter_converts_legacy_five_tuple():
    prices = _adapter_price_frame()
    legacy = (
        prices,
        {"AAA": prices},
        {"AAA"},
        {"AAA": 1_000_000},
        {
            "target_symbols": 2,
            "symbols_with_prices": 1,
            "cache_miss_symbols": 1,
            "spy_cached": True,
        },
    )

    adapted = LegacyGroupRankPrefetchAdapter().adapt(legacy)

    assert adapted.active_symbols == frozenset({"AAA"})
    assert adapted.stats.cache_coverage_ratio == 0.5
    assert adapted.stats.benchmark_available is True
    assert adapted.symbols_by_group == {}


def test_adapter_returns_typed_prefetch_unchanged():
    prefetch = _typed_prefetch()

    assert LegacyGroupRankPrefetchAdapter().adapt(prefetch) is prefetch


def test_adapter_rejects_unknown_legacy_shape():
    with pytest.raises(TypeError, match="legacy group prefetch"):
        LegacyGroupRankPrefetchAdapter().adapt(("too", "short"))
