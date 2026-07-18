from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

import app.services.group_rank_input_loader as group_loader_module
from app.services.derived_data_execution_policy import (
    resolve_derived_data_execution_policy,
)
from app.services.group_rank_input_loader import GroupRankInputLoader
from app.services.group_rank_models import (
    GroupRankPrefetchData,
    GroupRankPrefetchStats,
)


@dataclass
class FakeUniverseSource:
    symbols: tuple[str, ...]

    def active_symbols(self, db, market):
        return frozenset(self.symbols)


@dataclass
class FakeTaxonomySource:
    symbols_by_group: dict[str, tuple[str, ...]]

    def groups(self, db, market):
        return tuple(self.symbols_by_group)

    def symbols_for_group(self, db, group, market):
        return self.symbols_by_group[group]


@dataclass
class CountingTaxonomySource:
    symbols_by_group: dict[str, tuple[str, ...]]
    groups_calls: int = 0
    member_calls: list[str] = field(default_factory=list)

    def groups(self, db, market):
        self.groups_calls += 1
        return tuple(self.symbols_by_group)

    def symbols_for_group(self, db, group, market):
        self.member_calls.append(group)
        return self.symbols_by_group[group]


@dataclass
class FakeMarketCapSource:
    values: dict[str, float]

    def market_caps(self, db, symbols):
        return {
            symbol: self.values[symbol]
            for symbol in symbols
            if symbol in self.values
        }


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


def _policy(mode: str):
    return resolve_derived_data_execution_policy(
        execution_policy=mode,
        target_date=date(2026, 3, 20),
        current_date=date(2026, 3, 20),
    )


def _loader(
    *,
    price_cache,
    benchmark_cache,
    groups: dict[str, tuple[str, ...]],
    active: tuple[str, ...],
    market_caps: dict[str, float] | None = None,
    taxonomy_source=None,
) -> GroupRankInputLoader:
    return GroupRankInputLoader(
        price_cache=price_cache,
        benchmark_cache=benchmark_cache,
        universe_source=FakeUniverseSource(active),
        taxonomy_source=(
            taxonomy_source or FakeTaxonomySource(groups)
        ),
        market_cap_source=FakeMarketCapSource(market_caps or {}),
    )


def test_guarded_load_uses_only_cached_benchmark_and_stock_reads(db_session):
    benchmark_prices = _price_frame()
    stock_prices = _price_frame()
    price_cache = Mock()
    price_cache.get_cached_only_fresh.return_value = benchmark_prices
    price_cache.get_many_cached_only_fresh.return_value = {
        "AAPL": stock_prices,
    }
    price_cache.get_many.side_effect = AssertionError(
        "guarded group load must not call provider-capable stock reads"
    )
    benchmark_cache = Mock()
    benchmark_cache.get_benchmark_symbol.return_value = "SPY"
    benchmark_cache.get_benchmark_candidates.return_value = ["SPY"]
    benchmark_cache.get_benchmark_data.side_effect = AssertionError(
        "guarded group load must not call provider-capable benchmark reads"
    )
    loader = _loader(
        price_cache=price_cache,
        benchmark_cache=benchmark_cache,
        groups={"Software": ("AAPL",)},
        active=("AAPL",),
        market_caps={"AAPL": 1_000_000_000},
    )

    prefetch = loader.load(
        db_session,
        market="US",
        policy=_policy("refresh_guarded"),
    )

    assert prefetch.benchmark_prices is benchmark_prices
    assert prefetch.prices_by_symbol == {"AAPL": stock_prices}
    assert prefetch.symbols_by_group == {"Software": ("AAPL",)}
    assert prefetch.market_caps == {"AAPL": 1_000_000_000}
    assert prefetch.stats.cache_miss_symbols == 0
    assert prefetch.stats.benchmark_cached is True
    price_cache.get_many.assert_not_called()
    benchmark_cache.get_benchmark_data.assert_not_called()


def test_guarded_load_requires_target_session_for_benchmark(db_session):
    target = date(2026, 3, 20)
    price_cache = Mock()
    price_cache.get_cached_only_fresh.return_value = None
    benchmark_cache = Mock()
    benchmark_cache.get_benchmark_symbol.return_value = "SPY"
    benchmark_cache.get_benchmark_candidates.return_value = ["SPY"]
    loader = _loader(
        price_cache=price_cache,
        benchmark_cache=benchmark_cache,
        groups={"Software": ("AAPL",)},
        active=("AAPL",),
    )

    prefetch = loader.load(
        db_session,
        market="US",
        policy=_policy("refresh_guarded"),
        calculation_date=target,
    )

    assert prefetch.stats.benchmark_available is False
    price_cache.get_cached_only_fresh.assert_called_once_with(
        "SPY",
        period="2y",
        required_as_of_date=target,
    )


def test_guarded_load_requires_target_session_for_constituents(db_session):
    target = date(2026, 3, 20)
    benchmark_prices = _price_frame()
    price_cache = Mock()
    price_cache.get_cached_only_fresh.return_value = benchmark_prices
    price_cache.get_many_cached_only_fresh.return_value = {"AAPL": None}
    benchmark_cache = Mock()
    benchmark_cache.get_benchmark_symbol.return_value = "SPY"
    benchmark_cache.get_benchmark_candidates.return_value = ["SPY"]
    loader = _loader(
        price_cache=price_cache,
        benchmark_cache=benchmark_cache,
        groups={"Software": ("AAPL",)},
        active=("AAPL",),
    )

    prefetch = loader.load(
        db_session,
        market="US",
        policy=_policy("refresh_guarded"),
        calculation_date=target,
    )

    assert prefetch.stats.cache_miss_symbols == 1
    assert prefetch.stats.cache_miss_symbols_sample == ("AAPL",)
    price_cache.get_many_cached_only_fresh.assert_called_once_with(
        ["AAPL"],
        period="2y",
        required_as_of_date=target,
    )


def test_cache_only_load_uses_cached_fallback_benchmark(db_session):
    fallback_prices = _price_frame()
    stock_prices = _price_frame()
    price_cache = Mock()
    price_cache.get_cached_only_fresh.side_effect = [
        None,
        fallback_prices,
    ]
    price_cache.get_many_cached_only_fresh.return_value = {
        "SONY": stock_prices,
    }
    benchmark_cache = Mock()
    benchmark_cache.get_benchmark_symbol.return_value = "^N225"
    benchmark_cache.get_benchmark_candidates.return_value = [
        "^N225",
        "1306.T",
    ]
    loader = _loader(
        price_cache=price_cache,
        benchmark_cache=benchmark_cache,
        groups={"JP_Software": ("SONY",)},
        active=("SONY",),
    )

    prefetch = loader.load(
        db_session,
        market="JP",
        policy=_policy("strict_cache_only"),
    )

    assert prefetch.benchmark_prices is fallback_prices
    assert prefetch.stats.benchmark_symbol == "1306.T"
    assert prefetch.stats.benchmark_role == "fallback"


def test_historical_auto_load_uses_provider_capable_reads(db_session):
    benchmark_prices = _price_frame()
    stock_prices = _price_frame()
    price_cache = Mock()
    price_cache.get_many.return_value = {"AAPL": stock_prices}
    benchmark_cache = Mock()
    benchmark_cache.get_benchmark_symbol.return_value = "SPY"
    benchmark_cache.get_benchmark_data.return_value = benchmark_prices
    loader = _loader(
        price_cache=price_cache,
        benchmark_cache=benchmark_cache,
        groups={"Software": ("AAPL",)},
        active=("AAPL",),
    )
    policy = resolve_derived_data_execution_policy(
        execution_policy="auto",
        target_date=date(2026, 3, 19),
        current_date=date(2026, 3, 20),
    )

    prefetch = loader.load(db_session, market="US", policy=policy)

    assert prefetch.stats.cache_only is False
    benchmark_cache.get_benchmark_data.assert_called_once_with(
        market="US",
        period="2y",
    )
    price_cache.get_many.assert_called_once_with(["AAPL"], period="2y")
    price_cache.get_many_cached_only_fresh.assert_not_called()


def test_load_skips_unsupported_symbols_before_bulk_read(db_session):
    prices = _price_frame()
    price_cache = Mock()
    price_cache.get_many.return_value = {"AAPL": prices}
    benchmark_cache = Mock()
    benchmark_cache.get_benchmark_symbol.return_value = "SPY"
    benchmark_cache.get_benchmark_data.return_value = prices
    loader = _loader(
        price_cache=price_cache,
        benchmark_cache=benchmark_cache,
        groups={"Software": ("AAPL", "MZYX-U")},
        active=("AAPL", "MZYX-U"),
    )
    policy = resolve_derived_data_execution_policy(
        execution_policy="auto",
        target_date=date(2026, 3, 19),
        current_date=date(2026, 3, 20),
    )

    prefetch = loader.load(db_session, market="US", policy=policy)

    assert prefetch.symbols_by_group == {"Software": ("AAPL",)}
    assert prefetch.stats.skipped_unsupported_symbols == 1
    price_cache.get_many.assert_called_once_with(["AAPL"], period="2y")


def test_load_reads_each_taxonomy_membership_once(db_session):
    prices = _price_frame()
    taxonomy = CountingTaxonomySource(
        {
            "Software": ("AAA", "MZYX-U"),
            "Retail": ("BBB",),
        }
    )
    price_cache = Mock()
    price_cache.get_many.return_value = {
        "AAA": prices,
        "BBB": prices,
    }
    benchmark_cache = Mock()
    benchmark_cache.get_benchmark_symbol.return_value = "SPY"
    benchmark_cache.get_benchmark_data.return_value = prices
    loader = _loader(
        price_cache=price_cache,
        benchmark_cache=benchmark_cache,
        groups={},
        active=("AAA", "BBB", "MZYX-U"),
        taxonomy_source=taxonomy,
    )
    policy = resolve_derived_data_execution_policy(
        execution_policy="auto",
        target_date=date(2026, 3, 19),
        current_date=date(2026, 3, 20),
    )

    prefetch = loader.load(db_session, market="US", policy=policy)

    assert prefetch.group_names == ("Software", "Retail")
    assert taxonomy.groups_calls == 1
    assert taxonomy.member_calls == ["Software", "Retail"]
    assert prefetch.stats.skipped_unsupported_symbols == 1


def test_cache_only_missing_benchmark_returns_typed_empty_prefetch(db_session):
    price_cache = Mock()
    price_cache.get_cached_only_fresh.return_value = None
    benchmark_cache = Mock()
    benchmark_cache.get_benchmark_symbol.return_value = "^N225"
    benchmark_cache.get_benchmark_candidates.return_value = ["^N225"]
    loader = _loader(
        price_cache=price_cache,
        benchmark_cache=benchmark_cache,
        groups={"JP_Software": ("SONY",)},
        active=("SONY",),
    )

    prefetch = loader.load(
        db_session,
        market="JP",
        policy=_policy("refresh_guarded"),
    )

    assert prefetch.benchmark_prices is None
    assert prefetch.stats.benchmark_available is False
    assert prefetch.stats.benchmark_symbol == "^N225"
    assert prefetch.stats.market == "JP"
    assert prefetch.group_names == ("JP_Software",)


def test_complete_legacy_symbols_uses_typed_taxonomy_source(db_session):
    prices = _price_frame()
    loader = _loader(
        price_cache=Mock(),
        benchmark_cache=Mock(),
        groups={"Software": ("AAPL", "INACTIVE")},
        active=("AAPL",),
    )
    prefetch = GroupRankPrefetchData(
        benchmark_prices=prices,
        prices_by_symbol={"AAPL": prices},
        active_symbols=frozenset({"AAPL"}),
        market_caps={},
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
        symbols_by_group={},
        group_names=(),
    )

    completed = loader.complete_legacy_symbols(
        db_session,
        market="US",
        group_names=("Software",),
        prefetch=prefetch,
    )

    assert completed.symbols_by_group == {"Software": ("AAPL",)}


def test_loader_has_no_service_location_or_facade_callback():
    source = Path(group_loader_module.__file__).read_text()
    assert "wiring.bootstrap" not in source
    assert "IBDGroupRankService" not in source
    assert "getattr(" not in source
