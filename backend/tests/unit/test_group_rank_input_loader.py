from datetime import date
from unittest.mock import Mock

import pandas as pd

from app.services.derived_data_execution_policy import (
    resolve_derived_data_execution_policy,
)
from app.services.group_rank_input_loader import GroupRankInputLoader


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


def _configure_universe(monkeypatch, *, groups, symbols, active):
    stock_universe_service = Mock()
    stock_universe_service.get_active_symbols.return_value = active
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_stock_universe_service",
        lambda: stock_universe_service,
    )
    monkeypatch.setattr(
        "app.services.group_rank_input_loader.IBDIndustryService.get_all_groups",
        lambda db, **kwargs: groups,
    )
    monkeypatch.setattr(
        "app.services.group_rank_input_loader.IBDIndustryService.get_group_symbols",
        lambda db, group, **kwargs: symbols[group],
    )


def test_guarded_load_uses_only_cached_benchmark_and_stock_reads(
    db_session,
    monkeypatch,
):
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
    _configure_universe(
        monkeypatch,
        groups=["Software"],
        symbols={"Software": ["AAPL"]},
        active=["AAPL"],
    )
    loader = GroupRankInputLoader(
        price_cache=price_cache,
        benchmark_cache=benchmark_cache,
    )
    monkeypatch.setattr(
        loader,
        "_market_caps",
        lambda db, symbols: {"AAPL": 1_000_000_000},
    )

    prefetch = loader.load(
        db_session,
        market="US",
        policy=_policy("refresh_guarded"),
    )

    assert prefetch.benchmark_prices is benchmark_prices
    assert prefetch.prices_by_symbol == {"AAPL": stock_prices}
    assert prefetch.symbols_by_group == {"Software": ("AAPL",)}
    assert prefetch.stats.cache_miss_symbols == 0
    assert prefetch.stats.benchmark_cached is True
    price_cache.get_many.assert_not_called()
    benchmark_cache.get_benchmark_data.assert_not_called()


def test_cache_only_load_uses_cached_fallback_benchmark(
    db_session,
    monkeypatch,
):
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
    _configure_universe(
        monkeypatch,
        groups=["JP_Software"],
        symbols={"JP_Software": ["SONY"]},
        active=["SONY"],
    )
    loader = GroupRankInputLoader(
        price_cache=price_cache,
        benchmark_cache=benchmark_cache,
    )
    monkeypatch.setattr(loader, "_market_caps", lambda db, symbols: {})

    prefetch = loader.load(
        db_session,
        market="JP",
        policy=_policy("strict_cache_only"),
    )

    assert prefetch.benchmark_prices is fallback_prices
    assert prefetch.stats.benchmark_symbol == "1306.T"
    assert prefetch.stats.benchmark_role == "fallback"


def test_historical_auto_load_uses_provider_capable_reads(
    db_session,
    monkeypatch,
):
    benchmark_prices = _price_frame()
    stock_prices = _price_frame()
    price_cache = Mock()
    price_cache.get_many.return_value = {"AAPL": stock_prices}
    benchmark_cache = Mock()
    benchmark_cache.get_benchmark_symbol.return_value = "SPY"
    benchmark_cache.get_benchmark_data.return_value = benchmark_prices
    _configure_universe(
        monkeypatch,
        groups=["Software"],
        symbols={"Software": ["AAPL"]},
        active=["AAPL"],
    )
    loader = GroupRankInputLoader(
        price_cache=price_cache,
        benchmark_cache=benchmark_cache,
    )
    monkeypatch.setattr(loader, "_market_caps", lambda db, symbols: {})
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


def test_load_skips_unsupported_symbols_before_bulk_read(
    db_session,
    monkeypatch,
):
    prices = _price_frame()
    price_cache = Mock()
    price_cache.get_many.return_value = {"AAPL": prices}
    benchmark_cache = Mock()
    benchmark_cache.get_benchmark_symbol.return_value = "SPY"
    benchmark_cache.get_benchmark_data.return_value = prices
    _configure_universe(
        monkeypatch,
        groups=["Software"],
        symbols={"Software": ["AAPL", "MZYX-U"]},
        active=["AAPL", "MZYX-U"],
    )
    loader = GroupRankInputLoader(
        price_cache=price_cache,
        benchmark_cache=benchmark_cache,
    )
    monkeypatch.setattr(loader, "_market_caps", lambda db, symbols: {})
    policy = resolve_derived_data_execution_policy(
        execution_policy="auto",
        target_date=date(2026, 3, 19),
        current_date=date(2026, 3, 20),
    )

    prefetch = loader.load(db_session, market="US", policy=policy)

    assert prefetch.symbols_by_group == {"Software": ("AAPL",)}
    assert prefetch.stats.skipped_unsupported_symbols == 1
    price_cache.get_many.assert_called_once_with(["AAPL"], period="2y")


def test_cache_only_missing_benchmark_returns_typed_empty_prefetch(
    db_session,
):
    price_cache = Mock()
    price_cache.get_cached_only_fresh.return_value = None
    benchmark_cache = Mock()
    benchmark_cache.get_benchmark_symbol.return_value = "^N225"
    benchmark_cache.get_benchmark_candidates.return_value = ["^N225"]
    loader = GroupRankInputLoader(
        price_cache=price_cache,
        benchmark_cache=benchmark_cache,
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
