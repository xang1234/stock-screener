"""Default scan profile market scoping."""

from __future__ import annotations

from app.domain.scanning.defaults import (
    get_bootstrap_scan_profile,
    get_default_scan_profile,
)


def test_default_scan_profile_remains_backward_compatible() -> None:
    profile = get_default_scan_profile()

    assert profile["universe"] == "all"
    assert "benchmark_symbol" not in profile


def test_market_default_scan_profiles_scope_universe_and_benchmark() -> None:
    from app.services.benchmark_registry_service import benchmark_registry

    for market in benchmark_registry.supported_markets():
        benchmark_symbol = benchmark_registry.get_primary_symbol(market)
        profile = get_default_scan_profile(market)

        assert profile["universe"] == f"market:{market}"
        assert profile["benchmark_symbol"] == benchmark_symbol
        assert profile["criteria"]["benchmark_symbol"] == benchmark_symbol


def test_bootstrap_scan_profile_keeps_market_scope_but_skips_setup_engine() -> None:
    default_profile = get_default_scan_profile("US")
    bootstrap_profile = get_bootstrap_scan_profile("US")

    assert "setup_engine" in default_profile["screeners"]
    assert "setup_engine" not in bootstrap_profile["screeners"]
    assert bootstrap_profile["universe"] == default_profile["universe"]
    assert bootstrap_profile["benchmark_symbol"] == default_profile["benchmark_symbol"]
    assert bootstrap_profile["criteria"] == default_profile["criteria"]
