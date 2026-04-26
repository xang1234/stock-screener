"""Default scan profile market scoping."""

from __future__ import annotations

from app.domain.scanning.defaults import get_default_scan_profile


def test_default_scan_profile_remains_backward_compatible() -> None:
    profile = get_default_scan_profile()

    assert profile["universe"] == "all"
    assert "benchmark_symbol" not in profile


def test_market_default_scan_profiles_scope_universe_and_benchmark() -> None:
    expected_benchmarks = {
        "US": "SPY",
        "HK": "^HSI",
        "IN": "^NSEI",
        "JP": "^N225",
        "TW": "^TWII",
    }

    for market, benchmark_symbol in expected_benchmarks.items():
        profile = get_default_scan_profile(market)

        assert profile["universe"] == f"market:{market}"
        assert profile["benchmark_symbol"] == benchmark_symbol
        assert profile["criteria"]["benchmark_symbol"] == benchmark_symbol

