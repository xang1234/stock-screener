"""Tests for the MarketRegistry."""

from __future__ import annotations

import pytest

from app.domain.markets import Market, SUPPORTED_MARKET_CODES
from app.domain.markets.registry import MarketProfile, MarketRegistry, market_registry


def test_every_supported_market_has_complete_profile() -> None:
    for code in SUPPORTED_MARKET_CODES:
        profile = market_registry.profile(Market(code))

        assert profile.market == Market(code)
        assert profile.label
        assert profile.currency
        assert profile.timezone_name
        assert profile.calendar_id
        assert profile.exchanges
        assert profile.indexes
        assert profile.primary_benchmark_symbol


def test_representative_profile_lookups_accept_string_and_market() -> None:
    assert market_registry.profile("hk").calendar_id == "XHKG"
    assert market_registry.profile(Market("IN")).timezone_name == "Asia/Kolkata"


def test_supported_codes_are_in_runtime_order() -> None:
    assert market_registry.supported_market_codes() == ("US", "HK", "IN", "JP", "KR", "TW", "CN", "CA", "DE")
    assert market_registry.supported_markets() == tuple(Market(code) for code in market_registry.supported_market_codes())


def test_market_for_index_uses_registry_mapping() -> None:
    assert market_registry.market_for_index("HSI") == Market("HK")
    assert market_registry.market_for_index("NIKKEI225") == Market("JP")
    assert market_registry.market_for_index("TAIEX") == Market("TW")
    assert market_registry.market_for_index("unknown") is None


def test_market_for_exchange_uses_registry_mapping() -> None:
    assert market_registry.market_for_exchange("XBOM") == Market("IN")
    assert market_registry.market_for_exchange("KOSPI") == Market("KR")
    assert market_registry.market_for_exchange("KOSDAQ") == Market("KR")
    assert market_registry.market_for_exchange("SZSE") == Market("CN")
    assert market_registry.market_for_exchange("unknown") is None


def test_custom_registry_rejects_duplicate_market_profiles() -> None:
    us = market_registry.profile("US")

    try:
        MarketRegistry((us, us))
    except ValueError as exc:
        assert "Duplicate market profile" in str(exc)
    else:
        raise AssertionError("Expected duplicate profile rejection")


def test_custom_registry_rejects_duplicate_exchange_and_index_aliases() -> None:
    us = market_registry.profile("US")
    hk = market_registry.profile("HK")

    duplicate_exchange = MarketProfile(
        market=hk.market,
        label=hk.label,
        currency=hk.currency,
        timezone_name=hk.timezone_name,
        calendar_id=hk.calendar_id,
        provider_calendar_id=hk.provider_calendar_id,
        exchanges=("NYSE",),
        indexes=hk.indexes,
        primary_benchmark_symbol=hk.primary_benchmark_symbol,
        benchmark_fallback_symbol=hk.benchmark_fallback_symbol,
        benchmark_primary_kind=hk.benchmark_primary_kind,
        benchmark_fallback_kind=hk.benchmark_fallback_kind,
    )
    with pytest.raises(ValueError, match="Duplicate exchange alias"):
        MarketRegistry((us, duplicate_exchange))

    duplicate_index = MarketProfile(
        market=hk.market,
        label=hk.label,
        currency=hk.currency,
        timezone_name=hk.timezone_name,
        calendar_id=hk.calendar_id,
        provider_calendar_id=hk.provider_calendar_id,
        exchanges=hk.exchanges,
        indexes=("SP500",),
        primary_benchmark_symbol=hk.primary_benchmark_symbol,
        benchmark_fallback_symbol=hk.benchmark_fallback_symbol,
        benchmark_primary_kind=hk.benchmark_primary_kind,
        benchmark_fallback_kind=hk.benchmark_fallback_kind,
    )
    with pytest.raises(ValueError, match="Duplicate index alias"):
        MarketRegistry((us, duplicate_index))


def test_profile_type_is_public_contract() -> None:
    assert isinstance(market_registry.profile("US"), MarketProfile)
