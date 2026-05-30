"""Tests for the MarketRegistry."""

from __future__ import annotations

import pytest

from app.domain.markets import Market, SUPPORTED_MARKET_CODES
from app.domain.markets.catalog import get_market_catalog
from app.domain.markets.registry import BenchmarkFacts, MarketProfile, MarketRegistry, market_registry
from app.domain.universe.indexes import index_registry


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


def test_market_registry_identity_facts_derive_from_market_catalog() -> None:
    catalog = get_market_catalog()

    for code in catalog.supported_market_codes():
        entry = catalog.get(code)
        profile = market_registry.profile(code)

        assert profile.label == entry.label
        assert profile.currency == entry.default_currency
        assert profile.timezone_name == entry.display_timezone
        assert profile.calendar_id == entry.calendar_id
        assert profile.provider_calendar_id == entry.provider_calendar_id
        assert profile.exchanges == entry.exchanges


def test_supported_codes_are_in_runtime_order() -> None:
    assert market_registry.supported_market_codes() == tuple(
        get_market_catalog().supported_market_codes()
    )
    assert market_registry.supported_markets() == tuple(Market(code) for code in market_registry.supported_market_codes())


def test_catalog_registry_factory_rejects_missing_benchmark_facts() -> None:
    with pytest.raises(ValueError, match="Benchmark facts are missing"):
        MarketRegistry.from_catalog(
            get_market_catalog(),
            benchmark_facts={
                "US": BenchmarkFacts("SPY", "IVV", "etf", "etf"),
            },
        )


def test_market_for_index_uses_registry_mapping() -> None:
    assert market_registry.market_for_index("HSI") == Market("HK")
    assert market_registry.market_for_index("NIFTY50") == Market("IN")
    assert market_registry.market_for_index("NIKKEI225") == Market("JP")
    assert market_registry.market_for_index("TAIEX") == Market("TW")
    assert market_registry.market_for_index("CSI300") == Market("CN")
    assert market_registry.market_for_index("ASX200") == Market("AU")
    assert market_registry.market_for_index("FBMKLCI") == Market("MY")
    assert market_registry.market_for_index("unknown") is None


def test_market_registry_index_lookups_delegate_to_index_registry() -> None:
    for index_key in index_registry.supported_index_keys():
        assert market_registry.market_for_index(index_key) == Market(
            index_registry.market_for(index_key)
        )


def test_market_for_exchange_uses_registry_mapping() -> None:
    assert market_registry.market_for_exchange("XBOM") == Market("IN")
    assert market_registry.market_for_exchange("KOSPI") == Market("KR")
    assert market_registry.market_for_exchange("KOSDAQ") == Market("KR")
    assert market_registry.market_for_exchange("SZSE") == Market("CN")
    assert market_registry.market_for_exchange("ASX") == Market("AU")
    assert market_registry.market_for_exchange("KLSE") == Market("MY")
    assert market_registry.market_for_exchange("BSE") is None
    assert market_registry.market_for_exchange("unknown") is None


def test_mic_for_exchange_requires_market_context_for_ambiguous_aliases() -> None:
    assert market_registry.mic_for_exchange("IN", "BSE") == "XBOM"
    assert market_registry.mic_for_exchange("CN", "BSE") == "XBSE"
    assert market_registry.mic_for_exchange("HK", "SEHK") == "XHKG"
    assert market_registry.mic_for_exchange("AU", "ASX") == "XASX"
    assert market_registry.mic_for_exchange("MY", "KLSE") == "XKLS"
    assert market_registry.mic_for_exchange("US", "unknown") is None


def test_au_benchmark_facts_are_registered() -> None:
    profile = market_registry.profile("AU")

    assert profile.primary_benchmark_symbol == "^AXJO"
    assert profile.benchmark_fallback_symbol == "IOZ.AX"
    assert profile.benchmark_primary_kind == "index"
    assert profile.benchmark_fallback_kind == "etf"


def test_custom_registry_rejects_duplicate_market_profiles() -> None:
    us = market_registry.profile("US")

    try:
        MarketRegistry((us, us))
    except ValueError as exc:
        assert "Duplicate market profile" in str(exc)
    else:
        raise AssertionError("Expected duplicate profile rejection")


def test_custom_registry_does_not_own_exchange_alias_uniqueness() -> None:
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
        primary_benchmark_symbol=hk.primary_benchmark_symbol,
        benchmark_fallback_symbol=hk.benchmark_fallback_symbol,
        benchmark_primary_kind=hk.benchmark_primary_kind,
        benchmark_fallback_kind=hk.benchmark_fallback_kind,
    )
    registry = MarketRegistry((us, duplicate_exchange))

    assert registry.profile("HK").exchanges == ("NYSE",)


def test_profile_type_is_public_contract() -> None:
    assert isinstance(market_registry.profile("US"), MarketProfile)
