from __future__ import annotations

import pytest

from app.domain.markets import (
    MarketCapabilities,
    MarketCatalogEntry,
    MicFacts,
    market_registry,
)
from app.domain.markets.catalog import MarketCatalogError, get_market_catalog
from app.domain.universe.indexes import index_registry


def test_market_catalog_lists_supported_markets_in_runtime_order() -> None:
    catalog = get_market_catalog()

    assert catalog.supported_market_codes() == list(market_registry.supported_market_codes())


def test_market_catalog_entry_contains_stable_market_facts() -> None:
    catalog = get_market_catalog()

    hk = catalog.get("hk")

    assert hk.code == "HK"
    assert hk.label == "Hong Kong"
    assert hk.currency == "HKD"
    assert hk.timezone == "Asia/Hong_Kong"
    assert hk.calendar_id == "XHKG"
    assert hk.exchanges == ("HKEX", "SEHK", "XHKG")
    assert hk.indexes == ("HSI",)
    assert hk.capabilities.official_universe is True
    assert hk.capabilities.finviz_screening is False


def test_au_market_catalog_entry_is_harmonized() -> None:
    entry = get_market_catalog().get("AU")

    assert entry.label == "Australia"
    assert entry.primary_mic == "XASX"
    assert entry.mics == ("XASX",)
    assert entry.supported_currencies == ("AUD",)
    assert entry.default_currency == "AUD"
    assert entry.timezone == "Australia/Sydney"
    assert "ASX" in entry.exchanges
    assert "XASX" in entry.exchanges
    assert entry.capabilities.benchmark is True
    assert entry.capabilities.breadth is False
    assert entry.capabilities.fundamentals is True
    assert entry.capabilities.group_rankings is False
    assert entry.capabilities.feature_snapshot is True
    assert entry.capabilities.official_universe is True
    assert entry.capabilities.finviz_screening is False


def test_market_catalog_index_summaries_derive_from_index_registry() -> None:
    catalog = get_market_catalog()

    for market in catalog.supported_market_codes():
        assert catalog.get(market).indexes == tuple(
            definition.key for definition in index_registry.definitions(market)
        )


def test_market_catalog_filters_market_codes_by_capability_in_runtime_order() -> None:
    catalog = get_market_catalog()

    assert catalog.market_codes_with_capability("breadth") == (
        "US",
        "HK",
        "IN",
        "JP",
        "KR",
        "TW",
        "CN",
        "CA",
        "DE",
    )
    assert catalog.market_codes_with_capability("group_rankings") == (
        "US",
        "HK",
        "IN",
        "JP",
        "KR",
        "TW",
        "CN",
        "CA",
    )


def test_market_catalog_rejects_unknown_capability_filter() -> None:
    catalog = get_market_catalog()

    with pytest.raises(MarketCatalogError, match="Unsupported market capability"):
        catalog.market_codes_with_capability("not_a_capability")


def test_market_catalog_entry_exposes_canonical_mic_and_currency_facts() -> None:
    catalog = get_market_catalog()

    us = catalog.get("US")
    india = catalog.get("IN")

    assert us.primary_mic == "XNYS"
    assert us.mics == ("XNYS", "XNAS", "XASE")
    assert us.supported_currencies == ("USD",)
    assert us.default_currency == "USD"
    assert us.currency == "USD"  # Deprecated compatibility alias.
    assert us.primary_mic_facts.calendar_id == "XNYS"
    assert us.primary_mic_facts.timezone == "America/New_York"
    assert us.primary_mic_facts.default_currency == "USD"
    assert us.mic_facts_for("XNAS").timezone == "America/New_York"

    assert india.primary_mic == "XNSE"
    assert india.mics == ("XNSE", "XBOM")
    assert india.supported_currencies == ("INR",)
    assert india.primary_mic_facts.provider_calendar_id == "NSE"
    assert india.mic_facts_for("XBOM").calendar_id == "XBOM"


def test_market_catalog_entry_rejects_mic_facts_outside_declared_mics() -> None:
    with pytest.raises(ValueError, match="MIC facts must match mics"):
        MarketCatalogEntry(
            code="XX",
            label="Example",
            primary_mic="XAAA",
            mics=("XAAA",),
            supported_currencies=("USD",),
            default_currency="USD",
            mic_facts=(
                MicFacts(
                    mic="XAAA",
                    calendar_id="XAAA",
                    timezone="America/New_York",
                    default_currency="USD",
                ),
                MicFacts(
                    mic="XBBB",
                    calendar_id="XBBB",
                    timezone="America/Toronto",
                    default_currency="USD",
                ),
            ),
            exchanges=("XAAA",),
            capabilities=MarketCapabilities(
                benchmark=False,
                breadth=False,
                fundamentals=False,
                group_rankings=False,
                feature_snapshot=False,
                official_universe=False,
                finviz_screening=False,
            ),
        )


def test_market_catalog_entry_rejects_mic_fact_currency_outside_supported_currencies() -> None:
    with pytest.raises(ValueError, match="MIC default currencies"):
        MarketCatalogEntry(
            code="XX",
            label="Example",
            primary_mic="XAAA",
            mics=("XAAA", "XBBB"),
            supported_currencies=("USD",),
            default_currency="USD",
            mic_facts=(
                MicFacts(
                    mic="XAAA",
                    calendar_id="XAAA",
                    timezone="America/New_York",
                    default_currency="USD",
                ),
                MicFacts(
                    mic="XBBB",
                    calendar_id="XBBB",
                    timezone="America/Toronto",
                    default_currency="CAD",
                ),
            ),
            exchanges=("XAAA", "XBBB"),
            capabilities=MarketCapabilities(
                benchmark=False,
                breadth=False,
                fundamentals=False,
                group_rankings=False,
                feature_snapshot=False,
                official_universe=False,
                finviz_screening=False,
            ),
        )


def test_market_catalog_rejects_unknown_market() -> None:
    catalog = get_market_catalog()

    with pytest.raises(MarketCatalogError, match="Unsupported market 'EU'"):
        catalog.get("EU")


def test_market_catalog_runtime_payload_is_frontend_ready() -> None:
    payload = get_market_catalog().as_runtime_payload()

    assert payload["version"] == "2026-05-17.v1"
    assert [market["code"] for market in payload["markets"]] == list(
        market_registry.supported_market_codes()
    )
    assert payload["markets"][0] == {
        "code": "US",
        "label": "United States",
        "primary_mic": "XNYS",
        "mics": ["XNYS", "XNAS", "XASE"],
        "supported_currencies": ["USD"],
        "default_currency": "USD",
        "mic_facts": [
            {
                "mic": "XNYS",
                "calendar_id": "XNYS",
                "timezone": "America/New_York",
                "default_currency": "USD",
                "provider_calendar_id": None,
            },
            {
                "mic": "XNAS",
                "calendar_id": "XNAS",
                "timezone": "America/New_York",
                "default_currency": "USD",
                "provider_calendar_id": None,
            },
            {
                "mic": "XASE",
                "calendar_id": "XASE",
                "timezone": "America/New_York",
                "default_currency": "USD",
                "provider_calendar_id": None,
            },
        ],
        "currency": "USD",
        "timezone": "America/New_York",
        "calendar_id": "XNYS",
        "provider_calendar_id": None,
        "exchanges": ["NYSE", "NASDAQ", "AMEX"],
        "indexes": ["SP500"],
        "capabilities": {
            "benchmark": True,
            "breadth": True,
            "fundamentals": True,
            "group_rankings": True,
            "feature_snapshot": True,
            "official_universe": False,
            "finviz_screening": True,
        },
    }
