from __future__ import annotations

import pytest

from app.domain.markets.catalog import MarketCatalogError, get_market_catalog


def test_market_catalog_lists_supported_markets_in_runtime_order() -> None:
    catalog = get_market_catalog()

    assert catalog.supported_market_codes() == ["US", "HK", "IN", "JP", "KR", "TW", "CN", "CA", "DE"]


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


def test_market_catalog_rejects_unknown_market() -> None:
    catalog = get_market_catalog()

    with pytest.raises(MarketCatalogError, match="Unsupported market 'EU'"):
        catalog.get("EU")


def test_market_catalog_runtime_payload_is_frontend_ready() -> None:
    payload = get_market_catalog().as_runtime_payload()

    assert payload["version"] == "2026-05-09.v1"
    assert [market["code"] for market in payload["markets"]] == [
        "US",
        "HK",
        "IN",
        "JP",
        "KR",
        "TW",
        "CN",
        "CA",
        "DE",
    ]
    assert payload["markets"][0] == {
        "code": "US",
        "label": "United States",
        "currency": "USD",
        "timezone": "America/New_York",
        "calendar_id": "XNYS",
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
