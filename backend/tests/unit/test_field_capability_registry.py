"""Unit tests for screening-field capability registry."""

from __future__ import annotations

from app.services.field_capability_registry import (
    FALLBACK_BEHAVIOR_COMPUTED,
    FALLBACK_BEHAVIOR_FALLBACK,
    FALLBACK_BEHAVIOR_POLICY_EXCLUDED,
    FALLBACK_BEHAVIOR_PRIMARY,
    SOURCE_TECHNICALS,
    SUPPORT_STATE_COMPUTED,
    SUPPORT_STATE_PARTIAL,
    SUPPORT_STATE_SUPPORTED,
    SUPPORT_STATE_UNSUPPORTED,
    field_capability_registry,
)
from app.services.fundamentals_completeness import (
    field_source_map,
    field_tier_map,
    screening_fields,
)
from app.services.provider_routing_policy import (
    MARKET_HK,
    MARKET_JP,
    MARKET_TW,
    MARKET_US,
    POLICY_VERSION,
    PROVIDER_FINVIZ,
    PROVIDER_YFINANCE,
)


def _field_map(artifact: dict) -> dict:
    return {entry["field"]: entry for entry in artifact["fields"]}


def test_registry_is_versioned_and_shape_is_deterministic():
    artifact = field_capability_registry.artifact()

    assert artifact["registry_version"] == "2026.04.12.1"
    assert artifact["routing_policy_version"] == POLICY_VERSION
    assert artifact["markets"] == [MARKET_US, MARKET_HK, MARKET_JP, MARKET_TW]
    assert artifact["providers"] == ["finviz", "yfinance", "alphavantage", "technicals"]
    assert artifact["field_count"] == len(artifact["fields"])
    assert artifact["field_count"] == len(screening_fields())


def test_registry_enumerates_all_screening_fields_with_tier_and_source():
    artifact = field_capability_registry.artifact()
    by_field = _field_map(artifact)

    tiers = field_tier_map()
    sources = field_source_map()

    assert set(by_field.keys()) == set(screening_fields())
    for field in screening_fields():
        assert by_field[field]["tier"] == tiers[field]
        assert by_field[field]["canonical_source"] == sources[field]


def test_yfinance_core_field_is_partial_for_us_and_supported_for_asia():
    artifact = field_capability_registry.artifact()
    market_cap = _field_map(artifact)["market_cap"]["markets"]

    us = market_cap[MARKET_US]
    assert us["canonical_provider"] == PROVIDER_YFINANCE
    assert us["support_state"] == SUPPORT_STATE_PARTIAL
    assert us["fallback_behavior"] == FALLBACK_BEHAVIOR_FALLBACK
    assert us["canonical_provider_position"] == 1
    assert us["providers_before_canonical"] == [PROVIDER_FINVIZ]
    assert us["provider_states"][PROVIDER_YFINANCE] == SUPPORT_STATE_PARTIAL

    for market in (MARKET_HK, MARKET_JP, MARKET_TW):
        row = market_cap[market]
        assert row["canonical_provider"] == PROVIDER_YFINANCE
        assert row["support_state"] == SUPPORT_STATE_SUPPORTED
        assert row["fallback_behavior"] == FALLBACK_BEHAVIOR_PRIMARY
        assert row["canonical_provider_position"] == 0
        assert row["provider_states"][PROVIDER_YFINANCE] == SUPPORT_STATE_SUPPORTED


def test_finviz_only_field_is_unsupported_for_non_us_markets():
    artifact = field_capability_registry.artifact()
    short_interest = _field_map(artifact)["short_interest"]["markets"]

    us = short_interest[MARKET_US]
    assert us["canonical_provider"] == PROVIDER_FINVIZ
    assert us["support_state"] == SUPPORT_STATE_SUPPORTED
    assert us["fallback_behavior"] == FALLBACK_BEHAVIOR_PRIMARY
    assert us["canonical_provider_position"] == 0

    for market in (MARKET_HK, MARKET_JP, MARKET_TW):
        row = short_interest[market]
        assert row["canonical_provider"] == PROVIDER_FINVIZ
        assert row["support_state"] == SUPPORT_STATE_UNSUPPORTED
        assert row["fallback_behavior"] == FALLBACK_BEHAVIOR_POLICY_EXCLUDED
        assert row["canonical_provider_position"] is None
        assert row["provider_states"][PROVIDER_FINVIZ] == SUPPORT_STATE_UNSUPPORTED


def test_technical_fields_are_computed_for_all_markets():
    artifact = field_capability_registry.artifact()
    rsi = _field_map(artifact)["rsi_14"]["markets"]

    for market in (MARKET_US, MARKET_HK, MARKET_JP, MARKET_TW):
        row = rsi[market]
        assert row["canonical_provider"] == SOURCE_TECHNICALS
        assert row["support_state"] == SUPPORT_STATE_COMPUTED
        assert row["fallback_behavior"] == FALLBACK_BEHAVIOR_COMPUTED
        assert row["canonical_provider_position"] is None
        assert row["provider_states"][SOURCE_TECHNICALS] == SUPPORT_STATE_COMPUTED
        assert row["provider_states"][PROVIDER_FINVIZ] == SUPPORT_STATE_UNSUPPORTED
        assert row["provider_states"][PROVIDER_YFINANCE] == SUPPORT_STATE_UNSUPPORTED


def test_registry_artifact_is_repeatable():
    first = field_capability_registry.artifact()
    second = field_capability_registry.artifact()
    assert first == second
