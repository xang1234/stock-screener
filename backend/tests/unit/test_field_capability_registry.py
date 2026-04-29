"""Unit tests for screening-field capability registry."""

from __future__ import annotations

from app.services.field_capability_registry import (
    FALLBACK_BEHAVIOR_COMPUTED,
    FALLBACK_BEHAVIOR_FALLBACK,
    FALLBACK_BEHAVIOR_POLICY_EXCLUDED,
    FALLBACK_BEHAVIOR_PRIMARY,
    REASON_CODE_MISSING_SUPPORTED,
    REASON_CODE_NON_US_GAP,
    REASON_CODE_POLICY_EXCLUDED,
    SOURCE_TECHNICALS,
    SUPPORT_STATE_AVAILABLE,
    SUPPORT_STATE_COMPUTED,
    SUPPORT_STATE_MISSING,
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
    MARKET_CN,
    MARKET_HK,
    MARKET_IN,
    MARKET_JP,
    MARKET_KR,
    MARKET_TW,
    MARKET_US,
    POLICY_VERSION,
    PROVIDER_FINVIZ,
    PROVIDER_AKSHARE,
    PROVIDER_BAOSTOCK,
    PROVIDER_KRX,
    PROVIDER_OPENDART,
    PROVIDER_YFINANCE,
)


def _field_map(artifact: dict) -> dict:
    return {entry["field"]: entry for entry in artifact["fields"]}


def test_registry_is_versioned_and_shape_is_deterministic():
    artifact = field_capability_registry.artifact()

    assert artifact["registry_version"] == "2026.04.30.1"
    assert artifact["routing_policy_version"] == POLICY_VERSION
    assert artifact["markets"] == [
        MARKET_US,
        MARKET_HK,
        MARKET_IN,
        MARKET_JP,
        MARKET_KR,
        MARKET_TW,
        MARKET_CN,
    ]
    assert artifact["providers"] == [
        "finviz",
        "akshare",
        "baostock",
        "krx",
        "opendart",
        "yfinance",
        "alphavantage",
        "technicals",
    ]
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
    # Finviz can also supply market_cap in US and is primary in the chain.
    assert us["provider_states"][PROVIDER_FINVIZ] == SUPPORT_STATE_SUPPORTED

    for market in (MARKET_HK, MARKET_IN, MARKET_JP, MARKET_TW):
        row = market_cap[market]
        assert row["canonical_provider"] == PROVIDER_YFINANCE
        assert row["support_state"] == SUPPORT_STATE_SUPPORTED
        assert row["fallback_behavior"] == FALLBACK_BEHAVIOR_PRIMARY
        assert row["canonical_provider_position"] == 0
        assert row["provider_states"][PROVIDER_YFINANCE] == SUPPORT_STATE_SUPPORTED
        assert row["provider_states"][PROVIDER_FINVIZ] == SUPPORT_STATE_UNSUPPORTED

    kr = market_cap[MARKET_KR]
    assert kr["policy_provider_chain"] == [PROVIDER_KRX, PROVIDER_OPENDART, PROVIDER_YFINANCE]
    assert kr["provider_states"][PROVIDER_KRX] == SUPPORT_STATE_SUPPORTED
    assert kr["provider_states"][PROVIDER_OPENDART] == SUPPORT_STATE_UNSUPPORTED
    assert kr["provider_states"][PROVIDER_YFINANCE] == SUPPORT_STATE_PARTIAL

    cn = market_cap[MARKET_CN]
    assert cn["policy_provider_chain"] == [PROVIDER_AKSHARE, PROVIDER_BAOSTOCK, PROVIDER_YFINANCE]
    assert cn["provider_states"][PROVIDER_AKSHARE] == SUPPORT_STATE_SUPPORTED
    assert cn["provider_states"][PROVIDER_BAOSTOCK] == SUPPORT_STATE_PARTIAL
    assert cn["provider_states"][PROVIDER_YFINANCE] == SUPPORT_STATE_PARTIAL


def test_finviz_only_field_is_unsupported_for_non_us_markets():
    artifact = field_capability_registry.artifact()
    short_interest = _field_map(artifact)["short_interest"]["markets"]

    us = short_interest[MARKET_US]
    assert us["canonical_provider"] == PROVIDER_FINVIZ
    assert us["support_state"] == SUPPORT_STATE_SUPPORTED
    assert us["fallback_behavior"] == FALLBACK_BEHAVIOR_PRIMARY
    assert us["canonical_provider_position"] == 0

    for market in (MARKET_HK, MARKET_IN, MARKET_JP, MARKET_KR, MARKET_TW, MARKET_CN):
        row = short_interest[market]
        assert row["canonical_provider"] == PROVIDER_FINVIZ
        assert row["support_state"] == SUPPORT_STATE_UNSUPPORTED
        assert row["fallback_behavior"] == FALLBACK_BEHAVIOR_POLICY_EXCLUDED
        assert row["canonical_provider_position"] is None
        assert row["provider_states"][PROVIDER_FINVIZ] == SUPPORT_STATE_UNSUPPORTED


def test_technical_fields_are_computed_for_all_markets():
    artifact = field_capability_registry.artifact()
    rsi = _field_map(artifact)["rsi_14"]["markets"]

    us = rsi[MARKET_US]
    assert us["canonical_provider"] == SOURCE_TECHNICALS
    assert us["support_state"] == SUPPORT_STATE_COMPUTED
    assert us["fallback_behavior"] == FALLBACK_BEHAVIOR_COMPUTED
    assert us["canonical_provider_position"] is None
    assert us["provider_states"][SOURCE_TECHNICALS] == SUPPORT_STATE_COMPUTED
    # Finviz can provide RSI in US but technicals remain canonical.
    assert us["provider_states"][PROVIDER_FINVIZ] == SUPPORT_STATE_SUPPORTED
    assert us["provider_states"][PROVIDER_YFINANCE] == SUPPORT_STATE_UNSUPPORTED

    for market in (MARKET_HK, MARKET_IN, MARKET_JP, MARKET_KR, MARKET_TW, MARKET_CN):
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


def test_auxiliary_scan_field_is_enumerated():
    artifact = field_capability_registry.artifact()
    first_trade_date = _field_map(artifact)["first_trade_date"]["markets"]
    assert _field_map(artifact)["first_trade_date"]["tier"] == "auxiliary"

    us = first_trade_date[MARKET_US]
    assert us["canonical_provider"] == PROVIDER_YFINANCE
    assert us["support_state"] == SUPPORT_STATE_PARTIAL
    assert us["provider_states"][PROVIDER_YFINANCE] == SUPPORT_STATE_PARTIAL
    assert us["provider_states"][PROVIDER_FINVIZ] == SUPPORT_STATE_UNSUPPORTED

    for market in (MARKET_HK, MARKET_IN, MARKET_JP, MARKET_TW):
        row = first_trade_date[market]
        assert row["support_state"] == SUPPORT_STATE_SUPPORTED
        assert row["provider_states"][PROVIDER_YFINANCE] == SUPPORT_STATE_SUPPORTED

    kr = first_trade_date[MARKET_KR]
    assert kr["support_state"] == SUPPORT_STATE_PARTIAL
    assert kr["providers_before_canonical"] == [PROVIDER_KRX, PROVIDER_OPENDART]
    assert kr["provider_states"][PROVIDER_YFINANCE] == SUPPORT_STATE_PARTIAL

    cn = first_trade_date[MARKET_CN]
    assert cn["support_state"] == SUPPORT_STATE_PARTIAL
    assert cn["providers_before_canonical"] == [PROVIDER_AKSHARE, PROVIDER_BAOSTOCK]
    assert cn["provider_states"][PROVIDER_YFINANCE] == SUPPORT_STATE_PARTIAL


def test_china_native_core_and_statement_fields_are_explicitly_supported():
    artifact = field_capability_registry.artifact()
    by_field = _field_map(artifact)

    for field in ("market_cap", "shares_outstanding", "pe_ratio", "price_to_book", "eps_current"):
        row = by_field[field]["markets"][MARKET_CN]
        assert row["provider_states"][PROVIDER_AKSHARE] == SUPPORT_STATE_SUPPORTED
        assert row["provider_states"][PROVIDER_BAOSTOCK] in {
            SUPPORT_STATE_SUPPORTED,
            SUPPORT_STATE_PARTIAL,
        }

    revenue = by_field["revenue_current"]["markets"][MARKET_CN]
    assert revenue["provider_states"][PROVIDER_AKSHARE] == SUPPORT_STATE_SUPPORTED
    assert revenue["provider_states"][PROVIDER_BAOSTOCK] == SUPPORT_STATE_PARTIAL


def test_korea_opendart_statement_fields_are_explicitly_supported():
    artifact = field_capability_registry.artifact()
    revenue = _field_map(artifact)["revenue_current"]["markets"][MARKET_KR]

    assert revenue["policy_provider_chain"] == [PROVIDER_KRX, PROVIDER_OPENDART, PROVIDER_YFINANCE]
    assert revenue["provider_states"][PROVIDER_OPENDART] == SUPPORT_STATE_SUPPORTED
    assert revenue["provider_states"][PROVIDER_YFINANCE] == SUPPORT_STATE_PARTIAL

    by_field = _field_map(artifact)
    for field in ("revenue_growth", "quick_ratio"):
        if field in by_field:
            row = by_field[field]["markets"][MARKET_KR]
            assert row["provider_states"][PROVIDER_OPENDART] == SUPPORT_STATE_UNSUPPORTED


def test_non_us_missing_ownership_fields_surface_explicit_reason_codes():
    availability = field_capability_registry.derive_ownership_sentiment_availability(
        data={},
        market=MARKET_HK,
    )
    assert availability["institutional_ownership"]["status"] == SUPPORT_STATE_UNSUPPORTED
    assert availability["institutional_ownership"]["reason_code"] == REASON_CODE_NON_US_GAP
    assert availability["insider_ownership"]["status"] == SUPPORT_STATE_UNSUPPORTED
    assert availability["insider_ownership"]["reason_code"] == REASON_CODE_NON_US_GAP
    assert availability["short_interest"]["status"] == SUPPORT_STATE_UNSUPPORTED
    assert availability["short_interest"]["reason_code"] == REASON_CODE_POLICY_EXCLUDED


def test_present_non_us_ownership_field_is_marked_available():
    availability = field_capability_registry.derive_ownership_sentiment_availability(
        data={"institutional_ownership": 42.5},
        market=MARKET_JP,
    )
    assert availability["institutional_ownership"]["status"] == SUPPORT_STATE_AVAILABLE
    assert availability["institutional_ownership"]["reason_code"] is None


def test_us_missing_supported_ownership_field_is_marked_missing():
    availability = field_capability_registry.derive_ownership_sentiment_availability(
        data={},
        market=MARKET_US,
    )
    assert availability["institutional_ownership"]["status"] == SUPPORT_STATE_MISSING
    assert availability["institutional_ownership"]["reason_code"] == REASON_CODE_MISSING_SUPPORTED
