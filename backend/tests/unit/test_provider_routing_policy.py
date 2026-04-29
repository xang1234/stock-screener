"""Unit tests for the fundamentals provider routing policy."""
from __future__ import annotations

import logging

import pytest

from app.services import provider_routing_policy as policy
from app.services.provider_routing_policy import (
    MARKET_HK,
    MARKET_CN,
    MARKET_IN,
    MARKET_JP,
    MARKET_KR,
    MARKET_TW,
    MARKET_US,
    POLICY_VERSION,
    PROVIDER_ALPHAVANTAGE,
    PROVIDER_AKSHARE,
    PROVIDER_BAOSTOCK,
    PROVIDER_FINVIZ,
    PROVIDER_KRX,
    PROVIDER_OPENDART,
    PROVIDER_YFINANCE,
    is_supported,
    normalize_market,
    policy_version,
    providers_for,
    supported_markets,
)


class TestPolicyVersion:
    """The policy version string is part of the contract — pin it."""

    def test_policy_version_is_date_stamped(self):
        # Format: YYYY.MM.DD.N — bump when routing semantics change.
        assert POLICY_VERSION == "2026.04.30.1"
        assert policy_version() == POLICY_VERSION


class TestMatrixShape:
    """The matrix must cover every known market, explicitly."""

    def test_supported_markets_covers_us_and_asia(self):
        assert supported_markets() == (
            MARKET_CN,
            MARKET_HK,
            MARKET_IN,
            MARKET_JP,
            MARKET_KR,
            MARKET_TW,
            MARKET_US,
        )

    def test_every_known_market_has_a_policy(self):
        for market in policy.KNOWN_MARKETS:
            # providers_for must never raise for a known market.
            providers = providers_for(market)
            assert isinstance(providers, tuple)
            assert len(providers) >= 1, f"Market {market!r} has an empty policy"


class TestUSPolicy:
    """US preserves the legacy finviz -> yfinance -> alphavantage chain."""

    def test_us_provider_order(self):
        assert providers_for(MARKET_US) == (
            PROVIDER_FINVIZ,
            PROVIDER_YFINANCE,
            PROVIDER_ALPHAVANTAGE,
        )

    def test_us_supports_all_known_providers(self):
        for provider in (PROVIDER_FINVIZ, PROVIDER_YFINANCE, PROVIDER_ALPHAVANTAGE):
            assert is_supported(MARKET_US, provider) is True


class TestAsiaPolicy:
    """HK/IN/JP/TW route to yfinance only; KR has native providers first."""

    @pytest.mark.parametrize("market", [MARKET_HK, MARKET_IN, MARKET_JP, MARKET_TW])
    def test_asia_markets_are_yfinance_only(self, market):
        assert providers_for(market) == (PROVIDER_YFINANCE,)

    @pytest.mark.parametrize("market", [MARKET_HK, MARKET_IN, MARKET_JP, MARKET_TW])
    def test_asia_markets_reject_finviz(self, market):
        assert is_supported(market, PROVIDER_FINVIZ) is False

    @pytest.mark.parametrize("market", [MARKET_HK, MARKET_IN, MARKET_JP, MARKET_TW])
    def test_asia_markets_reject_alphavantage(self, market):
        assert is_supported(market, PROVIDER_ALPHAVANTAGE) is False

    @pytest.mark.parametrize("market", [MARKET_HK, MARKET_IN, MARKET_JP, MARKET_TW])
    def test_asia_markets_allow_yfinance(self, market):
        assert is_supported(market, PROVIDER_YFINANCE) is True

    def test_korea_provider_order(self):
        assert providers_for(MARKET_KR) == (
            PROVIDER_KRX,
            PROVIDER_OPENDART,
            PROVIDER_YFINANCE,
        )

    @pytest.mark.parametrize("provider", [PROVIDER_KRX, PROVIDER_OPENDART, PROVIDER_YFINANCE])
    def test_korea_allows_native_providers_and_yfinance_fallback(self, provider):
        assert is_supported(MARKET_KR, provider) is True

    @pytest.mark.parametrize("provider", [PROVIDER_FINVIZ, PROVIDER_ALPHAVANTAGE])
    def test_korea_rejects_us_only_providers(self, provider):
        assert is_supported(MARKET_KR, provider) is False

    def test_china_provider_order(self):
        assert providers_for(MARKET_CN) == (
            PROVIDER_AKSHARE,
            PROVIDER_BAOSTOCK,
            PROVIDER_YFINANCE,
        )

    @pytest.mark.parametrize("provider", [PROVIDER_AKSHARE, PROVIDER_BAOSTOCK, PROVIDER_YFINANCE])
    def test_china_allows_no_key_native_providers_and_yfinance_fallback(self, provider):
        assert is_supported(MARKET_CN, provider) is True

    @pytest.mark.parametrize("provider", [PROVIDER_FINVIZ, PROVIDER_ALPHAVANTAGE])
    def test_china_rejects_us_only_providers(self, provider):
        assert is_supported(MARKET_CN, provider) is False


class TestNormalization:
    """Canonical market resolution for messy caller inputs."""

    def test_none_defaults_to_us(self):
        assert normalize_market(None) == MARKET_US

    def test_empty_string_defaults_to_us(self):
        assert normalize_market("") == MARKET_US

    def test_whitespace_defaults_to_us(self):
        assert normalize_market("   ") == MARKET_US

    def test_lowercase_is_canonicalized(self):
        assert normalize_market("hk") == MARKET_HK
        assert normalize_market("in") == MARKET_IN
        assert normalize_market("jp") == MARKET_JP
        assert normalize_market("kr") == MARKET_KR
        assert normalize_market("cn") == MARKET_CN

    def test_mixed_case_with_padding(self):
        assert normalize_market("  Tw ") == MARKET_TW

    def test_unknown_market_defaults_to_us_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger=policy.__name__):
            result = normalize_market("XX")
        assert result == MARKET_US
        assert any("Unknown market" in rec.message for rec in caplog.records)

    def test_non_string_market_defaults_to_us_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger=policy.__name__):
            result = normalize_market(123)  # type: ignore[arg-type]
        assert result == MARKET_US
        assert any("Non-string market" in rec.message for rec in caplog.records)


class TestFailClosed:
    """Unknown providers must not be silently approved."""

    def test_is_supported_rejects_unknown_provider(self):
        assert is_supported(MARKET_US, "quandl") is False
        assert is_supported(MARKET_HK, "") is False

    def test_none_market_uses_us_policy(self):
        # Legacy callers passing market=None continue to get the US chain.
        assert providers_for(None) == (
            PROVIDER_FINVIZ,
            PROVIDER_YFINANCE,
            PROVIDER_ALPHAVANTAGE,
        )

    def test_unknown_non_empty_market_has_no_providers(self):
        assert providers_for("XX") == ()

    def test_unknown_non_empty_market_rejects_known_provider(self):
        assert is_supported("XX", PROVIDER_FINVIZ) is False


class TestDeterminism:
    """The same input always produces the same output — no accidental state."""

    def test_repeated_calls_return_equal_tuples(self):
        first = providers_for(MARKET_HK)
        second = providers_for(MARKET_HK)
        assert first == second
        # Tuples so callers can't mutate the matrix by accident.
        assert isinstance(first, tuple)
