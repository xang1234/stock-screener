"""Tests for the Market value object."""

from __future__ import annotations

import dataclasses

import pytest

from app.domain.markets.market import Market, SUPPORTED_MARKET_CODES, UnsupportedMarketError


def test_market_from_str_normalizes_case_and_whitespace() -> None:
    assert Market.from_str(" hk ") == Market("HK")


def test_market_is_immutable_hashable_and_renders_code() -> None:
    market = Market("HK")

    assert {market, Market("HK"), Market("US")} == {Market("HK"), Market("US")}
    assert str(market) == "HK"
    assert "HK" in repr(market)
    with pytest.raises(dataclasses.FrozenInstanceError):
        market.code = "US"  # type: ignore[misc]


@pytest.mark.parametrize("raw", ["hk", "HK ", "", "EU"])
def test_market_constructor_requires_canonical_supported_code(raw: str) -> None:
    with pytest.raises(UnsupportedMarketError):
        Market(raw)


@pytest.mark.parametrize("raw", [None, "", "  ", "EU"])
def test_market_from_str_rejects_missing_or_unknown_market(raw: str | None) -> None:
    with pytest.raises(UnsupportedMarketError):
        Market.from_str(raw)


def test_market_from_str_rejects_non_string() -> None:
    with pytest.raises(UnsupportedMarketError):
        Market.from_str(123)


def test_supported_market_codes_are_current_supported_markets() -> None:
    assert SUPPORTED_MARKET_CODES == frozenset({"US", "HK", "IN", "JP", "KR", "TW", "CN", "SG", "CA", "DE"})
