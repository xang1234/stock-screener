"""Unit tests for the shared symbol-format contract.

Covers the pure regex/normalizer and the FastAPI path-param validator.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.services.symbol_format import (
    SUPPORTED_SUFFIXES,
    SYMBOL_SHAPE_RE,
    is_valid_symbol_shape,
    normalize_symbol,
    require_valid_symbol,
)


class TestSymbolShapeRe:
    @pytest.mark.parametrize("symbol", [
        "NVDA",
        "GOOGL",
        "BRK.B",
        "0700.HK",
        "6758.T",
        "2330.TW",
        "2330.TWO",
        "A",
        "1",
        "X" * 20,  # exactly at max length
    ])
    def test_accepts_valid_shapes(self, symbol: str):
        assert SYMBOL_SHAPE_RE.match(symbol), f"{symbol!r} should match"

    @pytest.mark.parametrize("symbol", [
        "",
        ".HK",
        "-NVDA",
        "nvda",                 # lowercase not accepted (caller must normalize)
        "X" * 21,               # over DB schema width
        "NVDA ",                # trailing space
        "NV DA",                # internal space
        "NVDA$",                # cashtag in wrong position
        "NVDA;DROP TABLE",      # SQL injection shape
        "日経",                  # non-ASCII
    ])
    def test_rejects_invalid_shapes(self, symbol: str):
        assert not SYMBOL_SHAPE_RE.match(symbol), f"{symbol!r} should not match"


class TestIsValidSymbolShape:
    def test_none_is_invalid(self):
        assert is_valid_symbol_shape(None) is False

    def test_happy_path(self):
        assert is_valid_symbol_shape("0700.HK") is True


class TestNormalizeSymbol:
    def test_strips_and_uppercases(self):
        assert normalize_symbol("  nvda  ") == "NVDA"

    def test_strips_leading_cashtag(self):
        assert normalize_symbol("$AAPL") == "AAPL"

    def test_non_us_suffixed_tickers(self):
        # Every suffix in SUPPORTED_SUFFIXES must round-trip through the
        # normalizer; this pins the policy list against the regex.
        for suffix in SUPPORTED_SUFFIXES:
            candidate = f"7203{suffix}"
            assert normalize_symbol(candidate) == candidate.upper()

    def test_returns_none_for_empty(self):
        assert normalize_symbol("") is None
        assert normalize_symbol("   ") is None
        assert normalize_symbol("$") is None  # cashtag alone isn't a symbol

    def test_returns_none_for_malformed(self):
        assert normalize_symbol("NV DA") is None
        assert normalize_symbol("a" * 21) is None  # exceeds schema width
        assert normalize_symbol(None) is None


class TestRequireValidSymbol:
    def test_returns_normalized_symbol(self):
        assert require_valid_symbol("0700.hk") == "0700.HK"

    def test_raises_http_422_on_malformed(self):
        with pytest.raises(HTTPException) as exc_info:
            require_valid_symbol("NV DA")
        assert exc_info.value.status_code == 422
        assert "Invalid symbol format" in exc_info.value.detail

    def test_raises_http_422_on_empty(self):
        with pytest.raises(HTTPException) as exc_info:
            require_valid_symbol("")
        assert exc_info.value.status_code == 422
