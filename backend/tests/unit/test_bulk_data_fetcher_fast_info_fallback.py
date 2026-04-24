from __future__ import annotations

from types import SimpleNamespace

from app.services.bulk_data_fetcher import BulkDataFetcher


def _make_ticker(*, market_cap=None, shares=None, last_price=None):
    fast = SimpleNamespace(
        market_cap=market_cap,
        shares=shares,
        last_price=last_price,
    )
    return SimpleNamespace(fast_info=fast)


def test_fast_info_fills_market_cap_when_info_is_truncated():
    bdf = BulkDataFetcher()
    ticker = _make_ticker(market_cap=4_014_264_025_088.0, shares=14_685_000_000, last_price=273.42)
    result = bdf._extract_fundamentals(ticker, {"symbol": "AAPL"})

    assert result["market_cap"] == 4_014_264_025_088
    assert result["shares_outstanding"] == 14_685_000_000
    assert result["current_price"] == 273.42


def test_info_value_wins_over_fast_info():
    bdf = BulkDataFetcher()
    ticker = _make_ticker(market_cap=111, shares=222, last_price=3.14)
    result = bdf._extract_fundamentals(ticker, {"marketCap": 999_999, "sharesOutstanding": 10, "currentPrice": 50.0})

    assert result["market_cap"] == 999_999
    assert result["shares_outstanding"] == 10
    assert result["current_price"] == 50.0


def test_missing_from_both_filters_key_out():
    bdf = BulkDataFetcher()
    ticker = _make_ticker()  # all None
    result = bdf._extract_fundamentals(ticker, {})
    assert "market_cap" not in result
    assert "shares_outstanding" not in result
    assert "current_price" not in result


def test_none_ticker_is_safe():
    bdf = BulkDataFetcher()
    # Prior signature only took info; confirm None ticker still works
    result = bdf._extract_fundamentals(None, {"marketCap": 42})
    assert result["market_cap"] == 42


class _NaN:
    """Non-numeric sentinel that raises on int()/round()."""

    def __float__(self):
        raise ValueError("NaN")

    def __int__(self):
        raise ValueError("NaN")

    def __round__(self, _ndigits=None):
        raise ValueError("NaN")


def test_read_fast_info_parses_fields_independently():
    """A malformed single field must not discard valid values for the others."""
    fast = SimpleNamespace(market_cap=_NaN(), shares=500, last_price=42.5)
    ticker = SimpleNamespace(fast_info=fast)

    market_cap, shares, last_price = (
        BulkDataFetcher._read_fast_info_market_state(ticker)
    )

    assert market_cap is None
    assert shares == 500
    assert last_price == 42.5


def test_extract_fundamentals_rejects_nan_from_info_and_falls_back():
    """NaN from .info must not survive as market_cap / shares / current_price;
    it should be treated as missing and trigger the fast_info fallback."""
    bdf = BulkDataFetcher()
    ticker = _make_ticker(market_cap=7_777, shares=42, last_price=1.25)
    result = bdf._extract_fundamentals(
        ticker,
        {
            "marketCap": float("nan"),
            "sharesOutstanding": float("nan"),
            "currentPrice": float("nan"),
            "regularMarketPrice": float("nan"),
        },
    )
    assert result["market_cap"] == 7_777
    assert result["shares_outstanding"] == 42
    assert result["current_price"] == 1.25


def test_extract_fundamentals_drops_nan_when_no_fallback():
    """With a NaN from .info and no fast_info value, the key should be
    filtered out (not stored as NaN) so downstream FX / JSON code is safe."""
    bdf = BulkDataFetcher()
    ticker = _make_ticker()  # all None
    result = bdf._extract_fundamentals(ticker, {"marketCap": float("nan")})
    assert "market_cap" not in result


def test_read_fast_info_rejects_nan_and_inf_floats():
    """float('nan') survives float() silently; must be rejected so it can't
    poison JSON serialization or DB comparisons downstream."""
    fast = SimpleNamespace(market_cap=1_000, shares=10, last_price=float("nan"))
    ticker = SimpleNamespace(fast_info=fast)
    _, _, last_price = BulkDataFetcher._read_fast_info_market_state(ticker)
    assert last_price is None

    fast2 = SimpleNamespace(market_cap=1_000, shares=10, last_price=float("inf"))
    ticker2 = SimpleNamespace(fast_info=fast2)
    _, _, last_price2 = BulkDataFetcher._read_fast_info_market_state(ticker2)
    assert last_price2 is None


def test_zero_from_info_is_preserved_not_replaced_by_fast_info():
    """If .info legitimately reports 0 for market_cap / shares_outstanding,
    keep the 0 instead of falling through to fast_info (truthiness would
    drop the 0 and — combined with _assign_if_present — leave stale prior
    values in the DB)."""
    bdf = BulkDataFetcher()
    ticker = _make_ticker(market_cap=123_456_789, shares=999, last_price=5.0)
    result = bdf._extract_fundamentals(
        ticker,
        {"marketCap": 0, "sharesOutstanding": 0},
    )
    assert result["market_cap"] == 0
    assert result["shares_outstanding"] == 0
