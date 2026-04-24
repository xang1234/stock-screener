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
