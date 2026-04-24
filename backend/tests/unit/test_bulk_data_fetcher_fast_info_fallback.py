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
