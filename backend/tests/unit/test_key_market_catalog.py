from __future__ import annotations

from app.domain.markets.key_markets import (
    key_market_instruments,
    key_market_price_symbols,
    key_market_watchlist_defaults,
    resolve_key_market_price_symbol,
)


def test_key_market_catalog_separates_display_symbols_from_price_symbols() -> None:
    instruments = key_market_instruments("us")

    dxy = next(item for item in instruments if item.display_symbol == "TVC:DXY")

    assert dxy.display_name == "US Dollar Index"
    assert dxy.currency == "USD"
    assert dxy.price_symbol == "DX-Y.NYB"
    assert dxy.data_symbol == "DX-Y.NYB"
    assert resolve_key_market_price_symbol("FX:USDSGD") == "SGD=X"
    assert resolve_key_market_price_symbol("nvda") == "NVDA"


def test_key_market_catalog_provides_ui_watchlist_defaults() -> None:
    assert key_market_watchlist_defaults("US") == (
        {"symbol": "SPY", "display_name": "S&P 500 ETF"},
        {"symbol": "QQQ", "display_name": "Nasdaq 100 ETF"},
        {"symbol": "IWM", "display_name": "Russell 2000 ETF"},
        {"symbol": "TVC:DXY", "display_name": "US Dollar Index"},
        {"symbol": "FX:USDSGD", "display_name": "USD/SGD"},
        {"symbol": "BITSTAMP:BTCUSD", "display_name": "Bitcoin"},
        {"symbol": "GLD", "display_name": "Gold ETF"},
        {"symbol": "TLT", "display_name": "20+ Year Treasury ETF"},
        {"symbol": "TVC:VIX", "display_name": "Volatility Index"},
    )


def test_key_market_catalog_provides_static_refresh_price_symbols() -> None:
    assert key_market_price_symbols("US") == (
        "SPY",
        "QQQ",
        "IWM",
        "DX-Y.NYB",
        "SGD=X",
        "BTC-USD",
        "GLD",
        "TLT",
        "^VIX",
    )
