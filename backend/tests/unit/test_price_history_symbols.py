from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.services.price_history_symbols import (
    normalize_price_history_symbol,
    require_valid_price_history_symbol,
)


@pytest.mark.parametrize(
    ("display_symbol", "price_symbol"),
    [
        ("TVC:DXY", "DX-Y.NYB"),
        ("FX:USDSGD", "SGD=X"),
        ("TVC:VIX", "^VIX"),
        ("BITSTAMP:BTCUSD", "BTC-USD"),
    ],
)
def test_price_history_symbols_resolve_key_market_display_symbols(
    display_symbol: str,
    price_symbol: str,
) -> None:
    assert normalize_price_history_symbol(display_symbol) == price_symbol


def test_price_history_symbols_accept_canonical_stock_symbols() -> None:
    assert normalize_price_history_symbol("  nvda  ") == "NVDA"


def test_price_history_symbols_reject_unknown_tradingview_shapes() -> None:
    assert normalize_price_history_symbol("TVC:UNKNOWN") is None
    with pytest.raises(HTTPException):
        require_valid_price_history_symbol("TVC:UNKNOWN")
