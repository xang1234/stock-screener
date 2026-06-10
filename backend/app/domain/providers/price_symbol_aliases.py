"""Display-symbol aliases for price history provider lookups."""

from __future__ import annotations

YAHOO_PRICE_SYMBOL_ALIASES: dict[str, str] = {
    "TVC:DXY": "DX-Y.NYB",
    "FX:USDSGD": "SGD=X",
    "BITSTAMP:BTCUSD": "BTC-USD",
    "TVC:VIX": "^VIX",
}


def resolve_yahoo_price_symbol(symbol: str | None) -> str:
    normalized = str(symbol or "").strip().lstrip("$").upper()
    return YAHOO_PRICE_SYMBOL_ALIASES.get(normalized, normalized)
