"""Symbol normalization for cache-backed price-history endpoints."""

from __future__ import annotations

from app.domain.markets.key_markets import key_market_data_symbol_for_display
from app.services.symbol_format import normalize_symbol


def normalize_price_history_symbol(symbol: str | None) -> str | None:
    """Normalize symbols accepted by cache-backed price-history endpoints.

    Daily Snapshot key-market cards store TradingView display symbols for chart
    compatibility, while cached OHLCV rows use provider symbols. Resolve known
    key-market display symbols at this boundary, then fall back to the stock
    symbol contract for ordinary tickers.
    """
    if symbol is None:
        return None
    cleaned = symbol.strip().lstrip("$").upper()
    if not cleaned:
        return None
    resolved = key_market_data_symbol_for_display(cleaned)
    if resolved is not None:
        return resolved
    return normalize_symbol(cleaned)


def require_valid_price_history_symbol(symbol: str) -> str:
    """Validate + normalize a path-param symbol for price-history reads."""
    from fastapi import HTTPException  # local import: keeps module DI-free

    normalized = normalize_price_history_symbol(symbol)
    if normalized is None:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid symbol format: {symbol!r}",
        )
    return normalized
