"""Shared symbol support rules for static/daily build paths."""

from __future__ import annotations

YAHOO_UNSUPPORTED_SUFFIXES = ("U", "UN", "UNT", "UNIT", "R", "RT")
YAHOO_UNSUPPORTED_PREFIXES = ("W", "WS", "WT")


def is_unsupported_yahoo_price_symbol(symbol: str) -> bool:
    """Return True for derivative-style suffixes that Yahoo price history often lacks."""
    normalized = (symbol or "").strip().upper()
    if not normalized:
        return False

    for delimiter in ("-", ".", "/"):
        if delimiter not in normalized:
            continue
        suffix = normalized.rsplit(delimiter, 1)[1]
        if suffix in YAHOO_UNSUPPORTED_SUFFIXES:
            return True
        if any(suffix.startswith(prefix) for prefix in YAHOO_UNSUPPORTED_PREFIXES):
            return True
    return False


def split_supported_price_symbols(symbols: list[str] | tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Partition symbols into Yahoo price-supported and unsupported buckets."""
    supported: list[str] = []
    unsupported: list[str] = []
    for symbol in symbols:
        if is_unsupported_yahoo_price_symbol(symbol):
            unsupported.append(symbol)
        else:
            supported.append(symbol)
    return supported, unsupported
