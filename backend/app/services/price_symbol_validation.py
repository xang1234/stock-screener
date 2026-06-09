"""Provider price-symbol validation helpers.

These helpers describe symbols that should never be sent to a price provider.
They intentionally do not normalize invalid symbols into a different ticker:
for JP, stripping a leading zero can resolve a different Yahoo instrument.
"""

from __future__ import annotations


YAHOO_ZERO_PREFIXED_JP_SYMBOL_ERROR = (
    "JP local code is zero-prefixed; no price data expected from Yahoo. "
    "Do not strip the leading zero because that can resolve a different security."
)


def is_zero_prefixed_jp_local_code(local_code: str | None) -> bool:
    token = str(local_code or "").strip().upper()
    if not token:
        return False
    numeric_part = token[:-1] if token[-1].isalpha() else token
    return numeric_part.startswith("0") and numeric_part.isdigit()


def _jp_local_code_from_yahoo_symbol(symbol: str | None) -> str | None:
    normalized = str(symbol or "").strip().upper()
    if normalized.endswith(".T"):
        return normalized[:-2]
    if normalized.endswith(".JP"):
        return normalized[:-3]
    return None


def yahoo_price_no_data_error_for_symbol(symbol: str | None) -> str | None:
    local_code = _jp_local_code_from_yahoo_symbol(symbol)
    if local_code is not None and is_zero_prefixed_jp_local_code(local_code):
        return YAHOO_ZERO_PREFIXED_JP_SYMBOL_ERROR
    return None
