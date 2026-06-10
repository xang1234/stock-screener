"""Shared symbol-format contract for user-facing stock endpoints.

One canonical regex for validating stock symbols across watchlist CRUD,
search, stock detail endpoints, and CSV imports. Matches the DB schema
width of ``stock_universe.symbol`` (``VARCHAR(20)``), so a format-valid
symbol is always persistable.

Accepts US-shape tokens (``NVDA``, ``GOOGL``) alongside the suffixed
non-US forms supported today: ``.HK`` (Hong Kong), ``.NS``/``.BO``
(India), ``.T`` (Tokyo), ``.KS``/``.KQ`` (Korea), ``.TW`` /
``.TWO`` (Taiwan), ``.SS``/``.SZ``/``.BJ`` (mainland China),
``.SI`` (Singapore), and ``.DE`` / ``.F`` (Germany — Xetra and Frankfurt).
The regex itself doesn't enforce the
suffix list — ``SUPPORTED_SUFFIXES`` is the policy lookup, and
universe-membership is the authoritative existence check (see
``StockUniverse``).

Theme extraction (``multi_market_ticker_validator``) intentionally uses
a stricter 12-character cap because LLM output rarely needs longer
tickers; keeping it narrow there limits hallucination blast radius.
"""

from __future__ import annotations

import re
from typing import Final

from app.domain.providers.price_symbol_aliases import resolve_yahoo_price_symbol

# Primary subtag must be alphanumeric; remainder allows alphanumeric + '.' + '-'
# up to 19 characters (total max 20, matches stock_universe.symbol VARCHAR(20)).
SYMBOL_SHAPE_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,19}$")

# Known exchange suffixes. Extending to a new market (e.g. '.KS' for KRX)
# requires adding entries here, in the market-ingestion adapters, and in
# the Market/Exchange enums in schemas/universe.py.
SUPPORTED_SUFFIXES: Final[tuple[str, ...]] = (
    ".HK",
    ".NS",
    ".BO",
    ".T",
    ".KS",
    ".KQ",
    ".TW",
    ".TWO",
    ".SS",
    ".SZ",
    ".BJ",
    ".SI",
    ".DE",
    ".F",
)

_MAX_SYMBOL_LEN: Final[int] = 20


def is_valid_symbol_shape(symbol: str | None) -> bool:
    """Return True when ``symbol`` matches the canonical shape regex.

    Does NOT check universe membership — use ``StockUniverseService`` for
    that. This is a pure syntactic guard suitable for early 422 rejection
    on API path params and request bodies.
    """
    if symbol is None:
        return False
    return bool(SYMBOL_SHAPE_RE.match(symbol))


def normalize_symbol(symbol: str | None) -> str | None:
    """Strip, uppercase, and validate shape. Returns None on rejection.

    Uppercasing is critical because the regex is uppercase-only and the DB
    stores symbols in uppercase. Leading ``$`` (cashtag prefix from social
    media) is stripped to match :class:`SecurityMasterResolver` semantics.
    """
    if symbol is None:
        return None
    cleaned = symbol.strip().lstrip("$").upper()
    if not cleaned or len(cleaned) > _MAX_SYMBOL_LEN:
        return None
    if not SYMBOL_SHAPE_RE.match(cleaned):
        return None
    return cleaned


def normalize_price_history_symbol(symbol: str | None) -> str | None:
    """Normalize symbols accepted by cache-backed price-history endpoints.

    Daily Snapshot key-market cards store TradingView display symbols for chart
    compatibility, while cached OHLCV rows use Yahoo symbols. Resolve those
    known aliases first, then fall back to the canonical stock-symbol shape.
    """
    if symbol is None:
        return None
    cleaned = symbol.strip().lstrip("$").upper()
    if not cleaned:
        return None
    resolved = resolve_yahoo_price_symbol(cleaned)
    if resolved != cleaned:
        return resolved
    return normalize_symbol(cleaned)


def require_valid_symbol(symbol: str) -> str:
    """Validate + normalize a path-param symbol or raise HTTP 422.

    Use as a FastAPI dependency for ``/{symbol}`` routes:

        async def get_stock_info(symbol: str = Depends(require_valid_symbol)):
            ...

    422 (not 404) is the correct status for a malformed path param — 404
    is reserved for "valid shape, doesn't exist".
    """
    from fastapi import HTTPException  # local import: keeps module DI-free

    normalized = normalize_symbol(symbol)
    if normalized is None:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid symbol format: {symbol!r}",
        )
    return normalized


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
