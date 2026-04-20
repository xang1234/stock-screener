"""
Provider Routing Policy
=======================

Explicit, versioned, market-aware routing matrix for fundamentals providers.

Context
-------
Fundamentals providers have different geographic coverage:

- ``finviz``       : US-only (NYSE/NASDAQ/AMEX screener only).
- ``alphavantage`` : US-only free tier; overseas coverage is paid/limited and
  is not currently wired up in the hot path.
- ``yfinance``     : Global — supports local-suffix symbols (``.HK``, ``.T``,
  ``.TW``/``.TWO``) via the canonical symbols produced by
  ``SecurityMasterService``.

Prior to this module all symbols were routed identically (finviz preferred,
yfinance fallback). For non-US markets (HK/JP/TW) this produced:

* failed provider calls that wasted rate-limit quota,
* noisy rejection metrics that masked real regressions,
* harder incident diagnosis because the "first failure" was always finviz.

Policy shape
------------
The policy is a versioned mapping ``{market -> ordered provider tuple}``:

- Providers appear in *priority order*; the first entry is the preferred
  primary and subsequent entries are deterministic fallbacks.
- A provider absent from the tuple is *not attempted* for that market — this
  is the "unsupported provider call" the policy exists to prevent.

Callers
-------
- ``DataSourceService`` (single-symbol path): consults ``providers_for(market)``
  to decide whether to try finviz or skip straight to yfinance.
- ``HybridFundamentalsService`` (batch path): filters symbols per phase — the
  finviz Phase 3 loop skips symbols whose market excludes finviz.
- ``FundamentalsCacheService._fetch_and_cache``: resolves market from the
  stock universe so the single-symbol task chain is market-aware end-to-end.

Extending the policy
--------------------
To add a new market or provider, bump ``POLICY_VERSION`` and update
``_POLICY_MATRIX``. Keep this module the single source of truth — do NOT
introduce per-caller routing logic.
"""
from __future__ import annotations

import logging
from typing import FrozenSet, Mapping, Tuple

logger = logging.getLogger(__name__)


# --- Policy version ---------------------------------------------------------

POLICY_VERSION = "2026.04.21.1"
"""Bump (date-stamped) when routing semantics change.

Consumed by audit logs, cache keys, and provenance tags so downstream
consumers can detect a routing-semantics change and react accordingly.
"""


# --- Known providers --------------------------------------------------------

PROVIDER_FINVIZ = "finviz"
PROVIDER_YFINANCE = "yfinance"
PROVIDER_ALPHAVANTAGE = "alphavantage"

KNOWN_PROVIDERS: FrozenSet[str] = frozenset(
    {PROVIDER_FINVIZ, PROVIDER_YFINANCE, PROVIDER_ALPHAVANTAGE}
)


# --- Known markets ----------------------------------------------------------

MARKET_US = "US"
MARKET_HK = "HK"
MARKET_IN = "IN"
MARKET_JP = "JP"
MARKET_TW = "TW"

KNOWN_MARKETS: FrozenSet[str] = frozenset(
    {MARKET_US, MARKET_HK, MARKET_IN, MARKET_JP, MARKET_TW}
)

DEFAULT_MARKET = MARKET_US
"""Market used when the caller's value is ``None``/empty.

Defaulting to US preserves legacy behaviour for call sites that have not yet
been threaded with market context — they continue to see the existing
finviz -> yfinance -> alphavantage chain.
"""


# --- Policy matrix ----------------------------------------------------------

_POLICY_MATRIX: Mapping[str, Tuple[str, ...]] = {
    # US: finviz primary, yfinance fallback, alphavantage tertiary.
    # Matches the legacy DataSourceService ordering so default callers are
    # unaffected.
    MARKET_US: (PROVIDER_FINVIZ, PROVIDER_YFINANCE, PROVIDER_ALPHAVANTAGE),
    # HK / JP / TW: yfinance only. finviz screener is US-only and the
    # alphavantage free tier does not cover these markets.
    MARKET_HK: (PROVIDER_YFINANCE,),
    # IN: yfinance only. BSE-only .BO listings are admitted into the active
    # universe only after India ingest verifies Yahoo price coverage and skips
    # symbols with repeated unresolved Yahoo validation failures.
    MARKET_IN: (PROVIDER_YFINANCE,),
    MARKET_JP: (PROVIDER_YFINANCE,),
    MARKET_TW: (PROVIDER_YFINANCE,),
}


# --- Public API -------------------------------------------------------------

def normalize_market(market: str | None) -> str:
    """Return a canonical market string, defaulting to US for unknowns.

    - ``None`` / empty / whitespace -> ``DEFAULT_MARKET``.
    - Unknown (not in ``KNOWN_MARKETS``) -> ``DEFAULT_MARKET`` + warn log.
    - Known value (any case) -> canonical uppercase form.
    """
    if market is None:
        return DEFAULT_MARKET
    if not isinstance(market, str):
        logger.warning(
            "Non-string market %r (type %s) - falling back to %s policy "
            "(policy version %s).",
            market, type(market).__name__, DEFAULT_MARKET, POLICY_VERSION,
        )
        return DEFAULT_MARKET
    candidate = market.strip().upper()
    if not candidate:
        return DEFAULT_MARKET
    if candidate not in KNOWN_MARKETS:
        logger.warning(
            "Unknown market %r - falling back to %s policy "
            "(policy version %s).",
            market, DEFAULT_MARKET, POLICY_VERSION,
        )
        return DEFAULT_MARKET
    return candidate


def providers_for(market: str | None) -> Tuple[str, ...]:
    """Return the ordered provider tuple for ``market``.

    The first entry is the preferred primary; subsequent entries are
    deterministic fallbacks. Providers missing from the returned tuple
    MUST NOT be attempted for this market.
    """
    if isinstance(market, str):
        candidate = market.strip().upper()
        if candidate and candidate not in KNOWN_MARKETS:
            logger.warning(
                "Unknown market %r - failing closed with no providers "
                "(policy version %s).",
                market, POLICY_VERSION,
            )
            return ()
    return _POLICY_MATRIX[normalize_market(market)]


def is_supported(market: str | None, provider: str) -> bool:
    """Return True iff ``provider`` is allowed for ``market`` per policy.

    Unknown providers return False so callers fail closed rather than
    silently attempting an unvetted path.
    """
    if provider not in KNOWN_PROVIDERS:
        return False
    return provider in providers_for(market)


def supported_markets() -> Tuple[str, ...]:
    """Return the sorted tuple of markets the matrix has explicit rules for."""
    return tuple(sorted(_POLICY_MATRIX.keys()))


def policy_version() -> str:
    """Return the policy version string (for audit logs / cache keys)."""
    return POLICY_VERSION
