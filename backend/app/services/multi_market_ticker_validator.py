"""Multi-market ticker normalization + validation contract (T7.5).

Sits between the LLM's raw extraction output and the theme-mention
persistence layer. The contract here is the single source of truth for
"what does a valid ticker look like" — the extraction prompt must
document the same set of suffixes this module accepts, so prompt and
validator cannot silently drift.

Design
------
Normalization is a pure, deterministic, DB-free operation:

1. Try the T7.4 :func:`cjk_alias_resolver_service.resolve_alias` —
   handles canonical suffix passthrough, numeric-code normalization,
   and CJK/English company-name aliases.
2. Fall back to :meth:`SecurityMasterResolver.normalize_symbol` for
   US-shape tokens the CJK resolver doesn't recognize (``NVDA``,
   ``GOOGL``, etc.). This is just ``.strip().upper().lstrip("$")``.
3. Reject empty output so callers don't persist blank tickers.

The DB-dependent universe check remains on the caller
(``ThemeExtractionService._clean_tickers``) because the active
universe is pipeline state, not a property of the ticker string.

Drop-path logging
-----------------
Every rejection emits a structured ``logger.debug`` with a stable
``reason=...`` tag so operators can grep production logs for which
bucket is eating the most mentions (invalid shape vs. universe miss
vs. unresolvable CJK). No silent filters.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from .cjk_alias_resolver_service import METHOD_NONE, resolve_alias
from .security_master_service import (
    SecurityMasterResolver,
    security_master_resolver,
)

logger = logging.getLogger(__name__)

POLICY_VERSION: str = "2026.04.13.1"

# Shape regex — unchanged from the original ThemeExtractionService. It's
# permissive enough for US (NVDA), HK (0700.HK), JP (6758.T), TW (2330.TW),
# and TWO (2330.TWO) canonical forms. Max length 12 covers all canonical
# shapes with room for 5-digit HK/JP/TW codes.
TICKER_SHAPE_RE = re.compile(r"^[A-Z0-9][A-Z0-9\.\-]{0,11}$")

# Drop-reason tags used in logger.debug so ops can grep drop rates by bucket.
REASON_EMPTY = "empty_input"
REASON_FALSE_POSITIVE = "in_false_positive_list"
REASON_UNRESOLVABLE = "unresolvable_to_canonical"
REASON_SHAPE = "invalid_shape"
REASON_UNIVERSE_MISS = "not_in_active_universe"
REASON_UNIVERSE_EMPTY = "active_universe_empty"

_DROP_REASONS: frozenset[str] = frozenset({
    REASON_EMPTY, REASON_FALSE_POSITIVE, REASON_UNRESOLVABLE,
    REASON_SHAPE, REASON_UNIVERSE_MISS, REASON_UNIVERSE_EMPTY,
})


@dataclass(frozen=True)
class NormalizedTicker:
    """Result of normalizing a raw extracted ticker token.

    ``canonical`` is ``None`` when normalization failed — the caller
    MUST treat that as a drop with the recorded ``reason``. A non-None
    canonical is a shape-valid candidate; the caller still has to
    confirm universe membership.
    """

    raw: str
    canonical: Optional[str]
    reason: Optional[str]  # None on success; a REASON_* tag on drop


def normalize_extracted_ticker(
    raw: object,
    *,
    resolver: Optional[SecurityMasterResolver] = None,
) -> NormalizedTicker:
    """Normalize an LLM-extracted token to a canonical SecurityMaster symbol.

    ``raw`` is typed as ``object`` because LLM output is JSON and may
    contain non-string garbage (None, nested dicts). Non-string input
    is dropped with :data:`REASON_EMPTY`.

    The T7.4 CJK resolver handles:
    - ``"0700.HK"`` / ``"6758.T"`` / ``"2330.TW"`` → canonical passthrough
    - ``"Tencent"`` / ``"ソニー"`` / ``"台積電"`` → alias lookup
    - Halfwidth katakana, fullwidth digits, traditional/simplified Han

    For anything the CJK resolver returns ``METHOD_NONE`` on (plain US
    tickers, unknown company names), we fall back to the SecurityMaster
    normalizer — which just uppercases and strips a leading ``$``.
    That preserves the existing US behaviour so US-only deployments
    don't regress.
    """
    if not isinstance(raw, str):
        return NormalizedTicker(raw=str(raw), canonical=None, reason=REASON_EMPTY)
    stripped = raw.strip()
    if not stripped:
        return NormalizedTicker(raw=raw, canonical=None, reason=REASON_EMPTY)

    resolution = resolve_alias(stripped)
    if resolution.method != METHOD_NONE and resolution.canonical_symbol:
        return NormalizedTicker(
            raw=raw, canonical=resolution.canonical_symbol, reason=None,
        )

    sm = resolver or security_master_resolver
    fallback = sm.normalize_symbol(stripped)
    if not fallback:
        return NormalizedTicker(
            raw=raw, canonical=None, reason=REASON_UNRESOLVABLE,
        )
    return NormalizedTicker(raw=raw, canonical=fallback, reason=None)


def log_drop(
    *, raw: object, canonical: Optional[str], reason: str,
) -> None:
    """Emit a structured drop-path log line.

    Centralized so callers can't forget the ``reason=...`` tag and so
    the format stays stable for grep/ops dashboards. Unknown reason
    tags emit an additional warning so typos surface loudly — but the
    debug line is still recorded so the mention itself is traceable.
    """
    if reason not in _DROP_REASONS:
        logger.warning(
            "ticker_dropped: unknown reason tag %r (expected one of %s)",
            reason, sorted(_DROP_REASONS),
        )
    logger.debug(
        "ticker_dropped reason=%s raw=%r canonical=%r",
        reason, raw, canonical,
    )


def describe_policy() -> dict:
    """Stable snapshot for API / admin surfacing."""
    return {
        "policy_version": POLICY_VERSION,
        "accepted_suffixes": [".HK", ".T", ".TW", ".TWO"],
        "drop_reasons": sorted(_DROP_REASONS),
        "shape_regex": TICKER_SHAPE_RE.pattern,
    }


__all__ = [
    "POLICY_VERSION",
    "REASON_EMPTY",
    "REASON_FALSE_POSITIVE",
    "REASON_UNRESOLVABLE",
    "REASON_SHAPE",
    "REASON_UNIVERSE_MISS",
    "REASON_UNIVERSE_EMPTY",
    "NormalizedTicker",
    "normalize_extracted_ticker",
    "log_drop",
    "describe_policy",
]
