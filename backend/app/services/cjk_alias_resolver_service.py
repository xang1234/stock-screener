"""Deterministic CJK company/ticker alias resolver (T7.4).

Rule/alias-first, LLM-second — this module is the deterministic pre-pass
that sits between the multilingual extraction stage (T7.2/T7.3) and any
LLM-based ticker inference downstream (T7.5). Its contract is narrow:

Given a raw token that may be a company name in CJK or English, a local
exchange code, or an already-canonical symbol, return a canonical
SecurityMaster symbol **only if a deterministic rule or alias dictionary
entry matches**. Otherwise return ``METHOD_NONE`` so the caller can
decide whether to fall through to an LLM (which is more expensive and
non-deterministic).

Resolution precedence (highest → lowest)
----------------------------------------
1. ``symbol_passthrough`` — input already carries a SecurityMaster
   suffix (``.HK``, ``.T``, ``.TW``, ``.TWO``, ``.SS``, ``.SZ``,
   ``.BJ``) and a valid local code.
2. ``symbol_normalized`` — input is a bare exchange code (e.g. ``700``
   with ``hint_market="HK"``) that we can reshape to canonical form.
3. ``alias_exact`` — NFKC-normalized input matches an alias verbatim.
4. ``alias_folded`` — aggressive key (casefold + punctuation strip)
   matches an alias under the same transform.
5. ``METHOD_NONE`` — unresolved. Caller falls through to LLM.

NFKC does the heavy lifting for CJK variant normalization (halfwidth
katakana ``ｿﾆｰ`` → fullwidth ``ソニー``; fullwidth digits ``７００`` →
``700``). Simplified vs. Traditional Han (``腾讯`` vs. ``騰訊``) is NOT
covered by NFKC; both variants must be listed explicitly in
:mod:`app.services.cjk_alias_data`.

Policy version
--------------
:data:`POLICY_VERSION` must bump whenever either the normalization
rules or the alias corpus change. Callers persisting resolutions
should store the policy version so stale hits can be invalidated when
the resolver evolves.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .cjk_alias_data import ASIA_ALIAS_CORPUS
# Re-use the authoritative suffix↔market tables from SecurityMaster so the
# two modules can't silently drift when a new exchange is added. Ordering
# of _MARKET_BY_SUFFIX is defensive: longest/most-specific suffix first.
# With today's four entries (.HK/.TWO/.TW/.T) the final characters are
# all distinct, so order is not strictly required — but a future addition
# like ``.AT`` would collide with ``.T`` on endswith(), and putting it
# first preserves correctness without special-casing.
from .security_master_service import _MARKET_BY_SUFFIX, _SUFFIX_BY_MARKET

POLICY_VERSION: str = "2026.04.13.1"

METHOD_SYMBOL_PASSTHROUGH: str = "symbol_passthrough"
METHOD_SYMBOL_NORMALIZED: str = "symbol_normalized"
METHOD_ALIAS_EXACT: str = "alias_exact"
METHOD_ALIAS_FOLDED: str = "alias_folded"
METHOD_NONE: str = "none"

SUPPORTED_MARKETS: frozenset[str] = frozenset({"HK", "JP", "TW", "CN"})

# Exchange-local code shape by market. HK codes range 1-5 digits but
# canonicalize to a 4-digit zero-padded form (HSBC = 0005.HK); JP/TW
# codes are essentially always 4 digits (occasionally 5), so we keep
# those stricter to avoid eating numeric noise like "700 words".
_LOCAL_CODE_RE_HK_PASSTHROUGH = re.compile(r"^\d{1,5}$")
_LOCAL_CODE_RE_HK_NORMALIZED = re.compile(r"^\d{3,5}$")
_LOCAL_CODE_RE_JP_TW = re.compile(r"^\d{4,5}$")
_LOCAL_CODE_RE_CN = re.compile(r"^\d{6}$")

# Punctuation to strip when building the aggressive (folded) match key.
# Kept small and CJK-aware: interpuncts, middle dots, hyphens, commas,
# periods, en/em dashes, and whitespace. NOT unicode categories wholesale
# because Han/kana are technically "Lo" (Other Letter).
_FOLD_STRIP_RE = re.compile(r"[\s\.\,\-\u2010-\u2015\u00B7\u30FB\u2022]")


@dataclass(frozen=True)
class AliasResolution:
    """Result of a deterministic alias/symbol resolution attempt.

    ``canonical_symbol`` is ``None`` when ``method == METHOD_NONE``.
    Callers MUST check the method field rather than truthiness on the
    symbol because an empty-string return is not valid.
    """

    query: str
    canonical_symbol: Optional[str]
    market: Optional[str]
    method: str
    policy_version: str


# ---------------------------------------------------------------------------
# Pure normalization primitives (no I/O, deterministic)
# ---------------------------------------------------------------------------


def nfkc(text: str) -> str:
    """NFKC-normalize and strip whitespace.

    Canonicalizes width variants (halfwidth kana, fullwidth digits/Latin)
    while preserving Han (which NFKC does not alter).
    """
    return unicodedata.normalize("NFKC", text or "").strip()


def fold_key(text: str) -> str:
    """Aggressive comparison key: NFKC + casefold + punctuation strip.

    Used for ``METHOD_ALIAS_FOLDED`` matches so inputs like "Sony Group"
    and "SONYGROUP" hit the same entry. Han/kana characters are
    preserved unchanged; only Latin letters casefold.
    """
    base = nfkc(text).casefold()
    return _FOLD_STRIP_RE.sub("", base)


# ---------------------------------------------------------------------------
# Inverted alias index (built once at import time)
# ---------------------------------------------------------------------------


def _build_indexes() -> Tuple[
    Dict[str, Tuple[str, str]],  # exact (NFKC) alias -> (canonical, market)
    Dict[str, List[Tuple[str, str]]],  # folded key -> [(canonical, market), ...]
]:
    exact: Dict[str, Tuple[str, str]] = {}
    folded: Dict[str, List[Tuple[str, str]]] = {}
    for canonical, entry in ASIA_ALIAS_CORPUS.items():
        market = entry["market"]
        for alias in entry["aliases"]:
            exact_key = nfkc(alias)
            if exact_key and exact_key not in exact:
                # First writer wins for exact matches; the seed avoids
                # collisions by construction. Duplicate definitions are
                # a caller-authoring bug, not runtime ambiguity.
                exact[exact_key] = (canonical, market)

            folded_key = fold_key(alias)
            if not folded_key:
                continue
            # Dedup: two aliases for the same canonical can fold to the
            # same key (e.g. "Sony Group" and "Sony-Group" both fold to
            # "sonygroup"). Without this check the folded list would
            # contain a duplicate (canonical, market) pair and
            # _disambiguate would treat it as spurious ambiguity.
            bucket = folded.setdefault(folded_key, [])
            if (canonical, market) not in bucket:
                bucket.append((canonical, market))
    return exact, folded


_EXACT_INDEX, _FOLDED_INDEX = _build_indexes()


# ---------------------------------------------------------------------------
# Symbol-shape resolution (rule 1 & 2)
# ---------------------------------------------------------------------------


def _try_symbol_passthrough(
    query: str, normalized: str
) -> Optional[AliasResolution]:
    """Input already carries a known SecurityMaster suffix."""
    upper = normalized.upper()
    for suffix, market in _MARKET_BY_SUFFIX:
        if market not in SUPPORTED_MARKETS:
            continue
        if not upper.endswith(suffix):
            continue
        local_code = upper[: -len(suffix)]
        if market == "HK":
            if _LOCAL_CODE_RE_HK_PASSTHROUGH.match(local_code):
                canonical = f"{int(local_code):04d}{suffix}"
                return AliasResolution(
                    query=query, canonical_symbol=canonical, market=market,
                    method=METHOD_SYMBOL_PASSTHROUGH,
                    policy_version=POLICY_VERSION,
                )
        elif market in {"JP", "TW"} and _LOCAL_CODE_RE_JP_TW.match(local_code):
            return AliasResolution(
                query=query, canonical_symbol=f"{local_code}{suffix}",
                market=market, method=METHOD_SYMBOL_PASSTHROUGH,
                policy_version=POLICY_VERSION,
            )
        elif market == "CN" and _LOCAL_CODE_RE_CN.match(local_code):
            return AliasResolution(
                query=query, canonical_symbol=f"{local_code}{suffix}",
                market=market, method=METHOD_SYMBOL_PASSTHROUGH,
                policy_version=POLICY_VERSION,
            )
    return None


def _try_symbol_normalized(
    query: str, normalized: str, hint_market: Optional[str]
) -> Optional[AliasResolution]:
    """Bare numeric code + market hint → canonical symbol."""
    if hint_market is None:
        return None
    market = hint_market.strip().upper()
    if market not in SUPPORTED_MARKETS:
        return None
    suffix = _SUFFIX_BY_MARKET[market]
    if market == "HK":
        if not _LOCAL_CODE_RE_HK_NORMALIZED.match(normalized):
            return None
        local_code = f"{int(normalized):04d}"
    else:
        if market == "CN":
            if not _LOCAL_CODE_RE_CN.match(normalized):
                return None
            if normalized.startswith(("600", "601", "603", "605", "688")):
                suffix = ".SS"
            elif normalized.startswith(("000", "001", "002", "003", "300", "301")):
                suffix = ".SZ"
            elif normalized.startswith(("4", "8", "9")):
                suffix = ".BJ"
            else:
                return None
            local_code = normalized
        elif not _LOCAL_CODE_RE_JP_TW.match(normalized):
            return None
        else:
            # JP/TW canonical codes are preserved verbatim at 4 or 5 digits.
            local_code = normalized
    return AliasResolution(
        query=query,
        canonical_symbol=f"{local_code}{suffix}",
        market=market,
        method=METHOD_SYMBOL_NORMALIZED,
        policy_version=POLICY_VERSION,
    )


# ---------------------------------------------------------------------------
# Alias-dictionary resolution (rule 3 & 4)
# ---------------------------------------------------------------------------


def _disambiguate(
    candidates: List[Tuple[str, str]], hint_market: Optional[str]
) -> Optional[Tuple[str, str]]:
    """Pick a single (canonical, market) from folded matches.

    Unique match → return it. Multiple matches → disambiguate by
    hint_market if provided, otherwise return None (caller falls
    through to LLM).
    """
    if len(candidates) == 1:
        return candidates[0]
    if hint_market is not None:
        hint = hint_market.strip().upper()
        filtered = [c for c in candidates if c[1] == hint]
        if len(filtered) == 1:
            return filtered[0]
    return None


def _try_alias_exact(normalized: str) -> Optional[Tuple[str, str]]:
    return _EXACT_INDEX.get(normalized)


def _try_alias_folded(
    folded: str, hint_market: Optional[str]
) -> Optional[Tuple[str, str]]:
    if not folded:
        return None
    candidates = _FOLDED_INDEX.get(folded)
    if not candidates:
        return None
    return _disambiguate(candidates, hint_market)


# ---------------------------------------------------------------------------
# Public resolver
# ---------------------------------------------------------------------------


def resolve_alias(
    query: str, *, hint_market: Optional[str] = None
) -> AliasResolution:
    """Resolve ``query`` to a canonical symbol deterministically.

    Returns an :class:`AliasResolution` with ``method == METHOD_NONE``
    when no rule or alias matches — that is NOT an error, it simply
    means the caller should fall through to the LLM-based resolver.

    ``hint_market`` is consulted for symbol normalization (bare numeric
    codes) and to disambiguate folded-alias collisions.
    """
    normalized = nfkc(query)
    if not normalized:
        return AliasResolution(
            query=query, canonical_symbol=None, market=None,
            method=METHOD_NONE, policy_version=POLICY_VERSION,
        )

    passthrough = _try_symbol_passthrough(query, normalized)
    if passthrough is not None:
        return passthrough

    symbol_normalized = _try_symbol_normalized(query, normalized, hint_market)
    if symbol_normalized is not None:
        return symbol_normalized

    exact = _try_alias_exact(normalized)
    if exact is not None:
        canonical, market = exact
        return AliasResolution(
            query=query, canonical_symbol=canonical, market=market,
            method=METHOD_ALIAS_EXACT, policy_version=POLICY_VERSION,
        )

    # Deferred: only pay for casefold + punctuation-strip when the
    # exact-match path has missed.
    folded = _try_alias_folded(
        _FOLD_STRIP_RE.sub("", normalized.casefold()), hint_market,
    )
    if folded is not None:
        canonical, market = folded
        return AliasResolution(
            query=query, canonical_symbol=canonical, market=market,
            method=METHOD_ALIAS_FOLDED, policy_version=POLICY_VERSION,
        )

    return AliasResolution(
        query=query, canonical_symbol=None, market=None,
        method=METHOD_NONE, policy_version=POLICY_VERSION,
    )


# ---------------------------------------------------------------------------
# Policy surface
# ---------------------------------------------------------------------------


def policy_version() -> str:
    """Mirror of sibling policy modules (``translation_service`` etc.)."""
    return POLICY_VERSION


def describe_policy() -> dict:
    """Stable snapshot for API / admin surfacing and drift detection."""
    return {
        "policy_version": POLICY_VERSION,
        "supported_markets": sorted(SUPPORTED_MARKETS),
        "corpus_size": len(ASIA_ALIAS_CORPUS),
        "alias_count": sum(
            len(entry["aliases"]) for entry in ASIA_ALIAS_CORPUS.values()
        ),
        "methods": [
            METHOD_SYMBOL_PASSTHROUGH,
            METHOD_SYMBOL_NORMALIZED,
            METHOD_ALIAS_EXACT,
            METHOD_ALIAS_FOLDED,
            METHOD_NONE,
        ],
    }


__all__ = [
    "AliasResolution",
    "METHOD_ALIAS_EXACT",
    "METHOD_ALIAS_FOLDED",
    "METHOD_NONE",
    "METHOD_SYMBOL_NORMALIZED",
    "METHOD_SYMBOL_PASSTHROUGH",
    "POLICY_VERSION",
    "SUPPORTED_MARKETS",
    "describe_policy",
    "policy_version",
    "resolve_alias",
]
