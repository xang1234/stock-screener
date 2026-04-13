"""Deterministic language detection for theme content (T7.2).

Rule-first per the E7 epic — *rule-and-cache first, LLM second*. A pure
Unicode-script heuristic covers the US/HK/JP/TW markets without external
libraries or non-deterministic sampling:

- Hiragana / Katakana present  → Japanese (``ja``)
- Han (CJK ideograph) dominant → Chinese (``zh``)
- Otherwise                    → English (``en``)
- Empty / non-letter input     → ``und`` (BCP-47 "undetermined")

Caching
-------
The result is persisted on :attr:`ContentItem.source_language` (T7.1
column); the DB column IS the cache. :func:`detect_and_cache_language`
reads the column first and only runs detection on a miss. Re-running
the detection stage on the same row is therefore a no-op — the
acceptance criterion for T7.2 ("detection integrated and idempotent").

Policy version
--------------
:data:`POLICY_VERSION` bumps when the *semantics* change (different
threshold, added a 4th language). Bug fixes to the Unicode range
constants don't require a bump.
"""

from __future__ import annotations

from typing import Any, Optional

POLICY_VERSION: str = "2026.04.13.1"

LANGUAGE_EN: str = "en"
LANGUAGE_JA: str = "ja"
LANGUAGE_ZH: str = "zh"
LANGUAGE_UNKNOWN: str = "und"

SUPPORTED_LANGUAGES: frozenset[str] = frozenset(
    {LANGUAGE_EN, LANGUAGE_JA, LANGUAGE_ZH, LANGUAGE_UNKNOWN}
)

# Minimum CJK-ideograph proportion before we classify as Chinese. Below
# this, an English article citing a single Han character (e.g. a company
# name in a Bloomberg headline) stays classified as English.
_CJK_THRESHOLD: float = 0.20

# Kana proportion threshold. Japanese text *always* has some kana;
# Chinese essentially never does. 2% is low enough to catch short
# headlines and high enough to shrug off a stray kana in quoted text.
_KANA_THRESHOLD: float = 0.02


def _is_kana(ch: str) -> bool:
    cp = ord(ch)
    # Hiragana: U+3040 – U+309F; Katakana: U+30A0 – U+30FF;
    # Katakana phonetic extensions: U+31F0 – U+31FF.
    return 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF


def _is_cjk_ideograph(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF        # CJK Unified Ideographs
        or 0x3400 <= cp <= 0x4DBF     # Extension A
        or 0xF900 <= cp <= 0xFAFF     # Compatibility Ideographs
        or 0x20000 <= cp <= 0x2A6DF   # Extension B
    )


def detect_language(text: Optional[str]) -> str:
    """Deterministically classify ``text`` into ``en`` / ``ja`` / ``zh`` / ``und``.

    Pure function: no I/O, no randomness, no external library. Same
    input → same output across processes and Python versions.
    """
    if not text:
        return LANGUAGE_UNKNOWN

    kana = 0
    cjk = 0
    letters = 0
    for ch in text:
        if not ch.isalpha():
            # Skip digits, punctuation, whitespace, etc. — only scripted
            # characters contribute signal.
            continue
        letters += 1
        if _is_kana(ch):
            kana += 1
        elif _is_cjk_ideograph(ch):
            cjk += 1

    if letters == 0:
        return LANGUAGE_UNKNOWN
    if kana / letters >= _KANA_THRESHOLD:
        return LANGUAGE_JA
    if (cjk + kana) / letters >= _CJK_THRESHOLD:
        return LANGUAGE_ZH
    return LANGUAGE_EN


def detect_and_cache_language(
    content_item: Any,
    *,
    force_refresh: bool = False,
) -> str:
    """Return ``content_item.source_language``, detecting + persisting on miss.

    The DB column IS the cache (see module docstring). Does NOT commit —
    callers own the transaction boundary.

    Args:
        content_item: A :class:`app.models.theme.ContentItem` row.
        force_refresh: Re-detect and overwrite even if the column is
            already populated (useful for backfills after a policy
            version bump).

    Returns:
        The detected or cached BCP-47 short tag.
    """
    existing = content_item.source_language
    if existing and not force_refresh:
        return existing

    # Title + content together: a Japanese article whose translated
    # headline is in English would otherwise be mis-classified if we
    # only looked at the title.
    parts = (content_item.title, content_item.content)
    text = " ".join(p for p in parts if p)
    detected = detect_language(text)
    content_item.source_language = detected
    return detected


def policy_version() -> str:
    """Mirror of sibling policy modules (``provider_routing_policy`` etc.)."""
    return POLICY_VERSION


def describe_policy() -> dict:
    """Stable snapshot for API / admin surfacing."""
    return {
        "policy_version": POLICY_VERSION,
        "supported_languages": sorted(SUPPORTED_LANGUAGES),
        "cjk_threshold": _CJK_THRESHOLD,
        "kana_threshold": _KANA_THRESHOLD,
    }


__all__ = [
    "POLICY_VERSION",
    "LANGUAGE_EN",
    "LANGUAGE_JA",
    "LANGUAGE_ZH",
    "LANGUAGE_UNKNOWN",
    "SUPPORTED_LANGUAGES",
    "policy_version",
    "detect_language",
    "detect_and_cache_language",
    "describe_policy",
]
