"""Deterministic language detection for theme content.

Rule-first Unicode-script heuristic classifies content into the four
tags used across the multilingual pipeline:

- Hiragana / Katakana present  → Japanese (``ja``)
- Han (CJK ideograph) dominant → Chinese (``zh``)
- Otherwise                    → English (``en``)
- Empty / non-letter input     → ``und`` (BCP-47 "undetermined")

Kana is the load-bearing signal for Japanese: Japanese text always has
some kana; Chinese essentially never does. Even a 2% kana proportion
is decisive. The Chinese threshold is higher because an English
article citing a single Han character (e.g. ``日経`` in a Bloomberg
headline) should not flip to Chinese.

Caching
-------
The result is persisted on :attr:`ContentItem.source_language`; the DB
column IS the cache. :func:`detect_and_cache_language` reads the
column first and only detects on a miss, so re-running the detection
stage on the same row is a no-op.

Policy version
--------------
:data:`POLICY_VERSION` bumps when the *semantics* change: threshold
adjustment, added language, swapped algorithm. Bug fixes to the
Unicode range constants don't require a bump.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..models.theme import ContentItem

POLICY_VERSION: str = "2026.04.13.1"

LANGUAGE_EN: str = "en"
LANGUAGE_JA: str = "ja"
LANGUAGE_ZH: str = "zh"
LANGUAGE_UNKNOWN: str = "und"

SUPPORTED_LANGUAGES: frozenset[str] = frozenset(
    {LANGUAGE_EN, LANGUAGE_JA, LANGUAGE_ZH, LANGUAGE_UNKNOWN}
)

# Thresholds are part of the documented policy (exposed via
# describe_policy). Bumping either requires a POLICY_VERSION bump.
CJK_THRESHOLD: float = 0.20
KANA_THRESHOLD: float = 0.02

# Cap detection input to avoid scanning entire 20KB+ article bodies
# during bulk ingestion. Script ratios stabilise quickly — a 4KB prefix
# is statistically indistinguishable from the full body for the en / ja
# / zh triage this module does.
DETECTION_SAMPLE_LIMIT: int = 4096


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


def build_detection_text(
    title: Optional[str], content: Optional[str]
) -> str:
    """Combine title + content into a single deterministic sample string.

    Handles None and missing parts uniformly so both ingestion (dict
    payloads) and the ORM-aware wrapper produce the same input for the
    detector. The returned string is already truncated to
    :data:`DETECTION_SAMPLE_LIMIT` — callers don't need a second cap.
    """
    joined = " ".join(p for p in (title, content) if p)
    return joined[:DETECTION_SAMPLE_LIMIT]


def detect_language(text: Optional[str]) -> str:
    """Deterministically classify ``text`` into ``en`` / ``ja`` / ``zh`` / ``und``.

    Pure function: no I/O, no randomness, no external library. Same
    input → same output across processes and Python versions.
    """
    if not text:
        return LANGUAGE_UNKNOWN

    # Only scan up to the sample limit — stable ratios emerge quickly.
    sample = text[:DETECTION_SAMPLE_LIMIT]

    kana = 0
    cjk = 0
    letters = 0
    for ch in sample:
        if not ch.isalpha():
            # Skip digits, punctuation, whitespace — only scripted
            # characters contribute signal.
            continue
        letters += 1
        if _is_kana(ch):
            kana += 1
        elif _is_cjk_ideograph(ch):
            cjk += 1

    if letters == 0:
        return LANGUAGE_UNKNOWN
    if kana / letters >= KANA_THRESHOLD:
        return LANGUAGE_JA
    if cjk / letters >= CJK_THRESHOLD:
        return LANGUAGE_ZH
    return LANGUAGE_EN


def detect_and_cache_language(
    content_item: "ContentItem",
    *,
    force_refresh: bool = False,
) -> str:
    """Return ``content_item.source_language``, detecting + persisting on miss.

    The DB column IS the cache (see module docstring). Does NOT commit
    — callers own the transaction boundary.

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
    detected = detect_language(
        build_detection_text(content_item.title, content_item.content)
    )
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
        "cjk_threshold": CJK_THRESHOLD,
        "kana_threshold": KANA_THRESHOLD,
        "detection_sample_limit": DETECTION_SAMPLE_LIMIT,
    }


__all__ = [
    "POLICY_VERSION",
    "LANGUAGE_EN",
    "LANGUAGE_JA",
    "LANGUAGE_ZH",
    "LANGUAGE_UNKNOWN",
    "SUPPORTED_LANGUAGES",
    "CJK_THRESHOLD",
    "KANA_THRESHOLD",
    "DETECTION_SAMPLE_LIMIT",
    "build_detection_text",
    "policy_version",
    "detect_language",
    "detect_and_cache_language",
    "describe_policy",
]
