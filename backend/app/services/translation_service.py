"""Translation stage with DB-column caching and confidence metadata.

Part of the E7 multilingual pipeline:

    content ingest → [T7.2 detect_language] → [T7.3 translate] → extract

The translated text and a self-contained :class:`TranslationQuote`
snapshot are persisted on :class:`ContentItem` (T7.1 columns). The
columns ARE the cache — re-running the translate stage on the same
row is a no-op when translation already succeeded, and a retry when
the previous attempt recorded ``provider = "unavailable"``.

The translation provider is injected (no default wired) so this module
stays import-safe and unit-testable. Real deployments wire DeepL /
Google / LLM-based translators at the composition root.

Confidence
----------
Downstream extraction / QA scoring should gate on the confidence stored
in ``translation_metadata``; :data:`CONFIDENCE_DOWNGRADE_THRESHOLD`
names the cutoff below which extraction rating should be downgraded
(analogous to ``QUALITY_DOWNGRADE_THRESHOLD`` in the scoring policy).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

if TYPE_CHECKING:
    from ..models.theme import ContentItem

logger = logging.getLogger(__name__)

POLICY_VERSION: str = "2026.04.13.1"

DEFAULT_TARGET_LANGUAGE: str = "en"

# Below this translation confidence, callers should downgrade the
# extraction rating (mirrors scoring.py's QUALITY_DOWNGRADE_THRESHOLD
# convention: below the threshold, trust less).
CONFIDENCE_DOWNGRADE_THRESHOLD: float = 0.70

# Sentinel provider values used in translation_metadata when there is
# no "real" translation to record. Callers can pattern-match on these.
PROVIDER_IDENTITY: str = "identity"      # source language == target language
PROVIDER_UNAVAILABLE: str = "unavailable"  # translator failed / not configured


# Callable shape: (text, source_language, target_language) -> (translated_text, quote)
Translator = Callable[[str, str, str], Tuple[str, "TranslationQuote"]]


@dataclass(frozen=True)
class TranslationQuote:
    """Self-contained replay snapshot of a translation event.

    Mirrors :class:`FXQuote` in :mod:`app.services.fx_service` — the
    dataclass is the canonical shape, and :meth:`to_metadata` emits the
    dict that goes into ``ContentItem.translation_metadata``.
    """

    source_language: str
    target_language: str
    provider: str
    model: Optional[str]
    confidence: Optional[float]
    translated_at: date

    def to_metadata(self) -> dict:
        return {
            "provider": self.provider,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "model": self.model,
            "confidence": self.confidence,
            "translated_at": self.translated_at.isoformat(),
        }


def identity_metadata(source_language: str, target_language: str) -> dict:
    """Metadata for content already in the target language (no MT needed)."""
    return {
        "provider": PROVIDER_IDENTITY,
        "source_language": source_language,
        "target_language": target_language,
        "model": None,
        "confidence": 1.0,  # exact copy — perfect "translation"
        "translated_at": date.today().isoformat(),
    }


def unavailable_metadata(
    source_language: str,
    target_language: str,
    reason: str,
) -> dict:
    """Metadata for a failed translation attempt — enables retry next run."""
    return {
        "provider": PROVIDER_UNAVAILABLE,
        "source_language": source_language,
        "target_language": target_language,
        "model": None,
        "confidence": None,
        "translated_at": date.today().isoformat(),
        "reason": reason[:200],  # Cap so a huge stack trace can't bloat the row
    }


class TranslationService:
    """Translate-and-cache on :class:`ContentItem` rows.

    Idempotency contract:

    - ``source_language == target_language`` → mark as identity, copy
      originals into translated_* so downstream extraction always reads
      from the same fields.
    - ``translated_title`` present and ``metadata.target_language`` ==
      requested target AND provider != ``"unavailable"`` → cached,
      no-op. Second call returns immediately.
    - ``provider == "unavailable"`` → retry. Transient failures
      shouldn't permanently mark a row as un-translatable.
    - ``force_refresh=True`` → re-translate regardless of cache state
      (for backfills after a policy-version bump).

    Atomicity: title and content are translated together. If the
    translator fails partway, both translated_* fields are reset to
    None and ``unavailable`` metadata is recorded — never partial.
    """

    def __init__(
        self,
        translator: Translator,
        *,
        target_language: str = DEFAULT_TARGET_LANGUAGE,
    ) -> None:
        self._translator = translator
        self._target_language = target_language

    def translate_content_item(
        self,
        content_item: "ContentItem",
        *,
        force_refresh: bool = False,
    ) -> dict:
        """Ensure translated_* columns are populated. Returns the metadata written.

        Does NOT commit — callers own the transaction boundary.
        """
        source = content_item.source_language
        if source is None:
            raise ValueError(
                f"ContentItem {getattr(content_item, 'id', '?')} has no "
                f"source_language; run detect_and_cache_language (T7.2) first"
            )

        # Identity fast path: source already in target, no MT needed.
        if source == self._target_language:
            return self._write_identity(content_item, source)

        # Cache hit: we've already translated this to the requested target.
        if not force_refresh and _is_cached_translation(content_item, self._target_language):
            return content_item.translation_metadata

        # Invoke the injected translator. Both title and content must
        # succeed — if either raises, record unavailable and leave
        # translated_* as None so the next retry can re-attempt.
        try:
            translated_title, title_quote = self._translator(
                content_item.title or "", source, self._target_language,
            )
            translated_content, content_quote = self._translator(
                content_item.content or "", source, self._target_language,
            )
        except Exception as exc:  # translator contract is provider-agnostic
            logger.warning(
                "Translation failed for ContentItem %s (%s → %s): %s",
                getattr(content_item, "id", "?"), source, self._target_language, exc,
            )
            return self._write_unavailable(
                content_item, source, reason=f"{type(exc).__name__}: {exc}",
            )

        # Use the weaker of the two quote confidences (QA should gate
        # on the lower-quality half of the row).
        combined_confidence = _min_optional(
            title_quote.confidence, content_quote.confidence,
        )
        merged = TranslationQuote(
            source_language=source,
            target_language=self._target_language,
            provider=content_quote.provider,
            model=content_quote.model,
            confidence=combined_confidence,
            translated_at=content_quote.translated_at,
        )
        content_item.translated_title = translated_title
        content_item.translated_content = translated_content
        content_item.translation_metadata = merged.to_metadata()
        return content_item.translation_metadata

    def _write_identity(self, content_item: "ContentItem", source: str) -> dict:
        meta = identity_metadata(source, self._target_language)
        content_item.translated_title = content_item.title
        content_item.translated_content = content_item.content
        content_item.translation_metadata = meta
        return meta

    def _write_unavailable(
        self, content_item: "ContentItem", source: str, *, reason: str,
    ) -> dict:
        meta = unavailable_metadata(source, self._target_language, reason)
        content_item.translated_title = None
        content_item.translated_content = None
        content_item.translation_metadata = meta
        return meta


# ---------------------------------------------------------------------------
# Consumption helpers
# ---------------------------------------------------------------------------


def select_extraction_text(content_item: "ContentItem") -> Tuple[str, str, str]:
    """Pick the (title, content, language) tuple for downstream extraction.

    Returns the translated title/content when a usable translation is
    available; otherwise falls back to the original. The third tuple
    element is the language the returned text is in (the target
    language on a translation hit, the source on a fallback). Lets the
    extraction prompt condition on what it's actually reading.
    """
    meta = content_item.translation_metadata
    if meta is None:
        return (content_item.title or "", content_item.content or "", content_item.source_language or "und")

    provider = meta.get("provider")
    target = meta.get("target_language") or DEFAULT_TARGET_LANGUAGE

    # Identity path: translated_title mirrors the original; use it so the
    # call site doesn't need special-case branching.
    if provider == PROVIDER_IDENTITY:
        return (
            content_item.translated_title or content_item.title or "",
            content_item.translated_content or content_item.content or "",
            target,
        )

    # Real translation with populated columns.
    if (
        provider not in (None, PROVIDER_UNAVAILABLE)
        and content_item.translated_title is not None
    ):
        return (
            content_item.translated_title,
            content_item.translated_content or "",
            target,
        )

    # Unavailable / missing: fall back to originals so extraction can
    # still run, flagged with the source language. Callers should check
    # provider == "unavailable" to decide whether to downgrade confidence.
    return (
        content_item.title or "",
        content_item.content or "",
        content_item.source_language or "und",
    )


def translation_confidence(content_item: "ContentItem") -> Optional[float]:
    """Return the recorded confidence, or None if absent/unknown."""
    meta = content_item.translation_metadata
    if not meta:
        return None
    value = meta.get("confidence")
    return float(value) if value is not None else None


def should_downgrade_for_translation(content_item: "ContentItem") -> bool:
    """True when downstream scoring should downgrade this row for low MT confidence.

    Unavailable translations are treated as "below threshold" so
    extraction rating reflects the missing-text fallback. Identity and
    sufficiently-confident real translations pass through.
    """
    meta = content_item.translation_metadata
    if not meta:
        return False  # Haven't attempted translation yet — not our call
    if meta.get("provider") == PROVIDER_UNAVAILABLE:
        return True
    confidence = translation_confidence(content_item)
    if confidence is None:
        return False
    return confidence < CONFIDENCE_DOWNGRADE_THRESHOLD


# ---------------------------------------------------------------------------
# Policy surface
# ---------------------------------------------------------------------------


def policy_version() -> str:
    return POLICY_VERSION


def describe_policy() -> dict:
    return {
        "policy_version": POLICY_VERSION,
        "default_target_language": DEFAULT_TARGET_LANGUAGE,
        "confidence_downgrade_threshold": CONFIDENCE_DOWNGRADE_THRESHOLD,
        "identity_provider": PROVIDER_IDENTITY,
        "unavailable_provider": PROVIDER_UNAVAILABLE,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_cached_translation(
    content_item: "ContentItem",
    target_language: str,
) -> bool:
    """A previous successful translation to the same target is present."""
    meta = content_item.translation_metadata
    if not meta:
        return False
    if meta.get("target_language") != target_language:
        return False
    provider = meta.get("provider")
    if provider in (None, PROVIDER_UNAVAILABLE):
        return False
    if provider == PROVIDER_IDENTITY:
        return True
    return content_item.translated_title is not None


def _min_optional(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """``min`` that treats ``None`` as "unknown" — returns the defined value if one is None, else the minimum."""
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


__all__ = [
    "POLICY_VERSION",
    "DEFAULT_TARGET_LANGUAGE",
    "CONFIDENCE_DOWNGRADE_THRESHOLD",
    "PROVIDER_IDENTITY",
    "PROVIDER_UNAVAILABLE",
    "Translator",
    "TranslationQuote",
    "TranslationService",
    "identity_metadata",
    "unavailable_metadata",
    "select_extraction_text",
    "translation_confidence",
    "should_downgrade_for_translation",
    "policy_version",
    "describe_policy",
]
