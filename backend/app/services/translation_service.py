"""Translation stage with DB-column caching and confidence metadata.

Part of the multilingual content pipeline:

    content ingest → detect_language → translate → extract

The translated text and a self-contained :class:`TranslationQuote`
snapshot are persisted on :class:`ContentItem`. The columns ARE the
cache — re-running the translate stage on the same row is a no-op
when translation already succeeded, and a retry when the previous
attempt recorded ``provider = "unavailable"``.

The translation provider is injected (no default wired) so this module
stays import-safe and unit-testable. Real deployments wire DeepL /
Google / LLM-based translators at the composition root.

Confidence
----------
Downstream extraction / QA scoring should gate on the confidence stored
in ``translation_metadata``. :data:`CONFIDENCE_DOWNGRADE_THRESHOLD`
names the cutoff below which extraction rating should be downgraded.

Note on scales: this threshold is on the 0.0–1.0 MT-confidence scale,
distinct from the 0–100 integer ``field_completeness_score`` gated by
``QUALITY_DOWNGRADE_THRESHOLD = 60`` in ``scoring.py``. The two are
analogous in intent (below the cutoff, trust less) but live on
different scales because their sources differ — don't try to unify the
numeric values.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Callable, NamedTuple, Optional, Tuple

if TYPE_CHECKING:
    from ..models.theme import ContentItem

logger = logging.getLogger(__name__)

POLICY_VERSION: str = "2026.04.13.1"

DEFAULT_TARGET_LANGUAGE: str = "en"

# Below this translation confidence (0.0-1.0 scale), callers should
# downgrade the extraction rating.
CONFIDENCE_DOWNGRADE_THRESHOLD: float = 0.70

# Sentinel provider values used in translation_metadata when there is
# no "real" translation to record. Callers can pattern-match on these.
PROVIDER_IDENTITY: str = "identity"        # source language == target language
PROVIDER_UNAVAILABLE: str = "unavailable"  # translator failed / not configured
PROVIDER_EMPTY: str = "empty"              # input text was empty — no round-trip made

# Cap the ``reason`` field recorded in unavailable metadata so a huge
# stack trace can't bloat a JSONB row.
MAX_REASON_LENGTH: int = 200


# Injected translator contract (kept as a type alias, not a Protocol, to
# match the ``RateFetcher`` convention in ``fx_service``).
#
#   translator(text, source_language, target_language) -> (translated_text, quote)
#
# Exceptions propagate out of ``translate_content_item`` which then
# records ``unavailable`` metadata and resets translated_* to None so
# the next invocation retries. An empty input should never reach the
# translator — ``_translate_or_skip_empty`` short-circuits first.
Translator = Callable[[str, str, str], Tuple[str, "TranslationQuote"]]


class ExtractionText(NamedTuple):
    """Result of :func:`select_extraction_text` — named for readability."""
    title: str
    content: str
    language: str


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

    @staticmethod
    def identity_metadata(source_language: str, target_language: str) -> dict:
        """Metadata for content already in the target language (no MT needed)."""
        return {
            "provider": PROVIDER_IDENTITY,
            "source_language": source_language,
            "target_language": target_language,
            "model": None,
            "confidence": 1.0,
            "translated_at": date.today().isoformat(),
        }

    @staticmethod
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
            "reason": reason[:MAX_REASON_LENGTH],
        }


class TranslationService:
    """Translate-and-cache on :class:`ContentItem` rows.

    Idempotency contract:

    - ``source_language == target_language`` → mark as identity, copy
      originals into translated_* so downstream extraction always reads
      from the same fields.
    - ``translated_title`` present and ``metadata.target_language`` ==
      requested target AND provider != ``"unavailable"`` → cached,
      no-op.
    - ``provider == "unavailable"`` → retry. Transient failures
      shouldn't permanently mark a row as un-translatable.
    - ``force_refresh=True`` → re-translate regardless of cache state.

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
                f"source_language; run detect_and_cache_language first"
            )

        if source == self._target_language:
            if not force_refresh and _is_cached_translation(
                content_item, self._target_language,
            ):
                return content_item.translation_metadata
            return self._write_identity(content_item, source)

        if not force_refresh and _is_cached_translation(
            content_item, self._target_language,
        ):
            return content_item.translation_metadata

        # Atomic: both title and content must succeed, or neither is
        # written. Any translator exception resets translated_* to None
        # and records unavailable metadata so the next call retries.
        # Empty halves skip the translator entirely (no paid round trip).
        try:
            translated_title, title_quote = self._translate_or_skip_empty(
                content_item.title or "", source,
            )
            translated_content, content_quote = self._translate_or_skip_empty(
                content_item.content or "", source,
            )
        except Exception as exc:
            logger.warning(
                "Translation failed for ContentItem %s (%s → %s): %s",
                getattr(content_item, "id", "?"), source, self._target_language, exc,
            )
            return self._write_unavailable(
                content_item, source, reason=f"{type(exc).__name__}: {exc}",
            )

        # Merge: take the weaker confidence so QA gates on the lower
        # half of the row. Skipped halves contribute None, which
        # ``_min_optional`` treats as "unknown" (not zero).
        title_conf = title_quote.confidence if title_quote else None
        content_conf = content_quote.confidence if content_quote else None
        combined_confidence = _min_optional(title_conf, content_conf)

        # Provider / model come from whichever half actually called the
        # translator. Both skipped → explicit "empty" provider so
        # callers can see the stage ran but had no text to translate.
        carrier = content_quote or title_quote
        if carrier is not None:
            provider, model = carrier.provider, carrier.model
        else:
            provider, model = PROVIDER_EMPTY, None

        merged = TranslationQuote(
            source_language=source,
            target_language=self._target_language,
            provider=provider,
            model=model,
            confidence=combined_confidence,
            translated_at=date.today(),
        )
        content_item.translated_title = translated_title[:500]
        content_item.translated_content = translated_content
        content_item.translation_metadata = merged.to_metadata()
        return content_item.translation_metadata

    def _translate_or_skip_empty(
        self, text: str, source: str,
    ) -> Tuple[str, Optional["TranslationQuote"]]:
        """Invoke the translator, short-circuiting on empty/whitespace input.

        Returns ``(translated_text, quote_or_None)``. A ``None`` quote
        signals "no call made" so downstream merging can ignore this
        half. Saves a paid round-trip on rows where title or content is
        missing — common at bulk-ingest scale.
        """
        if not text.strip():
            return "", None
        return self._translator(text, source, self._target_language)

    def _write_identity(self, content_item: "ContentItem", source: str) -> dict:
        meta = TranslationQuote.identity_metadata(source, self._target_language)
        content_item.translated_title = content_item.title
        content_item.translated_content = content_item.content
        content_item.translation_metadata = meta
        return meta

    def _write_unavailable(
        self, content_item: "ContentItem", source: str, *, reason: str,
    ) -> dict:
        meta = TranslationQuote.unavailable_metadata(
            source, self._target_language, reason,
        )
        content_item.translated_title = None
        content_item.translated_content = None
        content_item.translation_metadata = meta
        return meta


# ---------------------------------------------------------------------------
# Consumption helpers
# ---------------------------------------------------------------------------


def select_extraction_text(content_item: "ContentItem") -> ExtractionText:
    """Pick the (title, content, language) tuple for downstream extraction.

    Returns translated title/content when a usable translation is
    available; otherwise falls back to the original. ``language`` is
    the language the returned text is in (target on translation hit,
    source on fallback), so the extraction prompt can condition on it.
    """
    meta = content_item.translation_metadata
    if meta is None:
        return ExtractionText(
            content_item.title or "",
            content_item.content or "",
            content_item.source_language or "und",
        )

    provider = meta.get("provider")
    target = meta.get("target_language") or DEFAULT_TARGET_LANGUAGE

    if (
        provider == PROVIDER_IDENTITY
        and meta.get("source_language") == content_item.source_language
        and target == content_item.source_language
    ):
        return ExtractionText(
            content_item.translated_title or content_item.title or "",
            content_item.translated_content or content_item.content or "",
            target,
        )

    if (
        provider not in (None, PROVIDER_UNAVAILABLE)
        and meta.get("source_language") == content_item.source_language
        and content_item.translated_title is not None
        and content_item.translated_content is not None
    ):
        return ExtractionText(
            content_item.translated_title,
            content_item.translated_content,
            target,
        )

    # Unavailable / corrupted / missing translated columns: fall back
    # to originals with source language tagged so callers can downgrade.
    return ExtractionText(
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
    """True when downstream scoring should downgrade this row for low MT confidence."""
    meta = content_item.translation_metadata
    if not meta:
        return False
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
        "empty_provider": PROVIDER_EMPTY,
        "max_reason_length": MAX_REASON_LENGTH,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_cached_translation(
    content_item: "ContentItem",
    target_language: str,
) -> bool:
    """A previous successful translation to the same (source, target) pair is present.

    Validates both target and source language so a later source-language
    correction (e.g. T7.2 re-detection after a policy bump) triggers a
    fresh translation rather than reusing a stale one.
    """
    meta = content_item.translation_metadata
    if not meta:
        return False
    if meta.get("target_language") != target_language:
        return False
    if meta.get("source_language") != content_item.source_language:
        return False
    provider = meta.get("provider")
    if provider in (None, PROVIDER_UNAVAILABLE):
        return False
    if provider == PROVIDER_IDENTITY:
        return (
            content_item.translated_title == content_item.title
            and content_item.translated_content == content_item.content
        )
    return (
        content_item.translated_title is not None
        and content_item.translated_content is not None
    )


def _min_optional(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """``min`` that treats ``None`` as "unknown", not zero."""
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
    "PROVIDER_EMPTY",
    "MAX_REASON_LENGTH",
    "Translator",
    "TranslationQuote",
    "TranslationService",
    "ExtractionText",
    "select_extraction_text",
    "translation_confidence",
    "should_downgrade_for_translation",
    "policy_version",
    "describe_policy",
]
