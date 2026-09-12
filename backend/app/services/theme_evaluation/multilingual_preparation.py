"""Source-preserving language preparation, with an explicit translator boundary."""

import re
from collections import Counter
from typing import Literal

from pydantic import Field, model_validator

from .records import Record


class TranslationSegment(Record):
    original: str
    translated: str | None
    status: Literal["translated", "identity", "unavailable"]
    failure_reason: str | None = None

    @model_validator(mode="after")
    def consistent_segment(self):
        if self.status == "unavailable":
            if self.translated is not None or not self.failure_reason:
                raise ValueError("unavailable_translation_requires_reason_and_no_text")
        elif self.translated is None or (
            self.original.strip() and not self.translated.strip()
        ):
            raise ValueError("translation_segment_missing")
        elif self.failure_reason:
            raise ValueError("available_translation_has_failure")
        if self.status == "identity" and self.translated != self.original:
            raise ValueError("identity_translation_changed")
        return self


class TextPreparation(Record):
    source_language: str
    supplied_language: str | None
    target_language: str
    segments: list[TranslationSegment]
    language_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_identity(self):
        for segment in self.segments:
            if (
                segment.status == "identity"
                and segment.original.strip()
                and (
                    self.source_language != self.target_language
                    or "language_metadata_conflict" in self.language_warnings
                )
            ):
                raise ValueError("identity_requires_matching_language")
        return self

    @property
    def warnings(self) -> list[str]:
        warnings = list(self.language_warnings)
        for segment in self.segments:
            if segment.failure_reason:
                warnings.append(segment.failure_reason)
            elif segment.status == "translated":
                if (
                    segment.translated.strip() == segment.original.strip()
                    and self.source_language != self.target_language
                ):
                    warnings.append("translation_unchanged")
                warnings.extend(
                    numerical_warnings(segment.original, segment.translated)
                )
        return list(dict.fromkeys(warnings))

    @property
    def status(self):
        missing = sum(s.status == "unavailable" for s in self.segments)
        if missing:
            return "unavailable" if missing == len(self.segments) else "partial"
        return "needs_review" if self.warnings else "success"


def finalize_translation(
    previous: TextPreparation,
    translations: list[str | None],
    *,
    missing_reason="translation_segment_failed",
) -> TextPreparation:
    """Attach outputs to exact stored segments; never detect or segment again."""
    if len(translations) != len(previous.segments):
        raise ValueError("translation_import_segment_count")
    segments = []
    for segment, translated in zip(previous.segments, translations):
        if segment.status == "identity":
            if translated != segment.original:
                raise ValueError("identity_translation_changed")
            segments.append(segment)
        else:
            segments.append(
                TranslationSegment(
                    original=segment.original,
                    translated=translated,
                    status="translated" if translated is not None else "unavailable",
                    failure_reason=missing_reason if translated is None else None,
                )
            )
    return TextPreparation(
        source_language=previous.source_language,
        supplied_language=previous.supplied_language,
        target_language=previous.target_language,
        language_warnings=previous.language_warnings,
        segments=segments,
    )


def detect_language(text: str, supplied: str | None = None):
    hangul = bool(re.search(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]", text))
    kana = bool(re.search(r"[\u3040-\u30ff\uff66-\uff9f]", text))
    han = bool(re.search(r"[\u3400-\u9fff]", text))
    warnings = []
    inferred = (
        "mixed" if hangul and kana else "ko" if hangul else "ja" if kana else "und"
    )
    if inferred == "mixed":
        warnings.append("mixed_scripts")
    elif han and inferred == "und":
        warnings.append("han_language_ambiguous")
    if supplied and supplied not in ("und", "unknown"):
        if not re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", supplied):
            raise ValueError("invalid_language_tag")
        if (inferred != "und" and supplied.lower().split("-")[0] != inferred) or (
            han and supplied.lower().split("-")[0] not in ("zh", "ja", "ko")
        ):
            warnings.append("language_metadata_conflict")
        return supplied, warnings
    if inferred == "und":
        warnings.append("language_unknown")
    return inferred, warnings


def segment_text(text: str, max_chars: int = 4000) -> list[str]:
    if not 1 <= max_chars <= 10000:
        raise ValueError("invalid_segment_limit")
    # Keep separators on the preceding paragraph, including CRLF and blank lines.
    paragraphs = re.findall(r".*?(?:(?:\r?\n){2,}|\Z)", text, flags=re.DOTALL)
    return [
        part[start : start + max_chars]
        for part in paragraphs
        if part
        for start in range(0, len(part), max_chars)
    ]


def numerical_warnings(original: str, translated: str) -> list[str]:
    pattern = r"[+−-]?\d[\d,]*(?:\.\d+)?\s*%?"
    warnings = []

    def numbers(value):
        return Counter(
            re.sub(r"[,\s]", "", token).replace("−", "-")
            for token in re.findall(pattern, value)
        )

    if numbers(original) != numbers(translated):
        warnings.append("numerical_tokens_changed")
    if re.search(r"\d[\d,.]*\s*(?:億|亿|万|萬|兆|억|조|만)", original):
        warnings.append("large_number_units_require_review")
    if Counter(re.findall(r"[$€£¥₩]", original)) != Counter(
        re.findall(r"[$€£¥₩]", translated)
    ):
        warnings.append("currency_notation_changed")
    return warnings


def prepare_text(
    text: str,
    *,
    language: str | None = None,
    target_language="en",
    translator=None,
    max_chars=4000,
) -> TextPreparation:
    source, warnings = detect_language(text, language)
    segments = []
    translations = []
    for original in segment_text(text, max_chars):
        identity = not original.strip() or (
            source == target_language and "language_metadata_conflict" not in warnings
        )
        segments.append(
            TranslationSegment(
                original=original,
                translated=original if identity else None,
                status="identity" if identity else "unavailable",
                failure_reason=None if identity else "translator_unavailable",
            )
        )
        translated = original if identity else None
        if not identity and translator is not None:
            try:
                translated = translator(original, source, target_language)
                if not isinstance(translated, str) or not translated.strip():
                    translated = None
            except Exception:  # noqa: BLE001 - provider failures become sanitized evidence gaps
                translated = None
        translations.append(translated)
    previous = TextPreparation(
        source_language=source,
        supplied_language=language,
        target_language=target_language,
        segments=segments,
        language_warnings=warnings,
    )
    return finalize_translation(
        previous,
        translations,
        missing_reason="translator_unavailable"
        if translator is None
        else "translation_segment_failed",
    )
