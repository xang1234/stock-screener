"""Source-preserving language preparation, with an explicit translator boundary."""

import re
from collections import Counter
from typing import Literal

from pydantic import Field

from .records import Record


class TranslationSegment(Record):
    original: str
    translated: str | None
    status: Literal["translated", "identity", "unavailable"]


class TextPreparation(Record):
    source_language: str
    supplied_language: str | None
    target_language: str
    segments: list[TranslationSegment]
    status: Literal["success", "partial", "unavailable", "needs_review"]
    warnings: list[str] = Field(default_factory=list)


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
    if any(
        unit in original for unit in ("億", "亿", "万", "萬", "兆", "억", "조", "만")
    ):
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
    for original in segment_text(text, max_chars):
        if not original.strip() or (
            source == target_language and "language_metadata_conflict" not in warnings
        ):
            segments.append(
                TranslationSegment(
                    original=original, translated=original, status="identity"
                )
            )
            continue
        translated = None
        if translator is not None:
            try:
                translated = translator(original, source, target_language)
                if not isinstance(translated, str) or not translated.strip():
                    raise ValueError("empty_translation")
                if translated.strip() == original.strip() and source != target_language:
                    warnings.append("translation_unchanged")
                warnings.extend(numerical_warnings(original, translated))
            except Exception:  # noqa: BLE001 - injected provider failures become explicit evidence gaps
                # Provider messages may contain source text or credentials.
                translated = None
                warnings.append("translation_segment_failed")
        else:
            warnings.append("translator_unavailable")
        segments.append(
            TranslationSegment(
                original=original,
                translated=translated,
                status="translated" if translated is not None else "unavailable",
            )
        )
    missing = sum(segment.translated is None for segment in segments)
    if missing:
        status = "unavailable" if missing == len(segments) else "partial"
    else:
        status = "needs_review" if warnings else "success"
    return TextPreparation(
        source_language=source,
        supplied_language=language,
        target_language=target_language,
        segments=segments,
        status=status,
        warnings=list(dict.fromkeys(warnings)),
    )
