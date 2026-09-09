"""Opt-in, source-preserving language preparation policy."""

import re
from typing import Literal

from .multilingual_preparation import (
    TextPreparation,
    TranslationSegment,
    finalize_translation,
)
from .records import Record

LANGUAGE_PREPARATION_POLICY = "text-language-efficient-v2"


class LanguageDecision(Record):
    """Observed metadata and the conservative action chosen from the source text."""

    source_language: str
    supplied_language: str | None
    content_kind: Literal["linguistic", "nonlinguistic"]
    action: Literal["identity", "translate", "retain_original"]
    warnings: list[str]


def preparation_cache_policy(translator=None) -> str:
    """Return the cache namespace the integration must put on ``TextRequest``."""
    translator_policy = (
        getattr(translator, "policy_version", None) or "translator-unavailable"
    )
    if not isinstance(translator_policy, str) or not re.fullmatch(
        r"[A-Za-z0-9._-]+", translator_policy
    ):
        raise ValueError("invalid_translator_policy_version")
    return f"{LANGUAGE_PREPARATION_POLICY}+{translator_policy}"


def _base_language(language: str | None) -> str | None:
    if not language or language.lower() in {"und", "unknown"}:
        return None
    if not re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", language):
        raise ValueError("invalid_language_tag")
    return language.lower().split("-", 1)[0]


def _script_language(text: str) -> tuple[str | None, list[str]]:
    hangul = bool(re.search(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]", text))
    kana = bool(re.search(r"[\u3040-\u30ff\uff66-\uff9f]", text))
    han = bool(re.search(r"[\u3400-\u9fff]", text))
    if hangul and kana:
        return "mixed", ["mixed_scripts"]
    if hangul:
        return "ko", []
    if kana:
        return "ja", []
    if han:
        return "und", ["han_language_ambiguous"]
    return None, []


def _is_nonlinguistic(text: str) -> bool:
    without_urls = re.sub(r"https?://[^\s]+", "", text, flags=re.IGNORECASE)
    return not any(character.isalpha() for character in without_urls)


def assess_language(text: str, supplied: str | None = None) -> LanguageDecision:
    """Choose identity only when supplied English agrees with the source scripts."""
    base = _base_language(supplied)
    if _is_nonlinguistic(text):
        warnings = ["nonlinguistic_content"]
        if base not in {None, "zxx"}:
            warnings.append("language_metadata_conflict")
        return LanguageDecision(
            source_language="zxx",
            supplied_language=supplied,
            content_kind="nonlinguistic",
            action="retain_original",
            warnings=warnings,
        )
    observed, warnings = _script_language(text)
    conflict = bool(base) and (
        (observed in {"ko", "ja"} and base != observed)
        or observed == "mixed"
        or (observed == "und" and base not in {"zh", "ja", "ko"})
        or (observed is None and base == "zxx")
    )
    if observed == "und" and base in {"zh", "ja", "ko"}:
        observed = base
        warnings.remove("han_language_ambiguous")
    if conflict:
        warnings.append("language_metadata_conflict")
    if base == "en" and observed is None:
        return LanguageDecision(
            source_language="en",
            supplied_language=supplied,
            content_kind="linguistic",
            action="identity",
            warnings=[],
        )
    source = observed or (base if base != "zxx" else None) or "und"
    if conflict and observed:
        source = observed
    if source == "und" and "han_language_ambiguous" not in warnings:
        warnings.append("language_unknown")
    return LanguageDecision(
        source_language=source,
        supplied_language=supplied,
        content_kind="linguistic",
        action="translate",
        warnings=warnings,
    )


def segment_text_v2(text: str, max_chars: int = 4000) -> list[str]:
    """Pack paragraphs into bounded segments while retaining every character."""
    if not 1 <= max_chars <= 10000:
        raise ValueError("invalid_segment_limit")
    paragraphs = re.findall(r".*?(?:(?:\r?\n){2,}|\Z)", text, flags=re.DOTALL)
    segments: list[str] = []
    pending = ""
    for paragraph in (part for part in paragraphs if part):
        while paragraph:
            capacity = max_chars - len(pending)
            if len(paragraph) <= capacity:
                pending += paragraph
                break
            if pending:
                segments.append(pending)
                pending = ""
                continue
            segments.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]
    if pending:
        segments.append(pending)
    return segments


def prepare_text_v2(
    text: str,
    *,
    language: str | None = None,
    target_language: str = "en",
    translator=None,
    max_chars: int = 4000,
) -> TextPreparation:
    """Prepare text under the opt-in language-efficient policy."""
    decision = assess_language(text, language)
    originals = segment_text_v2(text, max_chars)
    if decision.action == "retain_original":
        return TextPreparation(
            source_language=decision.source_language,
            supplied_language=decision.supplied_language,
            target_language=target_language,
            segments=[
                TranslationSegment(
                    original=original,
                    translated=None,
                    status="unavailable",
                    failure_reason="nonlinguistic_content",
                )
                for original in originals
            ],
            language_warnings=decision.warnings,
        )
    identity = decision.action == "identity" and target_language == "en"
    segments = [
        TranslationSegment(
            original=original,
            translated=original if identity else None,
            status="identity" if identity else "unavailable",
            failure_reason=None if identity else "translator_unavailable",
        )
        for original in originals
    ]
    translations = [original if identity else None for original in originals]
    if not identity and translator is not None:
        for index, original in enumerate(originals):
            try:
                translated = translator(
                    original, decision.source_language, target_language
                )
                translations[index] = (
                    translated
                    if isinstance(translated, str) and translated.strip()
                    else None
                )
            except Exception:  # noqa: BLE001 - provider errors remain exact gaps
                translations[index] = None
    previous = TextPreparation(
        source_language=decision.source_language,
        supplied_language=decision.supplied_language,
        target_language=target_language,
        segments=segments,
        language_warnings=decision.warnings,
    )
    return finalize_translation(
        previous,
        translations,
        missing_reason=(
            "translator_unavailable"
            if translator is None
            else "translation_segment_failed"
        ),
    )


__all__ = [
    "LANGUAGE_PREPARATION_POLICY",
    "LanguageDecision",
    "assess_language",
    "preparation_cache_policy",
    "prepare_text_v2",
    "segment_text_v2",
]
