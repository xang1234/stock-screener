"""Pure, source-bound selection between captured X and Kimi translations."""

from dataclasses import dataclass
from typing import Literal

from .bundle import sha256
from .preparation_results import TextResult
from .translation_quality import (
    QualityIssue,
    TranslationAssessment,
    assess_translation,
)


@dataclass(frozen=True)
class TranslationSelection:
    source_text_sha256: str
    selected_candidate: Literal["x", "kimi"] | None
    selected_provider: str | None
    eligible: bool
    assessment: TranslationAssessment
    issues: tuple[QualityIssue, ...]
    selected_result: TextResult | None


def _translated_text(original: str, result: TextResult) -> str:
    if (
        result.request.target_language != "en"
        or result.payload.target_language != "en"
    ):
        raise ValueError("translation_candidate_target_mismatch")
    if (
        result.request.input_sha256 != sha256(original.encode())
        or result.source_text != original
    ):
        raise ValueError("translation_candidate_source_mismatch")
    translated = [segment.translated for segment in result.payload.segments]
    if not translated or any(value is None for value in translated):
        return ""
    return "".join(translated)


def _assessment(original, language, result):
    return assess_translation(
        original,
        _translated_text(original, result) if result is not None else "",
        language=language,
    )


def _selection(original, candidate, result, assessment, issues):
    selected = result if assessment.disposition != "fallback" else None
    return TranslationSelection(
        source_text_sha256=sha256(original.encode()),
        selected_candidate=candidate if selected is not None else None,
        selected_provider=selected.request.provider if selected is not None else None,
        eligible=assessment.disposition == "use" and selected is not None,
        assessment=assessment,
        issues=issues,
        selected_result=selected,
    )


def select_translation(
    original: str,
    language: str | None,
    x_result: TextResult | None,
    kimi_result: TextResult | None,
) -> TranslationSelection:
    """Select already-created candidates without performing network or model calls."""
    if x_result is None and kimi_result is None:
        policy_version = _assessment(original, language, None).policy_version
        issue = QualityIssue(
            code="translation_candidate_missing",
            severity="blocker",
            source_excerpt=original,
            translated_excerpt="",
        )
        assessment = TranslationAssessment(
            policy_version=policy_version,
            disposition="fallback",
            issues=(issue,),
        )
        return _selection(original, None, None, assessment, assessment.issues)
    x_assessment = _assessment(original, language, x_result) if x_result else None
    if x_assessment and x_assessment.disposition != "fallback":
        return _selection(
            original, "x", x_result, x_assessment, x_assessment.issues
        )

    if kimi_result is not None:
        kimi_assessment = _assessment(original, language, kimi_result)
        issues = (
            (*x_assessment.issues, *kimi_assessment.issues)
            if x_assessment
            else kimi_assessment.issues
        )
        return _selection(original, "kimi", kimi_result, kimi_assessment, issues)

    assessment = x_assessment or _assessment(original, language, None)
    return _selection(original, None, None, assessment, assessment.issues)
