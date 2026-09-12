"""Conservative, versioned translation-adequacy decisions.

The policy proves only the finite token relationships normalized by
``translation_normalization``. Other semantic equivalence remains review work.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from .translation_normalization import (
    NormalizedText,
    ambiguous_values,
    normalize_decimal_percentages,
    normalize_text,
    quantity_associations,
    quantity_counter,
)

QUALITY_POLICY_V1 = "translation-quality-v1"
QUALITY_POLICY_V2 = "translation-quality-v2"
QualityPolicy = Literal["translation-quality-v1", "translation-quality-v2"]


@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: Literal["blocker", "review", "info"]
    source_excerpt: str
    translated_excerpt: str


@dataclass(frozen=True)
class TranslationAssessment:
    policy_version: str
    disposition: Literal["use", "fallback", "review"]
    issues: tuple[QualityIssue, ...]


def _issue(
    code: str,
    severity: Literal["blocker", "review", "info"],
    source: str,
    target: str,
) -> QualityIssue:
    return QualityIssue(code, severity, source, target)


_QUOTED_TEXT = re.compile(r'[『「“"]([^』」”"]+)[』」”"]')


def _repeated_quoted_quantities(
    source_quantities: Counter,
    target_quantities: Counter,
    original: str,
    translated: str,
) -> bool:
    retained = Counter()
    for match in _QUOTED_TEXT.finditer(original):
        content = match.group(1)
        if re.search(r"\d", content) and content in translated:
            retained.update(quantity_counter(normalize_text(content).quantities))
    return bool(retained) and target_quantities == source_quantities + retained


def _compare_quantities(
    source: NormalizedText, target: NormalizedText, original: str, translated: str
) -> QualityIssue | None:
    source_unambiguous = quantity_counter(source.quantities, include_ambiguous=False)
    target_unambiguous = quantity_counter(target.quantities, include_ambiguous=False)
    source_ambiguous = ambiguous_values(source.quantities)
    target_ambiguous = ambiguous_values(target.quantities)

    if not source_ambiguous and not target_ambiguous:
        if source_unambiguous == target_unambiguous:
            return None
        if _repeated_quoted_quantities(
            source_unambiguous,
            target_unambiguous,
            original,
            translated,
        ):
            return _issue("quantity_repeated_in_gloss", "review", original, translated)
        source_values = Counter(value for value, _unit in source_unambiguous.elements())
        target_values = Counter(value for value, _unit in target_unambiguous.elements())
        code = (
            "quantity_unit_changed"
            if source_values == target_values
            else "quantity_value_changed"
        )
        return _issue(code, "blocker", original, translated)

    source_values = Counter(
        quantity.value.normalize() for quantity in source.quantities
    )
    target_values = Counter(
        quantity.value.normalize() for quantity in target.quantities
    )
    if source_values != target_values:
        return _issue("quantity_value_changed", "blocker", original, translated)

    for value in source_values:
        source_units = Counter(
            quantity.unit
            for quantity in source.quantities
            if quantity.value.normalize() == value and quantity.unit != "ambiguous"
        )
        target_units = Counter(
            quantity.unit
            for quantity in target.quantities
            if quantity.value.normalize() == value and quantity.unit != "ambiguous"
        )
        common = source_units & target_units
        if (
            sum((source_units - common).values()) > target_ambiguous[value]
            or sum((target_units - common).values()) > source_ambiguous[value]
        ):
            return _issue("quantity_unit_changed", "blocker", original, translated)
    return _issue("ambiguous_quantity_unit", "review", original, translated)


def _token_issues(
    source: NormalizedText,
    target: NormalizedText,
    original: str,
    translated: str,
    policy_version: QualityPolicy,
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    if source.temporal != target.temporal:
        issues.append(_issue("temporal_value_changed", "blocker", original, translated))
    elif source.dates != target.dates and (source.dates or target.dates):
        issues.append(
            _issue("date_association_changed", "review", original, translated)
        )
    if source.identifiers != target.identifiers:
        issues.append(_issue("identifier_changed", "blocker", original, translated))
    quantity_issue = _compare_quantities(source, target, original, translated)
    if quantity_issue:
        issues.append(quantity_issue)
    if source.directions != target.directions and (
        source.directions or target.directions
    ):
        issues.append(
            _issue(
                "direction_changed"
                if policy_version == QUALITY_POLICY_V1
                else "direction_unanchored",
                "blocker" if policy_version == QUALITY_POLICY_V1 else "review",
                original,
                translated,
            )
        )
    if source.negated != target.negated:
        issues.append(
            _issue(
                "negation_changed"
                if policy_version == QUALITY_POLICY_V1
                else "negation_unanchored",
                "blocker" if policy_version == QUALITY_POLICY_V1 else "review",
                original,
                translated,
            )
        )
    if source.metrics != target.metrics and (source.metrics or target.metrics):
        issues.append(_issue("metric_changed", "review", original, translated))
    return issues


def _fragment_issue(
    source: NormalizedText, target: NormalizedText, original: str, translated: str
) -> QualityIssue | None:
    source_prose = source.text.strip()
    target_prose = target.text.strip()
    target_words = re.findall(r"\b\w+\b", target_prose, re.UNICODE)
    if (
        len(source_prose) >= 30
        and len(target_words) <= 2
        and len(target_prose) < len(source_prose) * 0.2
    ):
        return _issue("possible_translation_fragment", "review", original, translated)
    return None


def _association_context(associations: Counter) -> Counter:
    return Counter(association[:4] for association in associations.elements())


def assess_translation(
    original: str,
    translated: str,
    *,
    language: str | None,
    policy_version: QualityPolicy = QUALITY_POLICY_V2,
) -> TranslationAssessment:
    """Assess a derivative without mutating canonical text or archived status."""
    if policy_version not in {QUALITY_POLICY_V1, QUALITY_POLICY_V2}:
        raise ValueError("unsupported_translation_quality_policy")
    source_text = (
        original
        if policy_version == QUALITY_POLICY_V1
        else normalize_decimal_percentages(original)
    )
    target_text = (
        translated
        if policy_version == QUALITY_POLICY_V1
        else normalize_decimal_percentages(translated)
    )
    source = normalize_text(source_text)
    target = normalize_text(target_text)
    issues = _token_issues(
        source,
        target,
        original,
        translated,
        policy_version,
    )

    if original.strip() and not translated.strip():
        issues.insert(0, _issue("translation_empty", "blocker", original, translated))

    unanchored_polarity = {"direction_unanchored", "negation_unanchored"}
    if not issues or (
        policy_version == QUALITY_POLICY_V2
        and all(issue.code in unanchored_polarity for issue in issues)
    ):
        source_associations = quantity_associations(source.text)
        target_associations = quantity_associations(target.text)
        if (
            source_associations
            and target_associations
            and source_associations != target_associations
        ):
            same_value_context = _association_context(
                source_associations
            ) == _association_context(target_associations)
            issues.append(
                _issue(
                    "polarity_association_changed"
                    if same_value_context
                    else "quantity_association_changed",
                    "blocker" if same_value_context else "review",
                    original,
                    translated,
                )
            )

    fragment = _fragment_issue(source, target, original, translated)
    if fragment:
        issues.append(fragment)

    if not issues:
        base_language = (language or "").lower().split("-", 1)[0]
        source_prose = re.sub(r"\s+", " ", source.text).strip()
        target_prose = re.sub(r"\s+", " ", target.text).strip()
        if source_prose == target_prose and base_language not in {"en", "zxx", "art"}:
            issues.append(
                _issue("translation_unchanged", "review", original, translated)
            )
        elif original != translated and not source.has_supported_semantics:
            issues.append(
                _issue(
                    "semantic_equivalence_unverified", "review", original, translated
                )
            )

    disposition: Literal["use", "fallback", "review"]
    if any(issue.severity == "blocker" for issue in issues):
        disposition = "fallback"
    elif any(issue.severity == "review" for issue in issues):
        disposition = "review"
    else:
        disposition = "use"
    return TranslationAssessment(policy_version, disposition, tuple(issues))


__all__ = [
    "QUALITY_POLICY_V1",
    "QUALITY_POLICY_V2",
    "QualityIssue",
    "QualityPolicy",
    "TranslationAssessment",
    "assess_translation",
]
