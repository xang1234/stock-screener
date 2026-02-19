"""Pattern aggregation entrypoint for Setup Engine analysis layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.analysis.patterns.config import (
    DEFAULT_SETUP_ENGINE_PARAMETERS,
    SetupEngineParameters,
    assert_valid_setup_engine_parameters,
)
from app.analysis.patterns.detectors import (
    PatternDetector,
    PatternDetectorInput,
    default_pattern_detectors,
)
from app.analysis.patterns.models import PatternCandidate
from app.analysis.patterns.policy import (
    SetupEngineDataPolicyResult,
    policy_failed_checks,
    policy_invalidation_flags,
)


@dataclass(frozen=True)
class AggregatedPatternOutput:
    """Normalized detector output consumed by scanner payload assembly."""

    pattern_primary: str | None
    pattern_confidence: float | None
    pivot_price: float | None
    pivot_type: str | None
    pivot_date: str | None
    candidates: tuple[PatternCandidate, ...]
    passed_checks: tuple[str, ...]
    failed_checks: tuple[str, ...]
    key_levels: dict[str, float | None]
    invalidation_flags: tuple[str, ...]
    diagnostics: tuple[str, ...]


class SetupEngineAggregator:
    """Run detectors and normalize candidates for setup_engine payload use."""

    def __init__(self, detectors: Sequence[PatternDetector] | None = None):
        self._detectors: tuple[PatternDetector, ...] = tuple(
            detectors if detectors is not None else default_pattern_detectors()
        )

    def aggregate(
        self,
        detector_input: PatternDetectorInput,
        *,
        parameters: SetupEngineParameters = DEFAULT_SETUP_ENGINE_PARAMETERS,
        policy_result: SetupEngineDataPolicyResult | None = None,
    ) -> AggregatedPatternOutput:
        """Run detectors without leaking scanner concerns into analysis layer."""
        assert_valid_setup_engine_parameters(parameters)

        candidates: list[PatternCandidate] = []
        passed_checks: list[str] = []
        failed_checks: list[str] = []
        diagnostics: list[str] = []
        invalidation_flags: list[str] = []
        key_levels: dict[str, float | None] = {}

        for detector in self._detectors:
            try:
                result = detector.detect(detector_input, parameters)
            except Exception as exc:
                # Degrade gracefully: detector failures become explicit diagnostics.
                failed_checks.append(f"{detector.name}:detector_error")
                diagnostics.append(f"{detector.name}:error:{exc.__class__.__name__}")
                continue

            diagnostics.extend(result.warnings)
            passed_checks.extend(result.passed_checks)
            failed_checks.extend(result.failed_checks)
            if result.candidate is not None:
                candidates.append(result.candidate)

        primary = _pick_primary_candidate(candidates)

        if policy_result is not None:
            failed_checks.extend(policy_failed_checks(policy_result))
            invalidation_flags.extend(policy_invalidation_flags(policy_result))
            if policy_result["status"] == "insufficient":
                candidates = []
                primary = None

        if primary is None:
            failed_checks.append("no_primary_pattern")
        else:
            passed_checks.append("primary_pattern_selected")
            key_levels["pivot_price"] = primary.get("pivot_price")

        return AggregatedPatternOutput(
            pattern_primary=primary.get("pattern") if primary else None,
            pattern_confidence=primary.get("confidence_pct") if primary else None,
            pivot_price=primary.get("pivot_price") if primary else None,
            pivot_type=primary.get("pivot_type") if primary else None,
            pivot_date=primary.get("pivot_date") if primary else None,
            candidates=tuple(candidates),
            passed_checks=tuple(_stable_unique(passed_checks)),
            failed_checks=tuple(_stable_unique(failed_checks)),
            key_levels=key_levels,
            invalidation_flags=tuple(_stable_unique(invalidation_flags)),
            diagnostics=tuple(diagnostics),
        )


def _pick_primary_candidate(candidates: Sequence[PatternCandidate]) -> PatternCandidate | None:
    if not candidates:
        return None

    def _confidence(candidate: PatternCandidate) -> float:
        raw = candidate.get("confidence_pct")
        if raw is None:
            return float("-inf")
        return float(raw)

    return max(candidates, key=_confidence)


def _stable_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique
