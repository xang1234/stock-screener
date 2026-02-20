"""Pattern aggregation entrypoint for Setup Engine analysis layer.

TODO(SE-B7): Emit typed SetupEngineReport-compatible aggregation output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from app.analysis.patterns.calibration import (
    aggregation_rank_score,
    calibrate_candidates_for_aggregation,
)
from app.analysis.patterns.config import (
    DEFAULT_SETUP_ENGINE_PARAMETERS,
    SetupEngineParameters,
    assert_valid_setup_engine_parameters,
)
from app.analysis.patterns.detectors import (
    DetectorOutcome,
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
class DetectorExecutionTrace:
    """Deterministic execution trace row for one detector call."""

    execution_index: int
    detector_name: str
    outcome: str
    candidate_count: int
    passed_checks: tuple[str, ...]
    failed_checks: tuple[str, ...]
    warnings: tuple[str, ...]
    error_detail: str | None
    elapsed_ms: float


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
    detector_traces: tuple[DetectorExecutionTrace, ...]


@dataclass(frozen=True)
class PrimarySelectionResult:
    """Primary-candidate selection output with deterministic rationale fields."""

    primary: PatternCandidate | None
    used_confidence_fallback: bool
    structural_tie_break_applied: bool
    rationale: str


_TRIGGER_PATTERN_FAMILIES = frozenset({"nr7_inside_day", "first_pullback"})
_STRUCTURAL_TIE_EPSILON = 0.015


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
        detector_traces: list[DetectorExecutionTrace] = []

        for idx, detector in enumerate(self._detectors):
            t0 = time.perf_counter()
            result = detector.detect_safe(detector_input, parameters)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            detector_traces.append(
                DetectorExecutionTrace(
                    execution_index=idx,
                    detector_name=detector.name,
                    outcome=result.outcome.value,
                    candidate_count=len(result.candidates),
                    passed_checks=tuple(result.passed_checks),
                    failed_checks=tuple(result.failed_checks),
                    warnings=tuple(result.warnings),
                    error_detail=result.error_detail,
                    elapsed_ms=elapsed_ms,
                )
            )

            if result.outcome == DetectorOutcome.ERROR:
                failed_checks.append(
                    f"{detector.name}:{DetectorOutcome.ERROR.value}"
                )
                if result.error_detail:
                    diagnostics.append(
                        f"{detector.name}:{result.error_detail}"
                    )
                continue

            diagnostics.extend(result.warnings)
            passed_checks.extend(result.passed_checks)
            failed_checks.extend(result.failed_checks)

            # Candidates are already coerced by detect_safe().
            candidates.extend(result.candidates)

        calibrated_candidates = list(calibrate_candidates_for_aggregation(candidates))
        calibration_applied = bool(calibrated_candidates)
        candidates = calibrated_candidates

        if policy_result is not None:
            failed_checks.extend(policy_failed_checks(policy_result))
            invalidation_flags.extend(policy_invalidation_flags(policy_result))
            if policy_result["status"] == "insufficient":
                candidates = []
                calibration_applied = False

        selection = _select_primary_candidate(candidates, parameters=parameters)
        primary = selection.primary
        diagnostics.append(selection.rationale)

        if calibration_applied and candidates:
            passed_checks.append("cross_detector_calibration_applied")
        if detector_traces:
            passed_checks.append("detector_pipeline_executed")

        if primary is None:
            failed_checks.append("no_primary_pattern")
        else:
            if selection.used_confidence_fallback:
                failed_checks.append("primary_pattern_below_confidence_floor")
                passed_checks.append("primary_pattern_fallback_selected")
            else:
                passed_checks.append("primary_pattern_selected")
            if selection.structural_tie_break_applied:
                passed_checks.append("primary_pattern_structural_tie_break_applied")
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
            detector_traces=tuple(detector_traces),
        )


def _select_primary_candidate(
    candidates: Sequence[PatternCandidate],
    *,
    parameters: SetupEngineParameters,
) -> PrimarySelectionResult:
    if not candidates:
        return PrimarySelectionResult(
            primary=None,
            used_confidence_fallback=False,
            structural_tie_break_applied=False,
            rationale="primary_selection:no_candidates_available",
        )

    confidence_floor_pct = float(parameters.pattern_confidence_min_pct)
    qualified = [
        candidate
        for candidate in candidates
        if (_confidence_pct(candidate) is not None)
        and (_confidence_pct(candidate) >= confidence_floor_pct)
    ]

    used_confidence_fallback = len(qualified) == 0
    selection_pool = qualified if qualified else list(candidates)
    primary, structural_tie_break_applied = _pick_primary_candidate(selection_pool)
    if primary is None:
        return PrimarySelectionResult(
            primary=None,
            used_confidence_fallback=used_confidence_fallback,
            structural_tie_break_applied=False,
            rationale="primary_selection:selection_pool_empty",
        )

    mode = (
        "fallback_below_confidence_floor"
        if used_confidence_fallback
        else "qualified_rank_selection"
    )
    tie_break = (
        "structural_tie_break_applied"
        if structural_tie_break_applied
        else "no_structural_tie_break"
    )
    return PrimarySelectionResult(
        primary=primary,
        used_confidence_fallback=used_confidence_fallback,
        structural_tie_break_applied=structural_tie_break_applied,
        rationale=(
            f"primary_selection:{mode};{tie_break};"
            f"min_confidence_pct={confidence_floor_pct:.2f}"
        ),
    )


def _pick_primary_candidate(
    candidates: Sequence[PatternCandidate],
) -> tuple[PatternCandidate | None, bool]:
    if not candidates:
        return None, False

    scored: list[tuple[PatternCandidate, int, float]] = [
        (candidate, index, aggregation_rank_score(candidate))
        for index, candidate in enumerate(candidates)
    ]
    base_winner = max(
        scored,
        key=lambda row: _rank_tie_break_key(
            row[0],
            rank_score=row[2],
            index=row[1],
        ),
    )
    base_candidate, _, base_rank_score = base_winner
    if _is_structural_candidate(base_candidate):
        return base_candidate, False

    structural_close_candidates = [
        row
        for row in scored
        if _is_structural_candidate(row[0])
        and (base_rank_score - row[2]) <= _STRUCTURAL_TIE_EPSILON
    ]
    if not structural_close_candidates:
        return base_candidate, False

    structural_winner = max(
        structural_close_candidates,
        key=lambda row: _rank_tie_break_key(
            row[0],
            rank_score=row[2],
            index=row[1],
        ),
    )
    return structural_winner[0], True


def _confidence_ratio(candidate: PatternCandidate) -> float:
    raw = candidate.get("confidence")
    if raw is None and candidate.get("confidence_pct") is not None:
        raw = float(candidate["confidence_pct"]) / 100.0
    if raw is None:
        return float("-inf")
    return float(raw)


def _confidence_pct(candidate: PatternCandidate) -> float | None:
    if candidate.get("confidence_pct") is not None:
        return float(candidate["confidence_pct"])
    confidence = candidate.get("confidence")
    if confidence is None:
        return None
    return float(confidence) * 100.0


def _distance_score(candidate: PatternCandidate) -> float:
    distance = candidate.get("distance_to_pivot_pct")
    if distance is None:
        return float("-inf")
    return -abs(float(distance))


def _is_structural_candidate(candidate: PatternCandidate) -> bool:
    family = str(
        candidate.get("source_detector")
        or candidate.get("pattern")
        or ""
    ).lower()
    return family not in _TRIGGER_PATTERN_FAMILIES


def _rank_tie_break_key(
    candidate: PatternCandidate,
    *,
    rank_score: float,
    index: int,
) -> tuple[float, float, float, float, str, str, int]:
    quality = (
        float(candidate.get("quality_score"))
        if candidate.get("quality_score") is not None
        else float("-inf")
    )
    readiness = (
        float(candidate.get("readiness_score"))
        if candidate.get("readiness_score") is not None
        else float("-inf")
    )
    pattern = str(candidate.get("pattern") or "")
    source_detector = str(candidate.get("source_detector") or "")
    return (
        rank_score,
        _confidence_ratio(candidate),
        readiness + quality,
        _distance_score(candidate),
        pattern,
        source_detector,
        -index,
    )


def _stable_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique
