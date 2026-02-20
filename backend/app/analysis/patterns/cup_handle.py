"""Cup-with-handle detector entrypoint.

Expected input orientation:
- Weekly swing features in chronological order.
- Candidate enumeration must stay deterministic.

TODO(SE-C4b): Implement handle detection and upper-half constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.analysis.patterns.config import SetupEngineParameters
from app.analysis.patterns.detectors.base import (
    PatternDetector,
    PatternDetectorInput,
    PatternDetectorResult,
)
from app.analysis.patterns.models import PatternCandidateModel
from app.analysis.patterns.normalization import normalize_detector_input_ohlcv
from app.analysis.patterns.technicals import detect_swings

_CUP_MIN_WEEKS = 6
_CUP_MAX_WEEKS = 65
_CUP_MIN_DEPTH_PCT = 8.0
_CUP_MAX_DEPTH_PCT = 50.0
_CUP_MIN_RECOVERY_PCT = 90.0
_MAX_CUP_CANDIDATES = 5
_MAX_SWING_PAIR_EVALUATIONS = 3_000


@dataclass(frozen=True)
class _CupStructure:
    left_idx: int
    low_idx: int
    right_idx: int
    duration_weeks: int
    depth_pct: float
    recovery_strength_pct: float
    curvature_balance: float
    recency_weeks: int
    structure_score: float


@dataclass(frozen=True)
class _CupParseResult:
    candidates: tuple[_CupStructure, ...]
    capped: bool


class CupHandleDetector(PatternDetector):
    """Compile-safe entrypoint for cup-with-handle detection."""

    name = "cup_with_handle"

    def detect(
        self,
        detector_input: PatternDetectorInput,
        parameters: SetupEngineParameters,
    ) -> PatternDetectorResult:
        del parameters
        normalized = normalize_detector_input_ohlcv(
            features=detector_input.features,
            timeframe="weekly",
            min_bars=20,
            feature_key="weekly_ohlcv",
            fallback_bar_count=detector_input.weekly_bars,
        )
        if not normalized.prerequisites_ok:
            return PatternDetectorResult.insufficient_data(
                self.name, normalized=normalized
            )

        if normalized.frame is None:
            return PatternDetectorResult.insufficient_data(
                self.name,
                failed_checks=("missing_weekly_ohlcv_for_cup_structure",),
                warnings=normalized.warnings,
            )

        parse_result = _find_cup_structures(normalized.frame)
        if not parse_result.candidates:
            return PatternDetectorResult.no_detection(
                self.name,
                failed_checks=("cup_structure_not_found",),
                warnings=normalized.warnings,
            )

        warnings = list(normalized.warnings)
        if parse_result.capped:
            warnings.append("cup_search_capped_for_deterministic_runtime")

        high = normalized.frame["High"]
        low = normalized.frame["Low"]
        candidates: list[PatternCandidateModel] = []

        for rank, structure in enumerate(
            parse_result.candidates[:_MAX_CUP_CANDIDATES], start=1
        ):
            left_date = high.index[structure.left_idx].date().isoformat()
            low_date = low.index[structure.low_idx].date().isoformat()
            right_date = high.index[structure.right_idx].date().isoformat()
            left_lip_price = float(high.iat[structure.left_idx])
            low_price = float(low.iat[structure.low_idx])
            right_lip_price = float(high.iat[structure.right_idx])
            confidence = min(
                0.95,
                max(0.05, 0.35 + structure.structure_score * 0.55),
            )
            quality_score = min(
                100.0,
                max(
                    0.0,
                    35.0
                    + structure.curvature_balance * 30.0
                    + min(structure.recovery_strength_pct, 100.0) * 0.25,
                ),
            )
            readiness_score = max(
                0.0,
                min(100.0, 100.0 - structure.recency_weeks * 1.5),
            )

            candidates.append(
                PatternCandidateModel(
                    pattern=self.name,
                    timeframe="weekly",
                    source_detector=self.name,
                    pivot_price=right_lip_price,
                    pivot_type="cup_right_lip",
                    pivot_date=right_date,
                    confidence=confidence,
                    quality_score=quality_score,
                    readiness_score=readiness_score,
                    metrics={
                        "cup_rank": rank,
                        "left_lip_date": left_date,
                        "cup_low_date": low_date,
                        "right_lip_date": right_date,
                        "left_lip_price": round(left_lip_price, 4),
                        "cup_low_price": round(low_price, 4),
                        "right_lip_price": round(right_lip_price, 4),
                        "cup_duration_weeks": structure.duration_weeks,
                        "cup_depth_pct": round(structure.depth_pct, 4),
                        "recovery_strength_pct": round(
                            structure.recovery_strength_pct, 4
                        ),
                        "curvature_balance_pct": round(
                            structure.curvature_balance * 100.0, 4
                        ),
                        "cup_recency_weeks": structure.recency_weeks,
                    },
                    checks={
                        "cup_duration_in_range": (
                            _CUP_MIN_WEEKS
                            <= structure.duration_weeks
                            <= _CUP_MAX_WEEKS
                        ),
                        "cup_depth_in_range": (
                            _CUP_MIN_DEPTH_PCT
                            <= structure.depth_pct
                            <= _CUP_MAX_DEPTH_PCT
                        ),
                        "cup_recovery_in_range": structure.recovery_strength_pct
                        >= _CUP_MIN_RECOVERY_PCT,
                        "cup_curvature_balanced": structure.curvature_balance
                        >= 0.35,
                    },
                    notes=(
                        "cup_structure_candidate_only",
                        "handle_validation_pending",
                    ),
                )
            )

        return PatternDetectorResult.detected(
            self.name,
            tuple(candidates),
            passed_checks=("cup_structure_candidates_found",),
            warnings=tuple(warnings),
        )


def _find_cup_structures(frame: pd.DataFrame) -> _CupParseResult:
    highs = frame["High"]
    lows = frame["Low"]
    swings = detect_swings(highs, lows, left=2, right=2)
    swing_high_positions = [
        idx
        for idx, is_high in enumerate(
            swings["swing_high"].tolist()
        )
        if bool(is_high)
    ]

    if len(swing_high_positions) < 2:
        return _CupParseResult(candidates=(), capped=False)

    structures: list[_CupStructure] = []
    pair_evaluations = 0
    capped = False
    n = len(frame)

    for left_position, left_idx in enumerate(swing_high_positions[:-1]):
        left_lip_price = float(highs.iat[left_idx])
        if left_lip_price <= 0.0:
            continue

        for right_idx in swing_high_positions[left_position + 1 :]:
            duration_weeks = right_idx - left_idx
            if duration_weeks < _CUP_MIN_WEEKS:
                continue
            if duration_weeks > _CUP_MAX_WEEKS:
                break

            pair_evaluations += 1
            if pair_evaluations > _MAX_SWING_PAIR_EVALUATIONS:
                capped = True
                break

            segment_lows = lows.iloc[left_idx : right_idx + 1]
            low_offset = int(segment_lows.to_numpy(dtype=float).argmin())
            low_idx = left_idx + low_offset
            if low_idx <= left_idx or low_idx >= right_idx:
                continue

            cup_low_price = float(lows.iat[low_idx])
            depth_pct = ((left_lip_price - cup_low_price) / left_lip_price) * 100.0
            if depth_pct < _CUP_MIN_DEPTH_PCT or depth_pct > _CUP_MAX_DEPTH_PCT:
                continue

            right_lip_price = float(highs.iat[right_idx])
            recovery_strength_pct = (right_lip_price / left_lip_price) * 100.0
            if recovery_strength_pct < _CUP_MIN_RECOVERY_PCT:
                continue

            down_weeks = low_idx - left_idx
            up_weeks = right_idx - low_idx
            if down_weeks <= 0 or up_weeks <= 0:
                continue

            down_slope = (left_lip_price - cup_low_price) / down_weeks
            up_slope = (right_lip_price - cup_low_price) / up_weeks
            slope_scale = max(abs(down_slope), abs(up_slope), 1e-9)
            slope_imbalance = abs(down_slope - up_slope) / slope_scale
            curvature_balance = max(0.0, 1.0 - slope_imbalance)
            if min(down_weeks, up_weeks) <= 2:
                curvature_balance *= 0.6

            recency_weeks = n - 1 - right_idx
            recency_component = max(0.0, 1.0 - (recency_weeks / _CUP_MAX_WEEKS))
            structure_score = (
                0.45 * curvature_balance
                + 0.35 * min(recovery_strength_pct / 100.0, 1.05)
                + 0.20 * recency_component
            )

            structures.append(
                _CupStructure(
                    left_idx=left_idx,
                    low_idx=low_idx,
                    right_idx=right_idx,
                    duration_weeks=duration_weeks,
                    depth_pct=depth_pct,
                    recovery_strength_pct=recovery_strength_pct,
                    curvature_balance=curvature_balance,
                    recency_weeks=recency_weeks,
                    structure_score=structure_score,
                )
            )

        if capped:
            break

    structures.sort(
        key=lambda structure: (
            -structure.structure_score,
            structure.recency_weeks,
            abs(100.0 - structure.recovery_strength_pct),
            abs(22.0 - structure.depth_pct),
            structure.duration_weeks,
            -structure.right_idx,
            -structure.left_idx,
        )
    )
    return _CupParseResult(candidates=tuple(structures), capped=capped)
