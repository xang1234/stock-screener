"""Cup-with-handle detector entrypoint.

Expected input orientation:
- Weekly swing features in chronological order.
- Candidate enumeration must stay deterministic.
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
_HANDLE_MIN_WEEKS = 1
_HANDLE_MAX_WEEKS = 5
_HANDLE_MAX_DEPTH_PCT = 15.0
_HANDLE_MAX_VOLUME_RATIO = 1.10


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


@dataclass(frozen=True)
class _HandleCandidate:
    structure: _CupStructure
    start_idx: int
    end_idx: int
    duration_weeks: int
    handle_high: float
    handle_low: float
    pivot_idx: int
    handle_depth_pct: float
    upper_half_floor: float
    upper_half_margin_pct: float
    volume_ratio: float
    score: float
    recency_weeks: int


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
        volume = normalized.frame["Volume"]
        candidates: list[PatternCandidateModel] = []
        failed_reasons: list[str] = []

        for cup_rank, structure in enumerate(
            parse_result.candidates[:_MAX_CUP_CANDIDATES], start=1
        ):
            handle_candidate, rejected = _find_best_handle_candidate(
                normalized.frame,
                structure=structure,
            )
            failed_reasons.extend(rejected)
            if handle_candidate is None:
                continue

            left_date = high.index[structure.left_idx].date().isoformat()
            low_date = low.index[structure.low_idx].date().isoformat()
            right_date = high.index[structure.right_idx].date().isoformat()
            handle_start_date = high.index[
                handle_candidate.start_idx
            ].date().isoformat()
            handle_end_date = high.index[
                handle_candidate.end_idx
            ].date().isoformat()
            left_lip_price = float(high.iat[structure.left_idx])
            low_price = float(low.iat[structure.low_idx])
            right_lip_price = float(high.iat[structure.right_idx])
            right_side_volume_mean = float(
                volume.iloc[structure.low_idx : structure.right_idx + 1].mean()
            )
            handle_volume_mean = float(
                volume.iloc[
                    handle_candidate.start_idx : handle_candidate.end_idx + 1
                ].mean()
            )
            depth_component = max(
                0.0,
                1.0 - (handle_candidate.handle_depth_pct / _HANDLE_MAX_DEPTH_PCT),
            )
            volume_component = max(
                0.0,
                1.0 - max(0.0, handle_candidate.volume_ratio - 1.0),
            )
            confidence = min(
                0.95,
                max(
                    0.05,
                    0.30
                    + (structure.structure_score * 0.40)
                    + (depth_component * 0.18)
                    + (volume_component * 0.12),
                ),
            )
            quality_score = min(
                100.0,
                max(
                    0.0,
                    42.0
                    + (structure.curvature_balance * 20.0)
                    + (depth_component * 20.0)
                    + (min(structure.recovery_strength_pct, 100.0) * 0.18)
                    + min(handle_candidate.upper_half_margin_pct, 8.0),
                ),
            )
            readiness_score = max(
                0.0,
                min(
                    100.0,
                    78.0
                    + (volume_component * 10.0)
                    - (handle_candidate.recency_weeks * 1.2),
                ),
            )

            candidates.append(
                PatternCandidateModel(
                    pattern=self.name,
                    timeframe="weekly",
                    source_detector=self.name,
                    pivot_price=handle_candidate.handle_high,
                    pivot_type="handle_high",
                    pivot_date=high.index[handle_candidate.pivot_idx]
                    .date()
                    .isoformat(),
                    confidence=confidence,
                    quality_score=quality_score,
                    readiness_score=readiness_score,
                    metrics={
                        "cup_rank": cup_rank,
                        "left_lip_date": left_date,
                        "cup_low_date": low_date,
                        "right_lip_date": right_date,
                        "handle_start_date": handle_start_date,
                        "handle_end_date": handle_end_date,
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
                        "handle_duration_weeks": handle_candidate.duration_weeks,
                        "handle_high": round(handle_candidate.handle_high, 4),
                        "handle_low": round(handle_candidate.handle_low, 4),
                        "handle_depth_pct": round(
                            handle_candidate.handle_depth_pct, 4
                        ),
                        "handle_upper_half_floor": round(
                            handle_candidate.upper_half_floor, 4
                        ),
                        "handle_upper_half_margin_pct": round(
                            handle_candidate.upper_half_margin_pct, 4
                        ),
                        "right_side_volume_mean": round(
                            right_side_volume_mean, 4
                        ),
                        "handle_volume_mean": round(handle_volume_mean, 4),
                        "handle_volume_ratio_vs_right_side": round(
                            handle_candidate.volume_ratio, 6
                        ),
                        "handle_recency_weeks": handle_candidate.recency_weeks,
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
                        "handle_duration_in_range": (
                            _HANDLE_MIN_WEEKS
                            <= handle_candidate.duration_weeks
                            <= _HANDLE_MAX_WEEKS
                        ),
                        "handle_depth_in_range": handle_candidate.handle_depth_pct
                        <= _HANDLE_MAX_DEPTH_PCT,
                        "handle_in_upper_half": handle_candidate.handle_low
                        >= handle_candidate.upper_half_floor,
                        "handle_volume_contracting": handle_candidate.volume_ratio
                        <= _HANDLE_MAX_VOLUME_RATIO,
                    },
                    notes=(
                        "cup_and_handle_validated",
                        "pivot_from_handle_high",
                    ),
                )
            )

        if not candidates:
            return PatternDetectorResult.no_detection(
                self.name,
                failed_checks=tuple(_stable_unique(failed_reasons)),
                warnings=tuple(warnings),
            )

        return PatternDetectorResult.detected(
            self.name,
            tuple(candidates),
            passed_checks=(
                "cup_structure_candidates_found",
                "handle_candidates_validated",
            ),
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


def _find_best_handle_candidate(
    frame: pd.DataFrame,
    *,
    structure: _CupStructure,
) -> tuple[_HandleCandidate | None, tuple[str, ...]]:
    start_idx = structure.right_idx + 1
    if start_idx >= len(frame):
        return None, ("handle_missing_structure",)

    highs = frame["High"]
    lows = frame["Low"]
    volumes = frame["Volume"]
    left_lip_price = float(highs.iat[structure.left_idx])
    cup_low_price = float(lows.iat[structure.low_idx])
    cup_range = max(left_lip_price - cup_low_price, 1e-9)
    upper_half_floor = cup_low_price + (cup_range * 0.5)

    right_side_volume_mean = float(
        volumes.iloc[structure.low_idx : structure.right_idx + 1].mean()
    )
    if right_side_volume_mean <= 0.0 or pd.isna(right_side_volume_mean):
        right_side_volume_mean = 1.0

    rejected: list[str] = []
    valid_candidates: list[_HandleCandidate] = []
    n = len(frame)

    for duration_weeks in range(_HANDLE_MIN_WEEKS, _HANDLE_MAX_WEEKS + 1):
        end_idx = start_idx + duration_weeks - 1
        if end_idx >= n:
            break

        segment = frame.iloc[start_idx : end_idx + 1]
        handle_high_values = segment["High"].to_numpy(dtype=float)
        handle_low_values = segment["Low"].to_numpy(dtype=float)
        handle_volume_values = segment["Volume"].to_numpy(dtype=float)

        handle_high = float(handle_high_values.max())
        handle_low = float(handle_low_values.min())
        if handle_high <= 0.0:
            rejected.append("handle_missing_structure")
            continue

        handle_depth_pct = ((handle_high - handle_low) / handle_high) * 100.0
        upper_half_margin_pct = ((handle_low - upper_half_floor) / cup_range) * 100.0
        handle_volume_mean = float(handle_volume_values.mean())
        volume_ratio = handle_volume_mean / right_side_volume_mean

        depth_ok = handle_depth_pct <= _HANDLE_MAX_DEPTH_PCT
        upper_half_ok = handle_low >= upper_half_floor
        volume_ok = volume_ratio <= _HANDLE_MAX_VOLUME_RATIO

        if not depth_ok:
            rejected.append("handle_depth_rejected")
        if not upper_half_ok:
            rejected.append("handle_upper_half_rejected")
        if not volume_ok:
            rejected.append("handle_volume_rejected")
        if not (depth_ok and upper_half_ok and volume_ok):
            continue

        pivot_offset = int(handle_high_values.argmax())
        pivot_idx = start_idx + pivot_offset
        recency_weeks = (n - 1) - end_idx
        depth_component = max(0.0, 1.0 - (handle_depth_pct / _HANDLE_MAX_DEPTH_PCT))
        volume_component = max(0.0, 1.0 - max(0.0, volume_ratio - 1.0))
        score = (
            depth_component * 0.50
            + min(max(upper_half_margin_pct, 0.0), 8.0) / 8.0 * 0.25
            + volume_component * 0.15
            + max(0.0, 1.0 - (recency_weeks / 12.0)) * 0.10
        )

        valid_candidates.append(
            _HandleCandidate(
                structure=structure,
                start_idx=start_idx,
                end_idx=end_idx,
                duration_weeks=duration_weeks,
                handle_high=handle_high,
                handle_low=handle_low,
                pivot_idx=pivot_idx,
                handle_depth_pct=handle_depth_pct,
                upper_half_floor=upper_half_floor,
                upper_half_margin_pct=upper_half_margin_pct,
                volume_ratio=volume_ratio,
                score=score,
                recency_weeks=recency_weeks,
            )
        )

    if not valid_candidates:
        fallback = ("handle_missing_structure",) if not rejected else tuple(
            _stable_unique(rejected)
        )
        return None, fallback

    valid_candidates.sort(
        key=lambda candidate: (
            -candidate.score,
            candidate.handle_depth_pct,
            candidate.duration_weeks,
            candidate.recency_weeks,
            -candidate.pivot_idx,
        )
    )
    return valid_candidates[0], tuple(_stable_unique(rejected))


def _stable_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            unique.append(value)
    return unique
