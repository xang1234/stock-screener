"""High-Tight-Flag detector entrypoint.

Expected input orientation:
- Chronological daily bars (oldest -> newest).
- Pole and flag phases use strictly historical windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.analysis.patterns.config import SetupEngineParameters
from app.analysis.patterns.detectors.base import (
    PatternDetector,
    PatternDetectorInput,
    PatternDetectorResult,
)
from app.analysis.patterns.models import PatternCandidateModel
from app.analysis.patterns.normalization import normalize_detector_input_ohlcv

_POLE_MIN_BARS = 20
_POLE_MAX_BARS = 40
_MIN_POLE_RETURN = 1.0
_RECENT_POLE_BARS = 20
_MAX_POLE_CANDIDATES = 5
_FLAG_MIN_BARS = 3
_FLAG_MAX_BARS = 25
_FLAG_MAX_DEPTH_PCT = 25.0
_FLAG_MAX_VOLUME_RATIO = 1.15


@dataclass(frozen=True)
class _PoleCandidateWindow:
    start_idx: int
    end_idx: int
    window_bars: int
    pole_return: float
    recency_bars: int
    weighted_score: float


@dataclass(frozen=True)
class _FlagCandidate:
    pole: _PoleCandidateWindow
    start_idx: int
    end_idx: int
    duration_bars: int
    flag_high: float
    flag_low: float
    pivot_idx: int
    flag_depth_pct: float
    upper_half_floor: float
    upper_half_margin_pct: float
    volume_ratio: float
    score: float
    recency_bars: int


class HighTightFlagDetector(PatternDetector):
    """Compile-safe entrypoint for HTF detection."""

    name = "high_tight_flag"

    def detect(
        self,
        detector_input: PatternDetectorInput,
        parameters: SetupEngineParameters,
    ) -> PatternDetectorResult:
        del parameters
        normalized = normalize_detector_input_ohlcv(
            features=detector_input.features,
            timeframe="daily",
            min_bars=180,
            feature_key="daily_ohlcv",
            fallback_bar_count=detector_input.daily_bars,
        )
        if not normalized.prerequisites_ok:
            return PatternDetectorResult.insufficient_data(
                self.name, normalized=normalized
            )

        if normalized.frame is None:
            return PatternDetectorResult.insufficient_data(
                self.name,
                failed_checks=("missing_daily_ohlcv_for_pole_detection",),
                warnings=normalized.warnings,
            )

        pole_windows = _find_pole_windows(normalized.frame)
        if not pole_windows:
            return PatternDetectorResult.no_detection(
                self.name,
                failed_checks=("pole_return_below_threshold",),
                warnings=normalized.warnings,
            )

        candidates: list[PatternCandidateModel] = []
        failed_reasons: list[str] = []
        close = normalized.frame["Close"]
        high = normalized.frame["High"]
        low = normalized.frame["Low"]
        volume = normalized.frame["Volume"]

        for pole_rank, window in enumerate(pole_windows, start=1):
            flag_candidate, rejected = _find_best_flag_candidate(
                normalized.frame,
                pole_candidate=window,
            )
            failed_reasons.extend(rejected)
            if flag_candidate is None:
                continue

            start_date = close.index[window.start_idx].date().isoformat()
            end_date = close.index[window.end_idx].date().isoformat()
            flag_start_date = close.index[flag_candidate.start_idx].date().isoformat()
            flag_end_date = close.index[flag_candidate.end_idx].date().isoformat()
            start_close = float(close.iat[window.start_idx])
            end_close = float(close.iat[window.end_idx])
            pole_return_pct = window.pole_return * 100.0
            pole_interval_high = float(high.iloc[window.start_idx : window.end_idx + 1].max())
            pole_interval_low = float(low.iloc[window.start_idx : window.end_idx + 1].min())
            pole_volume_mean = float(
                volume.iloc[window.start_idx : window.end_idx + 1].mean()
            )
            flag_volume_mean = float(
                volume.iloc[
                    flag_candidate.start_idx : flag_candidate.end_idx + 1
                ].mean()
            )

            recency_weight = _recency_weight(flag_candidate.recency_bars)
            depth_component = max(
                0.0,
                1.0 - (flag_candidate.flag_depth_pct / _FLAG_MAX_DEPTH_PCT),
            )
            volume_component = max(
                0.0,
                1.0 - max(0.0, flag_candidate.volume_ratio - 1.0),
            )
            confidence = min(
                0.95,
                max(
                    0.05,
                    0.30
                    + (_confidence_from_window(window) * 0.45)
                    + (depth_component * 0.15)
                    + (volume_component * 0.10),
                ),
            )
            quality_score = min(
                100.0,
                max(
                    0.0,
                    40.0
                    + (depth_component * 30.0)
                    + min(flag_candidate.upper_half_margin_pct, 10.0)
                    + (volume_component * 10.0),
                ),
            )
            readiness_score = max(
                0.0,
                min(
                    100.0,
                    75.0
                    + (recency_weight * 30.0)
                    - (flag_candidate.duration_bars * 1.5),
                ),
            )

            candidates.append(
                PatternCandidateModel(
                    pattern=self.name,
                    timeframe="daily",
                    source_detector=self.name,
                    pivot_price=flag_candidate.flag_high,
                    pivot_type="flag_high",
                    pivot_date=close.index[flag_candidate.pivot_idx]
                    .date()
                    .isoformat(),
                    quality_score=quality_score,
                    readiness_score=readiness_score,
                    confidence=confidence,
                    metrics={
                        "pole_rank": pole_rank,
                        "pole_start_date": start_date,
                        "pole_end_date": end_date,
                        "pole_window_bars": window.window_bars,
                        "pole_return_pct": round(pole_return_pct, 4),
                        "pole_start_close": round(start_close, 4),
                        "pole_end_close": round(end_close, 4),
                        "pole_interval_high": round(pole_interval_high, 4),
                        "pole_interval_low": round(pole_interval_low, 4),
                        "pole_volume_mean": round(pole_volume_mean, 4),
                        "pole_recency_bars": window.recency_bars,
                        "flag_start_date": flag_start_date,
                        "flag_end_date": flag_end_date,
                        "flag_duration_bars": flag_candidate.duration_bars,
                        "flag_high": round(flag_candidate.flag_high, 4),
                        "flag_low": round(flag_candidate.flag_low, 4),
                        "flag_depth_pct": round(flag_candidate.flag_depth_pct, 4),
                        "flag_upper_half_floor": round(
                            flag_candidate.upper_half_floor, 4
                        ),
                        "flag_upper_half_margin_pct": round(
                            flag_candidate.upper_half_margin_pct, 4
                        ),
                        "flag_volume_mean": round(flag_volume_mean, 4),
                        "flag_volume_ratio_vs_pole": round(
                            flag_candidate.volume_ratio, 6
                        ),
                        "flag_recency_bars": flag_candidate.recency_bars,
                        "recency_weight": round(recency_weight, 6),
                        "is_recent_flag": flag_candidate.recency_bars
                        <= _RECENT_POLE_BARS,
                    },
                    checks={
                        "pole_window_in_range": bool(
                            _POLE_MIN_BARS
                            <= window.window_bars
                            <= _POLE_MAX_BARS
                        ),
                        "pole_return_threshold_met": bool(
                            window.pole_return >= _MIN_POLE_RETURN
                        ),
                        "flag_duration_in_range": bool(
                            _FLAG_MIN_BARS
                            <= flag_candidate.duration_bars
                            <= _FLAG_MAX_BARS
                        ),
                        "flag_depth_in_range": bool(
                            flag_candidate.flag_depth_pct <= _FLAG_MAX_DEPTH_PCT
                        ),
                        "flag_in_upper_half": bool(
                            flag_candidate.flag_low
                            >= flag_candidate.upper_half_floor
                        ),
                        "flag_volume_contracting": bool(
                            flag_candidate.volume_ratio <= _FLAG_MAX_VOLUME_RATIO
                        ),
                    },
                    notes=(
                        "pole_and_flag_validated",
                        "pivot_from_flag_high",
                    ),
                )
            )

        if not candidates:
            return PatternDetectorResult.no_detection(
                self.name,
                failed_checks=tuple(_stable_unique(failed_reasons)),
                warnings=normalized.warnings,
            )

        output_candidates = tuple(candidates[:_MAX_POLE_CANDIDATES])
        return PatternDetectorResult.detected(
            self.name,
            output_candidates,
            passed_checks=("pole_candidates_found", "flag_candidates_validated"),
            warnings=normalized.warnings,
        )


def _find_pole_windows(frame: pd.DataFrame) -> list[_PoleCandidateWindow]:
    close = frame["Close"]
    n = len(close)
    if n < _POLE_MIN_BARS:
        return []

    candidates: list[_PoleCandidateWindow] = []
    last_idx = n - 1
    closes = close.to_numpy(dtype=float)

    for window_bars in range(_POLE_MIN_BARS, _POLE_MAX_BARS + 1):
        end_indices = np.arange(window_bars - 1, n)
        start_indices = end_indices - window_bars + 1
        start_closes = closes[start_indices]
        end_closes = closes[end_indices]

        valid_start = start_closes > 0.0
        if not np.any(valid_start):
            continue

        pole_returns = np.full_like(end_closes, np.nan, dtype=float)
        np.divide(
            end_closes,
            start_closes,
            out=pole_returns,
            where=valid_start,
        )
        pole_returns -= 1.0
        valid = valid_start & (pole_returns >= _MIN_POLE_RETURN)
        if not np.any(valid):
            continue

        valid_start_indices = start_indices[valid]
        valid_end_indices = end_indices[valid]
        valid_returns = pole_returns[valid]
        recencies = last_idx - valid_end_indices
        recency_weights = np.array(
            [_recency_weight(int(value)) for value in recencies],
            dtype=float,
        )
        weighted_scores = valid_returns * (1.0 + recency_weights)

        for start_idx, end_idx, pole_return, recency_bars, weighted_score in zip(
            valid_start_indices,
            valid_end_indices,
            valid_returns,
            recencies,
            weighted_scores,
        ):
            candidates.append(
                _PoleCandidateWindow(
                    start_idx=int(start_idx),
                    end_idx=int(end_idx),
                    window_bars=window_bars,
                    pole_return=float(pole_return),
                    recency_bars=int(recency_bars),
                    weighted_score=float(weighted_score),
                )
            )

    candidates.sort(
        key=lambda candidate: (
            -candidate.weighted_score,
            candidate.recency_bars,
            candidate.window_bars,
            -candidate.end_idx,
            -candidate.start_idx,
        )
    )
    return candidates


def _recency_weight(recency_bars: int) -> float:
    if recency_bars <= 0:
        return 0.35
    if recency_bars >= _RECENT_POLE_BARS:
        return 0.0
    return 0.35 * (_RECENT_POLE_BARS - recency_bars) / _RECENT_POLE_BARS


def _confidence_from_window(window: _PoleCandidateWindow) -> float:
    return min(
        0.95,
        max(
            0.05,
            0.45
            + min(window.pole_return, 2.5) * 0.15
            + _recency_weight(window.recency_bars) * 0.6,
        ),
    )


def _find_best_flag_candidate(
    frame: pd.DataFrame,
    *,
    pole_candidate: _PoleCandidateWindow,
) -> tuple[_FlagCandidate | None, tuple[str, ...]]:
    start_idx = pole_candidate.end_idx + 1
    if start_idx >= len(frame):
        return None, ("flag_missing_structure",)

    highs = frame["High"].to_numpy(dtype=float)
    lows = frame["Low"].to_numpy(dtype=float)
    volumes = frame["Volume"].to_numpy(dtype=float)

    pole_slice = slice(pole_candidate.start_idx, pole_candidate.end_idx + 1)
    pole_high = float(np.nanmax(highs[pole_slice]))
    pole_low = float(np.nanmin(lows[pole_slice]))
    pole_range = max(pole_high - pole_low, 1e-9)
    upper_half_floor = pole_low + (pole_range * 0.5)
    pole_volume_mean = float(np.nanmean(volumes[pole_slice]))
    if pole_volume_mean <= 0.0 or pd.isna(pole_volume_mean):
        return None, ("flag_volume_baseline_invalid",)

    n = len(frame)
    max_duration = min(_FLAG_MAX_BARS, n - start_idx)
    if max_duration < _FLAG_MIN_BARS:
        return None, ("flag_missing_structure",)

    duration_bars = np.arange(_FLAG_MIN_BARS, max_duration + 1)
    duration_offsets = duration_bars - 1
    high_segment = highs[start_idx : start_idx + max_duration]
    low_segment = lows[start_idx : start_idx + max_duration]
    volume_segment = volumes[start_idx : start_idx + max_duration]

    cumulative_high = np.fmax.accumulate(high_segment)
    cumulative_low = np.fmin.accumulate(low_segment)
    volume_valid = ~np.isnan(volume_segment)
    cumulative_volume = np.cumsum(np.where(volume_valid, volume_segment, 0.0))
    cumulative_volume_count = np.cumsum(volume_valid)

    flag_highs = cumulative_high[duration_offsets]
    flag_lows = cumulative_low[duration_offsets]
    volume_counts = cumulative_volume_count[duration_offsets]
    flag_volume_means = np.full_like(flag_highs, np.nan, dtype=float)
    np.divide(
        cumulative_volume[duration_offsets],
        volume_counts,
        out=flag_volume_means,
        where=volume_counts > 0,
    )

    structure_ok = flag_highs > 0.0
    flag_depth_pct = ((flag_highs - flag_lows) / flag_highs) * 100.0
    upper_half_margin_pct = ((flag_lows - upper_half_floor) / pole_range) * 100.0
    volume_mean_ok = (flag_volume_means > 0.0) & ~np.isnan(flag_volume_means)
    volume_ratios = np.full_like(flag_volume_means, np.nan, dtype=float)
    np.divide(
        flag_volume_means,
        pole_volume_mean,
        out=volume_ratios,
        where=volume_mean_ok,
    )

    depth_ok = flag_depth_pct <= _FLAG_MAX_DEPTH_PCT
    upper_half_ok = flag_lows >= upper_half_floor
    volume_ok = volume_ratios <= _FLAG_MAX_VOLUME_RATIO
    valid = structure_ok & volume_mean_ok & depth_ok & upper_half_ok & volume_ok

    rejected: list[str] = []
    if np.any(~structure_ok):
        rejected.append("flag_missing_structure")
    if np.any(structure_ok & ~depth_ok):
        rejected.append("flag_depth_rejected")
    if np.any(structure_ok & ~upper_half_ok):
        rejected.append("flag_upper_half_rejected")
    if np.any(structure_ok & (~volume_mean_ok | ~volume_ok)):
        rejected.append("flag_volume_rejected")

    valid_positions = np.flatnonzero(valid)
    if len(valid_positions) == 0:
        fallback = ("flag_missing_structure",) if not rejected else tuple(
            _stable_unique(rejected)
        )
        return None, fallback

    depth_components = np.maximum(
        0.0,
        1.0 - (flag_depth_pct / _FLAG_MAX_DEPTH_PCT),
    )
    volume_components = np.maximum(
        0.0,
        1.0 - np.maximum(0.0, volume_ratios - 1.0),
    )
    recency_bars_values = (n - 1) - (start_idx + duration_bars - 1)
    scores = (
        depth_components * 0.50
        + np.minimum(np.maximum(upper_half_margin_pct, 0.0), 12.0) / 12.0 * 0.25
        + volume_components * 0.15
        + np.array(
            [_recency_weight(int(value)) for value in recency_bars_values],
            dtype=float,
        )
        * 0.10
    )

    def _candidate_sort_key(position: int) -> tuple[float, float, int, int, int]:
        pivot_offset = int(np.nanargmax(high_segment[: duration_bars[position]]))
        return (
            -float(scores[position]),
            float(flag_depth_pct[position]),
            int(duration_bars[position]),
            int(recency_bars_values[position]),
            -(start_idx + pivot_offset),
        )

    best_position = min(
        (int(position) for position in valid_positions),
        key=_candidate_sort_key,
    )
    best_duration = int(duration_bars[best_position])
    pivot_idx = start_idx + int(np.nanargmax(high_segment[:best_duration]))
    end_idx = start_idx + best_duration - 1

    return (
        _FlagCandidate(
            pole=pole_candidate,
            start_idx=start_idx,
            end_idx=end_idx,
            duration_bars=best_duration,
            flag_high=float(flag_highs[best_position]),
            flag_low=float(flag_lows[best_position]),
            pivot_idx=pivot_idx,
            flag_depth_pct=float(flag_depth_pct[best_position]),
            upper_half_floor=upper_half_floor,
            upper_half_margin_pct=float(upper_half_margin_pct[best_position]),
            volume_ratio=float(volume_ratios[best_position]),
            score=float(scores[best_position]),
            recency_bars=int(recency_bars_values[best_position]),
        ),
        tuple(_stable_unique(rejected)),
    )


def _stable_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            unique.append(value)
    return unique
