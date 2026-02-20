"""High-Tight-Flag detector entrypoint.

Expected input orientation:
- Chronological daily bars (oldest -> newest).
- Pole and flag phases use strictly historical windows.
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

        for pole_rank, window in enumerate(
            pole_windows[:_MAX_POLE_CANDIDATES], start=1
        ):
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
                        "pole_window_in_range": (
                            _POLE_MIN_BARS
                            <= window.window_bars
                            <= _POLE_MAX_BARS
                        ),
                        "pole_return_threshold_met": window.pole_return
                        >= _MIN_POLE_RETURN,
                        "flag_duration_in_range": (
                            _FLAG_MIN_BARS
                            <= flag_candidate.duration_bars
                            <= _FLAG_MAX_BARS
                        ),
                        "flag_depth_in_range": flag_candidate.flag_depth_pct
                        <= _FLAG_MAX_DEPTH_PCT,
                        "flag_in_upper_half": flag_candidate.flag_low
                        >= flag_candidate.upper_half_floor,
                        "flag_volume_contracting": flag_candidate.volume_ratio
                        <= _FLAG_MAX_VOLUME_RATIO,
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

        return PatternDetectorResult.detected(
            self.name,
            tuple(candidates),
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

    for end_idx in range(_POLE_MIN_BARS - 1, n):
        for window_bars in range(_POLE_MIN_BARS, _POLE_MAX_BARS + 1):
            start_idx = end_idx - window_bars + 1
            if start_idx < 0:
                continue

            start_close = closes[start_idx]
            end_close = closes[end_idx]
            if start_close <= 0.0:
                continue

            pole_return = (end_close / start_close) - 1.0
            if pole_return < _MIN_POLE_RETURN:
                continue

            recency_bars = last_idx - end_idx
            weighted_score = pole_return * (1.0 + _recency_weight(recency_bars))
            candidates.append(
                _PoleCandidateWindow(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    window_bars=window_bars,
                    pole_return=pole_return,
                    recency_bars=recency_bars,
                    weighted_score=weighted_score,
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

    highs = frame["High"]
    lows = frame["Low"]
    volumes = frame["Volume"]

    pole_high = float(highs.iloc[pole_candidate.start_idx : pole_candidate.end_idx + 1].max())
    pole_low = float(lows.iloc[pole_candidate.start_idx : pole_candidate.end_idx + 1].min())
    pole_range = max(pole_high - pole_low, 1e-9)
    upper_half_floor = pole_low + (pole_range * 0.5)
    pole_volume_mean = float(
        volumes.iloc[pole_candidate.start_idx : pole_candidate.end_idx + 1].mean()
    )
    if pole_volume_mean <= 0.0 or pd.isna(pole_volume_mean):
        pole_volume_mean = 1.0

    rejected: list[str] = []
    valid_candidates: list[_FlagCandidate] = []
    n = len(frame)

    for duration_bars in range(_FLAG_MIN_BARS, _FLAG_MAX_BARS + 1):
        end_idx = start_idx + duration_bars - 1
        if end_idx >= n:
            break

        segment = frame.iloc[start_idx : end_idx + 1]
        segment_high = segment["High"].to_numpy(dtype=float)
        segment_low = segment["Low"].to_numpy(dtype=float)
        segment_volume = segment["Volume"].to_numpy(dtype=float)

        flag_high = float(segment_high.max())
        flag_low = float(segment_low.min())
        if flag_high <= 0.0:
            rejected.append("flag_missing_structure")
            continue

        flag_depth_pct = ((flag_high - flag_low) / flag_high) * 100.0
        upper_half_margin_pct = ((flag_low - upper_half_floor) / pole_range) * 100.0
        flag_volume_mean = float(segment_volume.mean())
        volume_ratio = flag_volume_mean / pole_volume_mean

        depth_ok = flag_depth_pct <= _FLAG_MAX_DEPTH_PCT
        upper_half_ok = flag_low >= upper_half_floor
        volume_ok = volume_ratio <= _FLAG_MAX_VOLUME_RATIO

        if not depth_ok:
            rejected.append("flag_depth_rejected")
        if not upper_half_ok:
            rejected.append("flag_upper_half_rejected")
        if not volume_ok:
            rejected.append("flag_volume_rejected")
        if not (depth_ok and upper_half_ok and volume_ok):
            continue

        pivot_offset = int(segment_high.argmax())
        pivot_idx = start_idx + pivot_offset
        recency_bars = (n - 1) - end_idx
        depth_component = max(0.0, 1.0 - (flag_depth_pct / _FLAG_MAX_DEPTH_PCT))
        volume_component = max(0.0, 1.0 - max(0.0, volume_ratio - 1.0))
        score = (
            depth_component * 0.50
            + min(max(upper_half_margin_pct, 0.0), 12.0) / 12.0 * 0.25
            + volume_component * 0.15
            + _recency_weight(recency_bars) * 0.10
        )

        valid_candidates.append(
            _FlagCandidate(
                pole=pole_candidate,
                start_idx=start_idx,
                end_idx=end_idx,
                duration_bars=duration_bars,
                flag_high=flag_high,
                flag_low=flag_low,
                pivot_idx=pivot_idx,
                flag_depth_pct=flag_depth_pct,
                upper_half_floor=upper_half_floor,
                upper_half_margin_pct=upper_half_margin_pct,
                volume_ratio=volume_ratio,
                score=score,
                recency_bars=recency_bars,
            )
        )

    if not valid_candidates:
        fallback = ("flag_missing_structure",) if not rejected else tuple(
            _stable_unique(rejected)
        )
        return None, fallback

    valid_candidates.sort(
        key=lambda candidate: (
            -candidate.score,
            candidate.flag_depth_pct,
            candidate.duration_bars,
            candidate.recency_bars,
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
