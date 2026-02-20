"""High-Tight-Flag detector entrypoint.

Expected input orientation:
- Chronological daily bars (oldest -> newest).
- Pole and flag phases use strictly historical windows.

TODO(SE-C3b): Implement flag validation and pivot extraction.
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


@dataclass(frozen=True)
class _PoleCandidateWindow:
    start_idx: int
    end_idx: int
    window_bars: int
    pole_return: float
    recency_bars: int
    weighted_score: float


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
        close = normalized.frame["Close"]
        high = normalized.frame["High"]
        low = normalized.frame["Low"]

        for rank, window in enumerate(
            pole_windows[:_MAX_POLE_CANDIDATES], start=1
        ):
            start_date = close.index[window.start_idx].date().isoformat()
            end_date = close.index[window.end_idx].date().isoformat()
            start_close = float(close.iat[window.start_idx])
            end_close = float(close.iat[window.end_idx])
            pole_return_pct = window.pole_return * 100.0

            interval_high = float(
                high.iloc[window.start_idx : window.end_idx + 1].max()
            )
            interval_low = float(
                low.iloc[window.start_idx : window.end_idx + 1].min()
            )
            recency_weight = _recency_weight(window.recency_bars)
            confidence = _confidence_from_window(window)
            quality_score = min(100.0, 35.0 + 0.45 * pole_return_pct)
            readiness_score = max(0.0, 100.0 - (window.recency_bars * 3.0))

            candidates.append(
                PatternCandidateModel(
                    pattern=self.name,
                    timeframe="daily",
                    source_detector=self.name,
                    pivot_price=end_close,
                    pivot_type="pole_high",
                    pivot_date=end_date,
                    quality_score=quality_score,
                    readiness_score=readiness_score,
                    confidence=confidence,
                    metrics={
                        "pole_rank": rank,
                        "pole_start_date": start_date,
                        "pole_end_date": end_date,
                        "pole_window_bars": window.window_bars,
                        "pole_return_pct": round(pole_return_pct, 4),
                        "pole_start_close": round(start_close, 4),
                        "pole_end_close": round(end_close, 4),
                        "pole_interval_high": round(interval_high, 4),
                        "pole_interval_low": round(interval_low, 4),
                        "pole_recency_bars": window.recency_bars,
                        "recency_weight": round(recency_weight, 6),
                        "is_recent_pole": window.recency_bars <= _RECENT_POLE_BARS,
                    },
                    checks={
                        "pole_window_in_range": (
                            _POLE_MIN_BARS
                            <= window.window_bars
                            <= _POLE_MAX_BARS
                        ),
                        "pole_return_threshold_met": window.pole_return
                        >= _MIN_POLE_RETURN,
                        "pole_recent": window.recency_bars <= _RECENT_POLE_BARS,
                    },
                    notes=(
                        "pole_candidate_only",
                        "flag_validation_pending",
                    ),
                )
            )

        return PatternDetectorResult.detected(
            self.name,
            tuple(candidates),
            passed_checks=("pole_candidates_found",),
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
