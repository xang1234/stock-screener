"""Regression tests for detector subtasks SE-C3a, SE-C4a, and SE-C6a."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.analysis.patterns.config import DEFAULT_SETUP_ENGINE_PARAMETERS
from app.analysis.patterns.cup_handle import CupHandleDetector
from app.analysis.patterns.detectors import DetectorOutcome, PatternDetectorInput
from app.analysis.patterns.first_pullback import FirstPullbackDetector
from app.analysis.patterns.high_tight_flag import HighTightFlagDetector


def _ohlcv_frame(
    *,
    index: pd.DatetimeIndex,
    close: np.ndarray,
    low: np.ndarray | None = None,
    extra_cols: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    close_series = pd.Series(close.astype(float), index=index)
    low_values = (
        low.astype(float) if low is not None else (close_series.to_numpy() * 0.99)
    )
    frame = pd.DataFrame(
        {
            "Open": close_series.to_numpy() * 0.995,
            "High": close_series.to_numpy() * 1.01,
            "Low": low_values,
            "Close": close_series.to_numpy(),
            "Volume": np.full(len(close_series), 1_000_000.0),
        },
        index=index,
    )
    if extra_cols:
        for key, values in extra_cols.items():
            frame[key] = values.astype(float)
    return frame


def test_high_tight_flag_pole_detection_returns_ranked_candidates():
    index = pd.bdate_range("2025-01-02", periods=220)
    close = np.concatenate(
        [
            np.linspace(40.0, 50.0, 150, endpoint=False),
            np.linspace(50.0, 106.0, 30, endpoint=False),
            np.linspace(106.0, 114.0, 40),
        ]
    )
    frame = _ohlcv_frame(index=index, close=close)
    detector_input = PatternDetectorInput(
        symbol="HTF",
        timeframe="daily",
        daily_bars=len(frame),
        weekly_bars=60,
        features={"daily_ohlcv": frame},
    )

    detector = HighTightFlagDetector()
    first = detector.detect_safe(detector_input, DEFAULT_SETUP_ENGINE_PARAMETERS)
    second = detector.detect_safe(detector_input, DEFAULT_SETUP_ENGINE_PARAMETERS)

    assert first.outcome == DetectorOutcome.DETECTED
    assert first.candidates == second.candidates
    best = first.candidates[0]
    assert best["pattern"] == "high_tight_flag"
    assert best["metrics"]["pole_return_pct"] >= 100.0
    assert 20 <= best["metrics"]["pole_window_bars"] <= 40
    assert "pole_start_date" in best["metrics"]
    assert "pole_end_date" in best["metrics"]


def test_high_tight_flag_returns_no_detection_when_no_100pct_run():
    index = pd.bdate_range("2025-01-02", periods=220)
    close = np.linspace(90.0, 125.0, len(index))
    frame = _ohlcv_frame(index=index, close=close)
    detector_input = PatternDetectorInput(
        symbol="HTF",
        timeframe="daily",
        daily_bars=len(frame),
        weekly_bars=60,
        features={"daily_ohlcv": frame},
    )

    result = HighTightFlagDetector().detect_safe(
        detector_input, DEFAULT_SETUP_ENGINE_PARAMETERS
    )
    assert result.outcome == DetectorOutcome.NOT_DETECTED
    assert "pole_return_below_threshold" in result.failed_checks


def test_cup_structure_parsing_emits_depth_duration_and_recovery_metrics():
    index = pd.date_range("2023-01-06", periods=90, freq="W-FRI")
    close = np.concatenate(
        [
            np.linspace(88.0, 96.0, 12, endpoint=False),
            np.array([100.0, 102.0, 99.0]),
            np.linspace(98.0, 70.0, 20, endpoint=False),
            np.linspace(70.0, 96.0, 24, endpoint=False),
            np.array([99.0, 97.0, 96.0]),
            np.linspace(95.0, 94.0, 28),
        ]
    )
    frame = _ohlcv_frame(index=index, close=close)
    detector_input = PatternDetectorInput(
        symbol="CUP",
        timeframe="weekly",
        daily_bars=400,
        weekly_bars=len(frame),
        features={"weekly_ohlcv": frame},
    )

    result = CupHandleDetector().detect_safe(
        detector_input, DEFAULT_SETUP_ENGINE_PARAMETERS
    )
    assert result.outcome == DetectorOutcome.DETECTED
    best = result.candidates[0]
    assert best["pattern"] == "cup_with_handle"
    assert 6 <= best["metrics"]["cup_duration_weeks"] <= 65
    assert 8.0 <= best["metrics"]["cup_depth_pct"] <= 50.0
    assert best["metrics"]["recovery_strength_pct"] >= 90.0
    assert "left_lip_date" in best["metrics"]
    assert "cup_low_date" in best["metrics"]
    assert "right_lip_date" in best["metrics"]


def _first_pullback_frame(*, touch_positions: tuple[int, ...]) -> pd.DataFrame:
    length = 130
    index = pd.bdate_range("2025-01-02", periods=length)
    close = np.linspace(110.0, 128.0, length)
    low = np.maximum(close - 6.0, 102.5)
    for pos in touch_positions:
        low[pos] = 99.5
    ma_21 = np.full(length, 100.0)
    return _ohlcv_frame(
        index=index,
        close=close,
        low=low,
        extra_cols={"ma_21": ma_21},
    )


def test_first_pullback_counts_distinct_tests_and_marks_second_test():
    frame = _first_pullback_frame(touch_positions=(80, 90))
    detector_input = PatternDetectorInput(
        symbol="PB",
        timeframe="daily",
        daily_bars=len(frame),
        weekly_bars=60,
        features={"daily_ohlcv": frame},
    )

    result = FirstPullbackDetector().detect_safe(
        detector_input, DEFAULT_SETUP_ENGINE_PARAMETERS
    )
    assert result.outcome == DetectorOutcome.DETECTED
    best = result.candidates[0]
    assert best["pattern"] == "first_pullback"
    assert best["metrics"]["tests_count"] == 2
    assert best["metrics"]["is_first_test"] is False
    assert best["metrics"]["is_second_test"] is True
    assert "pullback_orderliness_score" in best["metrics"]


def test_first_pullback_does_not_double_count_clustered_touch_bars():
    frame = _first_pullback_frame(touch_positions=(80, 81, 82))
    detector_input = PatternDetectorInput(
        symbol="PB",
        timeframe="daily",
        daily_bars=len(frame),
        weekly_bars=60,
        features={"daily_ohlcv": frame},
    )

    result = FirstPullbackDetector().detect_safe(
        detector_input, DEFAULT_SETUP_ENGINE_PARAMETERS
    )
    assert result.outcome == DetectorOutcome.DETECTED
    best = result.candidates[0]
    assert best["metrics"]["tests_count"] == 1
    assert best["metrics"]["is_first_test"] is True
