"""Regression tests for detector subtasks SE-C3/C4/C6 implementations."""

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
    volume: np.ndarray | None = None,
) -> pd.DataFrame:
    close_series = pd.Series(close.astype(float), index=index)
    low_values = (
        low.astype(float) if low is not None else (close_series.to_numpy() * 0.99)
    )
    volume_values = (
        volume.astype(float)
        if volume is not None
        else np.full(len(close_series), 1_000_000.0)
    )
    frame = pd.DataFrame(
        {
            "Open": close_series.to_numpy() * 0.995,
            "High": close_series.to_numpy() * 1.01,
            "Low": low_values,
            "Close": close_series.to_numpy(),
            "Volume": volume_values,
        },
        index=index,
    )
    if extra_cols:
        for key, values in extra_cols.items():
            frame[key] = values.astype(float)
    return frame


def test_high_tight_flag_returns_flag_validated_candidate():
    index = pd.bdate_range("2025-01-02", periods=220)
    close = np.concatenate(
        [
            np.linspace(40.0, 50.0, 150, endpoint=False),
            np.linspace(50.0, 120.0, 30, endpoint=False),
            np.linspace(118.0, 112.0, 12, endpoint=False),
            np.linspace(112.0, 116.0, 28),
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

    result = HighTightFlagDetector().detect_safe(
        detector_input, DEFAULT_SETUP_ENGINE_PARAMETERS
    )
    assert result.outcome == DetectorOutcome.DETECTED
    best = result.candidates[0]
    assert best["pattern"] == "high_tight_flag"
    assert best["pivot_type"] == "flag_high"
    assert best["metrics"]["flag_depth_pct"] <= 25.0
    assert best["checks"]["flag_in_upper_half"] is True
    assert "flag_start_date" in best["metrics"]
    assert "flag_end_date" in best["metrics"]


def test_high_tight_flag_reports_rejection_reason_when_flag_missing():
    index = pd.bdate_range("2025-01-02", periods=220)
    close = np.concatenate(
        [
            np.full(200, 50.0),
            np.linspace(50.0, 101.0, 20),
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

    result = HighTightFlagDetector().detect_safe(
        detector_input, DEFAULT_SETUP_ENGINE_PARAMETERS
    )
    assert result.outcome == DetectorOutcome.NOT_DETECTED
    assert "flag_missing_structure" in result.failed_checks


def test_cup_with_handle_returns_handle_validated_candidate():
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
    assert best["pivot_type"] == "handle_high"
    assert 1 <= best["metrics"]["handle_duration_weeks"] <= 5
    assert best["metrics"]["handle_depth_pct"] <= 15.0
    assert best["checks"]["handle_in_upper_half"] is True
    assert best["metrics"]["handle_volume_ratio_vs_right_side"] <= 1.1


def test_cup_with_handle_rejects_when_handle_too_deep():
    index = pd.date_range("2023-01-06", periods=90, freq="W-FRI")
    close = np.concatenate(
        [
            np.linspace(88.0, 96.0, 12, endpoint=False),
            np.array([100.0, 102.0, 99.0]),
            np.linspace(98.0, 70.0, 20, endpoint=False),
            np.linspace(70.0, 96.0, 24, endpoint=False),
            np.array([99.0, 80.0, 78.0, 76.0, 75.0]),
            np.linspace(75.0, 74.0, 26),
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
    assert result.outcome == DetectorOutcome.NOT_DETECTED
    assert any(
        check in result.failed_checks
        for check in ("handle_depth_rejected", "handle_upper_half_rejected")
    )


def _first_pullback_frame(
    *,
    touch_positions: tuple[int, ...],
    no_resumption_after_last_touch: bool = False,
) -> pd.DataFrame:
    length = 130
    index = pd.bdate_range("2025-01-02", periods=length)
    close = np.linspace(110.0, 128.0, length)
    if no_resumption_after_last_touch and touch_positions:
        last_touch = touch_positions[-1]
        if last_touch + 1 < length:
            close[last_touch + 1 :] = np.linspace(
                close[last_touch] - 0.5,
                close[last_touch] - 8.0,
                length - (last_touch + 1),
            )
    low = np.maximum(close - 6.0, 96.0)
    for pos in touch_positions:
        low[pos] = 99.5
    ma_21 = np.full(length, 100.0)
    return _ohlcv_frame(
        index=index,
        close=close,
        low=low,
        extra_cols={"ma_21": ma_21},
    )


def test_first_pullback_marks_resumption_pivot_mode_when_trigger_exists():
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
    assert best["pivot_type"] == "resumption_high"
    assert best["metrics"]["pivot_mode_chosen"] == "resumption_high"
    assert best["metrics"]["pivot_mode_alternate"] == "pullback_high"
    assert best["checks"]["resumption_trigger_confirmed"] is True
    assert best["metrics"]["tests_count"] == 2


def test_first_pullback_falls_back_to_pullback_high_when_no_trigger():
    frame = _first_pullback_frame(
        touch_positions=(80,),
        no_resumption_after_last_touch=True,
    )
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
    assert best["pivot_type"] == "pullback_high"
    assert best["metrics"]["pivot_mode_chosen"] == "pullback_high"
    assert best["checks"]["resumption_trigger_confirmed"] is False
    assert best["metrics"]["resumption_high_price"] is None


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
