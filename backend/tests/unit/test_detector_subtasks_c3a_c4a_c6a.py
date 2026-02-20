"""Regression tests for detector subtasks SE-C3/C4/C6 implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd

import app.analysis.patterns.cup_handle as cup_module
import app.analysis.patterns.high_tight_flag as htf_module
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


def test_high_tight_flag_validates_beyond_first_five_poles(monkeypatch):
    index = pd.bdate_range("2025-01-02", periods=220)
    close = np.linspace(40.0, 120.0, len(index))
    frame = _ohlcv_frame(index=index, close=close)
    detector_input = PatternDetectorInput(
        symbol="HTF",
        timeframe="daily",
        daily_bars=len(frame),
        weekly_bars=60,
        features={"daily_ohlcv": frame},
    )

    poles = [
        htf_module._PoleCandidateWindow(
            start_idx=40 + i,
            end_idx=80 + i,
            window_bars=20,
            pole_return=1.1,
            recency_bars=10 + i,
            weighted_score=2.0 - (i * 0.01),
        )
        for i in range(10)
    ]

    def _mock_find_poles(_frame):
        return poles

    call_index = {"value": 0}

    def _mock_find_flag(_frame, *, pole_candidate):
        call_index["value"] += 1
        if call_index["value"] < 7:
            return None, ("flag_depth_rejected",)
        return (
            htf_module._FlagCandidate(
                pole=pole_candidate,
                start_idx=120,
                end_idx=124,
                duration_bars=5,
                flag_high=112.0,
                flag_low=106.0,
                pivot_idx=121,
                flag_depth_pct=5.36,
                upper_half_floor=95.0,
                upper_half_margin_pct=3.0,
                volume_ratio=0.95,
                score=0.8,
                recency_bars=8,
            ),
            (),
        )

    monkeypatch.setattr(htf_module, "_find_pole_windows", _mock_find_poles)
    monkeypatch.setattr(
        htf_module, "_find_best_flag_candidate", _mock_find_flag
    )

    result = HighTightFlagDetector().detect_safe(
        detector_input, DEFAULT_SETUP_ENGINE_PARAMETERS
    )
    assert result.outcome == DetectorOutcome.DETECTED
    assert result.candidates[0]["metrics"]["pole_rank"] == 7


def test_high_tight_flag_rejects_invalid_volume_baseline():
    index = pd.bdate_range("2025-01-02", periods=220)
    close = np.linspace(40.0, 120.0, len(index))
    volume = np.zeros(len(index))
    frame = _ohlcv_frame(index=index, close=close, volume=volume)
    pole = htf_module._PoleCandidateWindow(
        start_idx=50,
        end_idx=90,
        window_bars=20,
        pole_return=1.2,
        recency_bars=5,
        weighted_score=2.0,
    )

    candidate, rejected = htf_module._find_best_flag_candidate(
        frame,
        pole_candidate=pole,
    )

    assert candidate is None
    assert "flag_volume_baseline_invalid" in rejected


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


def test_cup_with_handle_validates_beyond_first_five_cup_structures(monkeypatch):
    index = pd.date_range("2023-01-06", periods=110, freq="W-FRI")
    close = np.linspace(70.0, 120.0, len(index))
    frame = _ohlcv_frame(index=index, close=close)
    detector_input = PatternDetectorInput(
        symbol="CUP",
        timeframe="weekly",
        daily_bars=400,
        weekly_bars=len(frame),
        features={"weekly_ohlcv": frame},
    )

    structures = tuple(
        cup_module._CupStructure(
            left_idx=10 + i,
            low_idx=20 + i,
            right_idx=30 + i,
            duration_weeks=20,
            depth_pct=20.0,
            recovery_strength_pct=95.0,
            curvature_balance=0.7,
            recency_weeks=15 + i,
            structure_score=0.9 - (i * 0.01),
        )
        for i in range(10)
    )

    def _mock_find_cups(_frame):
        return cup_module._CupParseResult(candidates=structures, capped=False)

    call_index = {"value": 0}

    def _mock_find_handle(_frame, *, structure):
        call_index["value"] += 1
        if call_index["value"] < 7:
            return None, ("handle_depth_rejected",)
        return (
            cup_module._HandleCandidate(
                structure=structure,
                start_idx=55,
                end_idx=58,
                duration_weeks=4,
                handle_high=111.0,
                handle_low=106.0,
                pivot_idx=56,
                handle_depth_pct=4.5,
                upper_half_floor=95.0,
                upper_half_margin_pct=3.0,
                volume_ratio=0.92,
                score=0.8,
                recency_weeks=5,
            ),
            (),
        )

    monkeypatch.setattr(cup_module, "_find_cup_structures", _mock_find_cups)
    monkeypatch.setattr(
        cup_module, "_find_best_handle_candidate", _mock_find_handle
    )

    result = CupHandleDetector().detect_safe(
        detector_input, DEFAULT_SETUP_ENGINE_PARAMETERS
    )
    assert result.outcome == DetectorOutcome.DETECTED
    assert result.candidates[0]["metrics"]["cup_rank"] == 7


def test_cup_with_handle_rejects_invalid_volume_baseline():
    index = pd.date_range("2023-01-06", periods=90, freq="W-FRI")
    close = np.linspace(70.0, 95.0, len(index))
    volume = np.zeros(len(index))
    frame = _ohlcv_frame(index=index, close=close, volume=volume)
    structure = cup_module._CupStructure(
        left_idx=10,
        low_idx=25,
        right_idx=40,
        duration_weeks=30,
        depth_pct=20.0,
        recovery_strength_pct=95.0,
        curvature_balance=0.7,
        recency_weeks=8,
        structure_score=0.8,
    )

    candidate, rejected = cup_module._find_best_handle_candidate(
        frame,
        structure=structure,
    )

    assert candidate is None
    assert "handle_volume_baseline_invalid" in rejected


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
