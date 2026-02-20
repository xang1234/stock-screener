"""Regression tests for detector subtasks SE-C3/C4/C6 implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd

import app.analysis.patterns.cup_handle as cup_module
import app.analysis.patterns.high_tight_flag as htf_module
import app.analysis.patterns.three_weeks_tight as twt_module
import app.analysis.patterns.vcp_wrapper as vcp_module
from app.analysis.patterns.config import DEFAULT_SETUP_ENGINE_PARAMETERS
from app.analysis.patterns.cup_handle import CupHandleDetector
from app.analysis.patterns.detectors import DetectorOutcome, PatternDetectorInput
from app.analysis.patterns.first_pullback import FirstPullbackDetector
from app.analysis.patterns.high_tight_flag import HighTightFlagDetector
from app.analysis.patterns.three_weeks_tight import ThreeWeeksTightDetector
from app.analysis.patterns.vcp_wrapper import VCPWrapperDetector


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


def test_vcp_wrapper_maps_legacy_output_and_reverses_orientation(monkeypatch):
    index = pd.bdate_range("2025-01-02", periods=220)
    close = np.linspace(80.0, 130.0, len(index))
    frame = _ohlcv_frame(index=index, close=close)
    detector_input = PatternDetectorInput(
        symbol="VCP",
        timeframe="daily",
        daily_bars=len(frame),
        weekly_bars=60,
        features={"daily_ohlcv": frame},
    )

    observed = {"first_price": None, "first_volume": None}

    def _mock_detect_vcp(self, prices, volumes):  # noqa: ARG001
        observed["first_price"] = float(prices.iloc[0])
        observed["first_volume"] = float(volumes.iloc[0])
        return {
            "vcp_detected": True,
            "vcp_score": 82.5,
            "num_bases": 4,
            "contracting_depth": True,
            "contraction_ratio": 0.62,
            "depth_score": 88.0,
            "contracting_volume": True,
            "volume_score": 74.0,
            "tight_near_highs": True,
            "tightness_score": 90.0,
            "atr_score": 71.0,
            "atr_contraction_ratio": 0.68,
            "pivot_info": {
                "pivot": 132.25,
                "distance_pct": 1.8,
                "ready_for_breakout": True,
            },
            "current_price": float(prices.iloc[0]),
            "distance_from_high_pct": 1.2,
        }

    monkeypatch.setattr(vcp_module.VCPDetector, "detect_vcp", _mock_detect_vcp)
    result = VCPWrapperDetector().detect_safe(
        detector_input, DEFAULT_SETUP_ENGINE_PARAMETERS
    )

    assert result.outcome == DetectorOutcome.DETECTED
    best = result.candidates[0]
    assert best["pattern"] == "vcp"
    assert best["pivot_type"] == "vcp_pivot"
    assert best["quality_score"] == 82.5
    assert best["checks"]["vcp_detected_by_legacy"] is True
    assert observed["first_price"] == float(frame["Close"].iat[-1])
    assert observed["first_volume"] == float(frame["Volume"].iat[-1])


def test_vcp_wrapper_propagates_legacy_no_detection_checks(monkeypatch):
    index = pd.bdate_range("2025-01-02", periods=220)
    close = np.linspace(80.0, 130.0, len(index))
    frame = _ohlcv_frame(index=index, close=close)
    detector_input = PatternDetectorInput(
        symbol="VCP",
        timeframe="daily",
        daily_bars=len(frame),
        weekly_bars=60,
        features={"daily_ohlcv": frame},
    )

    def _mock_detect_vcp(self, prices, volumes):  # noqa: ARG001
        return {
            "vcp_detected": False,
            "vcp_score": 54.0,
            "num_bases": 2,
            "contracting_depth": False,
            "tight_near_highs": False,
        }

    monkeypatch.setattr(vcp_module.VCPDetector, "detect_vcp", _mock_detect_vcp)
    result = VCPWrapperDetector().detect_safe(
        detector_input, DEFAULT_SETUP_ENGINE_PARAMETERS
    )
    assert result.outcome == DetectorOutcome.NOT_DETECTED
    assert "vcp_score_below_legacy_threshold" in result.failed_checks
    assert "vcp_insufficient_bases" in result.failed_checks


def test_three_weeks_tight_detects_strict_mode_from_weekly_frame():
    index = pd.date_range("2024-01-05", periods=30, freq="W-FRI")
    close = np.concatenate(
        [
            np.linspace(70.0, 95.0, 24, endpoint=False),
            np.array([100.0, 100.2, 99.9, 100.1, 100.0, 100.15]),
        ]
    )
    frame = _ohlcv_frame(index=index, close=close)
    detector_input = PatternDetectorInput(
        symbol="3WT",
        timeframe="weekly",
        daily_bars=260,
        weekly_bars=len(frame),
        features={"weekly_ohlcv": frame},
    )

    result = ThreeWeeksTightDetector().detect_safe(
        detector_input, DEFAULT_SETUP_ENGINE_PARAMETERS
    )
    assert result.outcome == DetectorOutcome.DETECTED
    best = result.candidates[0]
    assert best["pattern"] == "three_weeks_tight"
    assert best["pivot_type"] == "tight_area_high"
    assert best["metrics"]["weeks_tight"] >= 3
    assert best["metrics"]["tight_mode"] == "strict"
    assert best["checks"]["tight_band_ok"] is True


def test_three_weeks_tight_uses_relaxed_mode_when_strict_fails():
    index = pd.date_range("2024-01-05", periods=30, freq="W-FRI")
    close = np.concatenate(
        [
            np.linspace(70.0, 95.0, 24, endpoint=False),
            np.array([100.0, 101.1, 99.8, 100.6, 100.3, 100.7]),
        ]
    )
    frame = _ohlcv_frame(index=index, close=close)
    detector_input = PatternDetectorInput(
        symbol="3WT",
        timeframe="weekly",
        daily_bars=260,
        weekly_bars=len(frame),
        features={"weekly_ohlcv": frame},
    )

    result = ThreeWeeksTightDetector().detect_safe(
        detector_input, DEFAULT_SETUP_ENGINE_PARAMETERS
    )
    assert result.outcome == DetectorOutcome.DETECTED
    best = result.candidates[0]
    assert best["metrics"]["tight_mode"] == "relaxed"
    assert best["checks"]["tight_mode_relaxed"] is True


def test_three_weeks_tight_resamples_from_daily_when_weekly_absent():
    index = pd.bdate_range("2025-01-02", periods=160)
    close = np.linspace(70.0, 95.0, len(index))
    close[-15:] = np.array(
        [100.0, 100.2, 99.9, 100.1, 100.0, 100.1, 100.2, 100.15, 100.0, 100.1, 100.2, 100.05, 100.0, 100.1, 100.12]
    )
    daily = _ohlcv_frame(index=index, close=close)
    detector_input = PatternDetectorInput(
        symbol="3WT",
        timeframe="daily",
        daily_bars=len(daily),
        weekly_bars=0,
        features={"daily_ohlcv": daily},
    )

    result = ThreeWeeksTightDetector().detect_safe(
        detector_input, DEFAULT_SETUP_ENGINE_PARAMETERS
    )
    assert result.outcome == DetectorOutcome.DETECTED
    assert "weekly_ohlcv_resampled_from_daily" in result.warnings


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
    assert "confidence_penalty_v_shape" in best["metrics"]
    assert "confidence_penalty_weak_handle" in best["metrics"]
    assert "confidence_after_penalties" in best["metrics"]


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


def test_cup_with_handle_emits_confidence_penalties_for_weak_structure(
    monkeypatch,
):
    index = pd.date_range("2023-01-06", periods=120, freq="W-FRI")
    close = np.linspace(80.0, 120.0, len(index))
    frame = _ohlcv_frame(index=index, close=close)
    detector_input = PatternDetectorInput(
        symbol="CUP",
        timeframe="weekly",
        daily_bars=400,
        weekly_bars=len(frame),
        features={"weekly_ohlcv": frame},
    )

    structures = (
        cup_module._CupStructure(
            left_idx=20,
            low_idx=40,
            right_idx=70,
            duration_weeks=50,
            depth_pct=30.0,
            recovery_strength_pct=96.0,
            curvature_balance=0.20,
            recency_weeks=8,
            structure_score=0.78,
        ),
    )

    def _mock_find_cups(_frame):
        return cup_module._CupParseResult(candidates=structures, capped=False)

    def _mock_find_handle(_frame, *, structure):
        return (
            cup_module._HandleCandidate(
                structure=structure,
                start_idx=71,
                end_idx=74,
                duration_weeks=4,
                handle_high=111.0,
                handle_low=98.0,
                pivot_idx=72,
                handle_depth_pct=11.5,
                upper_half_floor=97.0,
                upper_half_margin_pct=1.0,
                volume_ratio=1.05,
                score=0.65,
                recency_weeks=4,
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
    best = result.candidates[0]
    assert best["metrics"]["confidence_penalty_v_shape"] > 0.0
    assert best["metrics"]["confidence_penalty_weak_handle"] > 0.0
    assert (
        best["metrics"]["confidence_after_penalties"]
        < best["metrics"]["confidence_base"]
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
    assert best["metrics"]["pullback_depth_pct_from_high"] > 0.0
    assert best["checks"]["pullback_depth_positive"] is True


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


def test_first_pullback_depth_uses_span_min_low_across_multiple_tests():
    frame = _first_pullback_frame(touch_positions=(80, 90))
    frame.loc[frame.index[80], "Low"] = 98.6
    frame.loc[frame.index[90], "Low"] = 99.5
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
    latest_touch_depth = (
        best["metrics"]["pullback_high_price"] - best["metrics"]["latest_test_low"]
    )
    assert best["metrics"]["tests_count"] == 2
    assert best["metrics"]["pullback_span_low"] == 98.6
    assert best["metrics"]["pullback_depth_points"] > latest_touch_depth
