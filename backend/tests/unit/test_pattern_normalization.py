"""Tests for detector input DataFrame normalization guards."""

import pandas as pd

from app.analysis.patterns.detectors.base import PatternDetectorInput
from app.analysis.patterns.normalization import (
    normalize_detector_input_ohlcv,
    normalize_ohlcv_frame,
)
from app.analysis.patterns.vcp_wrapper import VCPWrapperDetector
from app.analysis.patterns.config import DEFAULT_SETUP_ENGINE_PARAMETERS


def test_normalize_ohlcv_frame_sorts_and_drops_nan_rows():
    idx = pd.to_datetime(["2026-02-05", "2026-02-03", "2026-02-04"])
    frame = pd.DataFrame(
        {
            "open": [10, 11, 12],
            "high": [11, 12, 13],
            "low": [9, 10, 11],
            "close": [10.5, None, 12.5],
            "volume": [100, 120, 130],
        },
        index=idx,
    )

    result = normalize_ohlcv_frame(frame, timeframe="daily", min_bars=2)

    assert result.prerequisites_ok is True
    assert result.frame is not None
    assert list(result.frame.index) == sorted(result.frame.index.tolist())
    assert len(result.frame) == 2
    assert "ohlcv_sorted_chronologically" in result.warnings


def test_normalize_ohlcv_frame_missing_column_fails():
    frame = pd.DataFrame(
        {
            "Open": [10, 11],
            "High": [11, 12],
            "Low": [9, 10],
            "Close": [10.5, 11.5],
        },
        index=pd.bdate_range("2026-01-01", periods=2),
    )

    result = normalize_ohlcv_frame(frame, timeframe="daily", min_bars=2)

    assert result.prerequisites_ok is False
    assert "missing_column_volume" in result.failed_checks


def test_normalize_detector_input_fallback_uses_bar_counts():
    result = normalize_detector_input_ohlcv(
        features={},
        timeframe="weekly",
        min_bars=8,
        feature_key="weekly_ohlcv",
        fallback_bar_count=10,
    )
    assert result.prerequisites_ok is True
    assert result.frame is None
    assert "missing_ohlcv_frame_using_bar_count_fallback" in result.warnings


def test_detector_returns_explicit_non_detection_on_invalid_frame():
    bad_frame = pd.DataFrame(
        {
            "Open": [10, 11],
            "High": [11, 12],
            "Low": [9, 10],
            "Volume": [100, 120],
        },
        index=pd.bdate_range("2026-01-01", periods=2),
    )
    detector_input = PatternDetectorInput(
        symbol="AAPL",
        timeframe="daily",
        daily_bars=200,
        weekly_bars=40,
        features={"daily_ohlcv": bad_frame},
    )

    result = VCPWrapperDetector().detect(detector_input, DEFAULT_SETUP_ENGINE_PARAMETERS)

    assert "insufficient_data" in result.failed_checks
    assert "missing_column_close" in result.failed_checks
