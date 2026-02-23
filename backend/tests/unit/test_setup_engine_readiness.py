"""Tests for central breakout-readiness feature computation."""

import numpy as np
import pandas as pd
import pytest

from app.analysis.patterns.readiness import (
    compute_breakout_readiness_features,
    readiness_features_to_payload_fields,
)
from app.analysis.patterns.technicals import (
    average_true_range,
    bollinger_bands,
    rolling_percentile_rank,
    rolling_slope,
)


def _price_frame(periods: int = 320) -> pd.DataFrame:
    idx = pd.bdate_range("2025-01-02", periods=periods)
    close = np.linspace(100.0, 220.0, periods)
    return pd.DataFrame(
        {
            "Open": close * 0.995,
            "High": close * 1.02,
            "Low": close * 0.98,
            "Close": close,
            "Volume": np.linspace(1_000_000.0, 1_500_000.0, periods),
        },
        index=idx,
    )


def test_compute_breakout_readiness_features_matches_formula_spec():
    frame = _price_frame()
    benchmark = pd.Series(
        np.linspace(50.0, 90.0, len(frame)),
        index=frame.index,
    )
    pivot_price = 210.0

    computed = compute_breakout_readiness_features(
        frame,
        pivot_price=pivot_price,
        benchmark_close=benchmark,
    )

    close = frame["Close"]
    latest_close = float(close.iloc[-1])
    atr14 = average_true_range(frame, period=14, method="wilder")
    atr14_pct_series = 100.0 * (atr14 / close)

    bb = bollinger_bands(close, window=20, stddev=2.0)
    bb_width_pct_series = 100.0 * ((bb["upper"] - bb["lower"]) / bb["middle"])
    volume_vs_50d_series = frame["Volume"] / frame["Volume"].rolling(50, min_periods=50).mean()

    rs_series = close / benchmark
    rs_vs_spy_65d_series = ((rs_series / rs_series.shift(65)) - 1.0) * 100.0

    assert computed.distance_to_pivot_pct == pytest.approx(
        100.0 * ((latest_close - pivot_price) / pivot_price)
    )
    assert computed.atr14_pct == pytest.approx(100.0 * (float(atr14.iloc[-1]) / latest_close))
    assert computed.atr14_pct_trend == pytest.approx(
        float(rolling_slope(atr14_pct_series, window=20).iloc[-1])
    )
    assert computed.bb_width_pct == pytest.approx(float(bb_width_pct_series.iloc[-1]))
    assert computed.bb_width_pctile_252 == pytest.approx(
        float(rolling_percentile_rank(bb_width_pct_series, window=252).iloc[-1])
    )
    assert computed.volume_vs_50d == pytest.approx(float(volume_vs_50d_series.iloc[-1]))
    assert computed.rs == pytest.approx(float(rs_series.iloc[-1]))
    assert computed.rs_line_new_high is True
    assert computed.rs_vs_spy_65d == pytest.approx(float(rs_vs_spy_65d_series.iloc[-1]))
    assert computed.rs_vs_spy_trend_20d == pytest.approx(
        float(rolling_slope(rs_series, window=20).iloc[-1])
    )

    # bb_squeeze: True when bb_width_pctile_252 is at or below the 20th percentile
    assert isinstance(computed.bb_squeeze, bool)

    # up_down_volume_ratio_10d: 10-bar up-volume / down-volume ratio
    assert computed.up_down_volume_ratio_10d is not None

    # quiet_days_10d: count of quiet days in last 10 bars
    assert isinstance(computed.quiet_days_10d, int)
    assert 0 <= computed.quiet_days_10d <= 10


def test_compute_breakout_readiness_features_without_benchmark_rs_fields_are_null():
    computed = compute_breakout_readiness_features(
        _price_frame(),
        pivot_price=200.0,
        benchmark_close=None,
    )

    assert computed.rs is None
    assert computed.rs_line_new_high is False
    assert computed.rs_vs_spy_65d is None
    assert computed.rs_vs_spy_trend_20d is None


def test_compute_breakout_readiness_features_handles_short_history_with_null_windows():
    computed = compute_breakout_readiness_features(
        _price_frame(periods=25),
        pivot_price=100.0,
    )

    assert computed.atr14_pct is not None
    assert computed.atr14_pct_trend is None
    assert computed.bb_width_pct is not None
    assert computed.bb_width_pctile_252 is None
    assert computed.bb_squeeze is False  # requires bb_width_pctile_252 which is None
    assert computed.volume_vs_50d is None  # needs 50 bars
    assert computed.up_down_volume_ratio_10d is not None  # only needs 10 bars
    assert computed.quiet_days_10d == 0  # volume_sma is NaN (< 50 bars), so no quiet days detected


def test_readiness_features_to_payload_fields_exports_expected_keys():
    computed = compute_breakout_readiness_features(
        _price_frame(),
        pivot_price=200.0,
    )
    payload_fields = readiness_features_to_payload_fields(computed)

    assert set(payload_fields.keys()) == {
        "distance_to_pivot_pct",
        "atr14_pct",
        "atr14_pct_trend",
        "bb_width_pct",
        "bb_width_pctile_252",
        "bb_squeeze",
        "volume_vs_50d",
        "up_down_volume_ratio_10d",
        "quiet_days_10d",
        "rs",
        "rs_line_new_high",
        "rs_vs_spy_65d",
        "rs_vs_spy_trend_20d",
    }

