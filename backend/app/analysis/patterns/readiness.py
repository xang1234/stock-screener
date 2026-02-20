"""Central breakout-readiness feature computation for Setup Engine.

This module computes readiness primitives once so scanner/report layers can
reuse a single deterministic implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.analysis.patterns.technicals import (
    average_true_range,
    bollinger_bands,
    rolling_percentile_rank,
    rolling_slope,
)


@dataclass(frozen=True)
class BreakoutReadinessFeatures:
    """Computed readiness primitives consumed by setup_engine payload builders."""

    distance_to_pivot_pct: float | None
    atr14_pct: float | None
    atr14_pct_trend: float | None
    bb_width_pct: float | None
    bb_width_pctile_252: float | None
    volume_vs_50d: float | None
    rs: float | None
    rs_line_new_high: bool
    rs_vs_spy_65d: float | None
    rs_vs_spy_trend_20d: float | None


@dataclass(frozen=True)
class BreakoutReadinessTraceInputs:
    """Raw intermediate values captured during readiness computation.

    Used by ``build_score_trace()`` to record formula inputs for each field.
    This is opt-in side-channel data — kept separate from
    ``BreakoutReadinessFeatures`` to avoid polluting the value object.
    """

    latest_close: float | None
    latest_atr14: float | None
    latest_bb_upper: float | None
    latest_bb_lower: float | None
    latest_bb_mid: float | None
    latest_volume: float | None
    volume_sma_50: float | None
    pivot_price: float | None
    bb_pctile_window: int
    latest_benchmark_close: float | None
    rs_current: float | None
    rs_252_max: float | None
    rs_65d_ago: float | None


def _resolve_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    lower_map = {str(col).lower(): col for col in frame.columns}
    if column.lower() in lower_map:
        return frame[lower_map[column.lower()]]
    raise KeyError(f"Missing required column '{column}'")


def _last_valid(series: pd.Series) -> float | None:
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.iloc[-1])


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)


def _compute_readiness_core(
    price_frame: pd.DataFrame,
    *,
    pivot_price: float | None,
    benchmark_close: pd.Series | None = None,
    atr_period: int = 14,
    trend_lookback: int = 20,
    bb_window: int = 20,
    bb_pctile_window: int = 252,
    volume_sma_window: int = 50,
    rs_lookback: int = 252,
    rs_vs_spy_window: int = 65,
) -> tuple[BreakoutReadinessFeatures, BreakoutReadinessTraceInputs]:
    """Compute readiness features and capture raw trace inputs.

    Returns both the canonical features and the raw intermediates needed
    by ``build_score_trace()`` for formula auditability.
    """

    if not isinstance(price_frame, pd.DataFrame):
        raise ValueError("price_frame must be a pandas DataFrame")

    close = _resolve_series(price_frame, "Close").astype(float)
    volume = _resolve_series(price_frame, "Volume").astype(float)
    latest_close = _last_valid(close)

    distance_to_pivot_pct: float | None = None
    if (
        pivot_price is not None
        and latest_close is not None
        and float(pivot_price) != 0.0
    ):
        distance_to_pivot_pct = 100.0 * (
            (latest_close - float(pivot_price)) / float(pivot_price)
        )

    atr14 = average_true_range(
        price_frame,
        period=atr_period,
        method="wilder",
    )
    atr14_latest = _last_valid(atr14)

    atr14_pct: float | None = None
    if atr14_latest is not None and latest_close is not None and latest_close != 0.0:
        atr14_pct = 100.0 * (atr14_latest / latest_close)

    atr14_pct_series = _safe_divide(atr14, close.abs()) * 100.0
    atr14_pct_trend = _last_valid(rolling_slope(atr14_pct_series, window=trend_lookback))

    bands = bollinger_bands(close, window=bb_window, stddev=2.0)
    bb_mid = bands["middle"]
    bb_width_pct_series = (
        _safe_divide((bands["upper"] - bands["lower"]), bb_mid.abs()) * 100.0
    )
    bb_width_pct = _last_valid(bb_width_pct_series)
    bb_width_pctile_252 = _last_valid(
        rolling_percentile_rank(bb_width_pct_series, window=bb_pctile_window)
    )

    volume_sma = volume.rolling(window=volume_sma_window, min_periods=volume_sma_window).mean()
    volume_vs_50d = _last_valid(_safe_divide(volume, volume_sma))

    # Capture raw trace intermediates for volume.
    latest_volume = _last_valid(volume)
    volume_sma_50_val = _last_valid(volume_sma)

    # Capture raw BB intermediates.
    latest_bb_upper = _last_valid(bands["upper"])
    latest_bb_lower = _last_valid(bands["lower"])
    latest_bb_mid = _last_valid(bb_mid)

    rs: float | None = None
    rs_line_new_high = False
    rs_vs_spy_65d: float | None = None
    rs_vs_spy_trend_20d: float | None = None

    # Trace intermediates for RS fields.
    latest_benchmark_close: float | None = None
    rs_current: float | None = None
    rs_252_max: float | None = None
    rs_65d_ago: float | None = None

    if benchmark_close is not None:
        aligned_benchmark = benchmark_close.astype(float).reindex(close.index)
        rs_series = _safe_divide(close, aligned_benchmark)

        rs = _last_valid(rs_series)
        rs_current = rs
        latest_benchmark_close = _last_valid(aligned_benchmark)

        rs_vs_spy_65d = _last_valid(
            (_safe_divide(rs_series, rs_series.shift(rs_vs_spy_window)) - 1.0) * 100.0
        )
        rs_vs_spy_trend_20d = _last_valid(
            rolling_slope(rs_series, window=trend_lookback)
        )

        rs_tail = rs_series.dropna().tail(rs_lookback)
        if not rs_tail.empty:
            rs_line_new_high = bool(rs_tail.iloc[-1] >= rs_tail.max() - 1e-12)
            rs_252_max = float(rs_tail.max())

        # Capture rs value from 65 days ago for trace.
        rs_shifted = rs_series.shift(rs_vs_spy_window).dropna()
        if not rs_shifted.empty:
            rs_65d_ago = float(rs_shifted.iloc[-1])

    features = BreakoutReadinessFeatures(
        distance_to_pivot_pct=distance_to_pivot_pct,
        atr14_pct=atr14_pct,
        atr14_pct_trend=atr14_pct_trend,
        bb_width_pct=bb_width_pct,
        bb_width_pctile_252=bb_width_pctile_252,
        volume_vs_50d=volume_vs_50d,
        rs=rs,
        rs_line_new_high=rs_line_new_high,
        rs_vs_spy_65d=rs_vs_spy_65d,
        rs_vs_spy_trend_20d=rs_vs_spy_trend_20d,
    )

    trace_inputs = BreakoutReadinessTraceInputs(
        latest_close=latest_close,
        latest_atr14=atr14_latest,
        latest_bb_upper=latest_bb_upper,
        latest_bb_lower=latest_bb_lower,
        latest_bb_mid=latest_bb_mid,
        latest_volume=latest_volume,
        volume_sma_50=volume_sma_50_val,
        pivot_price=float(pivot_price) if pivot_price is not None else None,
        bb_pctile_window=bb_pctile_window,
        latest_benchmark_close=latest_benchmark_close,
        rs_current=rs_current,
        rs_252_max=rs_252_max,
        rs_65d_ago=rs_65d_ago,
    )

    return features, trace_inputs


def compute_breakout_readiness_features(
    price_frame: pd.DataFrame,
    *,
    pivot_price: float | None,
    benchmark_close: pd.Series | None = None,
    atr_period: int = 14,
    trend_lookback: int = 20,
    bb_window: int = 20,
    bb_pctile_window: int = 252,
    volume_sma_window: int = 50,
    rs_lookback: int = 252,
    rs_vs_spy_window: int = 65,
) -> BreakoutReadinessFeatures:
    """Compute canonical breakout-readiness fields from OHLCV and benchmark series.

    Formula spec (implemented exactly):
    - distance_to_pivot_pct = 100*(close - pivot)/pivot
    - atr14_pct = 100*ATR14/close
    - atr14_pct_trend = slope(atr14_pct, lookback=20)
    - bb_width_pct = 100*(bb_upper-bb_lower)/bb_mid
    - bb_width_pctile_252 = pct_rank(bb_width_pct[-1], bb_width_pct[-252:])
    - volume_vs_50d = vol / SMA(vol,50)
    - rs = close/spy_close
    """
    features, _trace = _compute_readiness_core(
        price_frame,
        pivot_price=pivot_price,
        benchmark_close=benchmark_close,
        atr_period=atr_period,
        trend_lookback=trend_lookback,
        bb_window=bb_window,
        bb_pctile_window=bb_pctile_window,
        volume_sma_window=volume_sma_window,
        rs_lookback=rs_lookback,
        rs_vs_spy_window=rs_vs_spy_window,
    )
    return features


def compute_breakout_readiness_features_with_trace(
    price_frame: pd.DataFrame,
    *,
    pivot_price: float | None,
    benchmark_close: pd.Series | None = None,
    atr_period: int = 14,
    trend_lookback: int = 20,
    bb_window: int = 20,
    bb_pctile_window: int = 252,
    volume_sma_window: int = 50,
    rs_lookback: int = 252,
    rs_vs_spy_window: int = 65,
) -> tuple[BreakoutReadinessFeatures, BreakoutReadinessTraceInputs]:
    """Compute readiness features and return raw trace inputs for auditability.

    Same computation as ``compute_breakout_readiness_features()`` but also
    returns the ``BreakoutReadinessTraceInputs`` side-channel for
    ``build_score_trace()``.
    """
    return _compute_readiness_core(
        price_frame,
        pivot_price=pivot_price,
        benchmark_close=benchmark_close,
        atr_period=atr_period,
        trend_lookback=trend_lookback,
        bb_window=bb_window,
        bb_pctile_window=bb_pctile_window,
        volume_sma_window=volume_sma_window,
        rs_lookback=rs_lookback,
        rs_vs_spy_window=rs_vs_spy_window,
    )


def readiness_features_to_payload_fields(
    features: BreakoutReadinessFeatures,
) -> dict[str, float | bool | None]:
    """Convert typed readiness features into setup_engine top-level fields."""
    return {
        "distance_to_pivot_pct": features.distance_to_pivot_pct,
        "atr14_pct": features.atr14_pct,
        "atr14_pct_trend": features.atr14_pct_trend,
        "bb_width_pct": features.bb_width_pct,
        "bb_width_pctile_252": features.bb_width_pctile_252,
        "volume_vs_50d": features.volume_vs_50d,
        "rs": features.rs,
        "rs_line_new_high": features.rs_line_new_high,
        "rs_vs_spy_65d": features.rs_vs_spy_65d,
        "rs_vs_spy_trend_20d": features.rs_vs_spy_trend_20d,
    }
