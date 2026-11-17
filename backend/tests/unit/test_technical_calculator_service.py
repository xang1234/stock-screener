from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.technical_calculator_service import TechnicalCalculatorService


def _price_frame_with_volume(volume: list[float]) -> pd.DataFrame:
    periods = len(volume)
    closes = [10.0 + (index * 0.25) for index in range(periods)]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [close + 1.0 for close in closes],
            "Low": [close - 1.0 for close in closes],
            "Close": closes,
            "Volume": volume,
        },
        index=pd.date_range("2026-01-01", periods=periods),
    )


def test_calculate_all_skips_volume_metrics_when_recent_volume_is_missing():
    frame = _price_frame_with_volume([float("nan")] * 60)

    result = TechnicalCalculatorService().calculate_all(frame)

    assert "avg_volume" not in result
    assert "relative_volume" not in result


def test_calculate_all_skips_volume_metrics_when_recent_average_is_infinite():
    volume = [1000.0] * 60
    volume[-2] = float("inf")
    frame = _price_frame_with_volume(volume)

    result = TechnicalCalculatorService().calculate_all(frame)

    assert "avg_volume" not in result
    assert "relative_volume" not in result


def _random_walk_frame(periods: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    closes = 100.0 + np.cumsum(rng.normal(0.0, 1.0, periods))
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes + rng.uniform(0.0, 2.0, periods),
            "Low": closes - rng.uniform(0.0, 2.0, periods),
            "Close": closes,
            "Volume": [1_000_000.0] * periods,
        },
        index=pd.date_range("2025-01-01", periods=periods),
    )


def _true_range(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["Close"].shift(1)
    return pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - prev_close).abs(),
            (frame["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _wilder_atr_reference(frame: pd.DataFrame, period: int = 14) -> float:
    """Textbook Wilder ATR: seed with the mean of the first `period` true ranges, then RMA."""
    true_range = _true_range(frame)
    atr = true_range.iloc[:period].mean()
    for value in true_range.iloc[period:]:
        atr = (atr * (period - 1) + value) / period
    return float(atr)


def test_atr_uses_wilder_smoothing():
    frame = _random_walk_frame()
    expected = _wilder_atr_reference(frame)
    service = TechnicalCalculatorService()

    # Seed differences decay by 13/14 per bar, so after 250 bars only the
    # smoothing constant can move the result at this tolerance.
    assert service._calculate_atr(frame, period=14) == pytest.approx(expected, rel=1e-6)
    assert service.calculate_all(frame)["atr_14"] == round(expected, 2)


def test_atr_seeds_with_first_period_mean_on_short_history():
    # calculate_all accepts 20 bars (recent IPOs), where an ewm seed of the
    # first true range would still dominate the result.
    frame = _random_walk_frame(periods=20)
    frame.iloc[0, frame.columns.get_loc("High")] += 20.0

    atr = TechnicalCalculatorService()._calculate_atr(frame, period=14)

    assert atr == pytest.approx(_wilder_atr_reference(frame), rel=1e-9)


def test_atr_is_not_the_span_ema():
    frame = _random_walk_frame()
    span_ema = float(_true_range(frame).ewm(span=14, adjust=False).mean().iloc[-1])

    atr = TechnicalCalculatorService()._calculate_atr(frame, period=14)

    assert abs(atr - span_ema) > 0.01


def test_atr_skips_bars_with_missing_prices():
    frame = _random_walk_frame()
    frame.iloc[100, frame.columns.get_loc("Close")] = float("nan")

    atr = TechnicalCalculatorService()._calculate_atr(frame, period=14)

    assert atr == pytest.approx(_wilder_atr_reference(frame.dropna()), rel=1e-6)
