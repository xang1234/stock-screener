from __future__ import annotations

import pandas as pd

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
