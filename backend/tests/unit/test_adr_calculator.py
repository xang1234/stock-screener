from __future__ import annotations

import numpy as np
import pandas as pd

from app.scanners.criteria.adr_calculator import ADRCalculator


def _make_price_data(highs, lows, closes) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=len(highs))
    return pd.DataFrame(
        {
            "High": highs,
            "Low": lows,
            "Close": closes,
        },
        index=dates,
    )


def test_calculate_adr_percent_matches_expected_on_clean_input():
    price_data = _make_price_data(
        highs=[105.0, 108.0, 103.0, 104.0, 110.0],
        lows=[100.0, 102.0, 100.0, 99.0, 104.0],
        closes=[102.0, 105.0, 101.0, 100.0, 106.0],
    )

    expected = round(
        float(
            np.mean(
                [
                    ((105.0 - 100.0) / 102.0) * 100.0,
                    ((108.0 - 102.0) / 105.0) * 100.0,
                    ((103.0 - 100.0) / 101.0) * 100.0,
                    ((104.0 - 99.0) / 100.0) * 100.0,
                    ((110.0 - 104.0) / 106.0) * 100.0,
                ]
            )
        ),
        2,
    )

    assert ADRCalculator().calculate_adr_percent(price_data, period=5) == expected


def test_calculate_adr_percent_skips_invalid_rows_but_keeps_valid_threshold():
    price_data = _make_price_data(
        highs=[105.0, 108.0, 103.0, 104.0, 110.0],
        lows=[100.0, 102.0, 100.0, 99.0, 104.0],
        closes=[102.0, 105.0, 101.0, 0.0, 106.0],
    )

    expected = round(
        float(
            np.mean(
                [
                    ((105.0 - 100.0) / 102.0) * 100.0,
                    ((108.0 - 102.0) / 105.0) * 100.0,
                    ((103.0 - 100.0) / 101.0) * 100.0,
                    ((110.0 - 104.0) / 106.0) * 100.0,
                ]
            )
        ),
        2,
    )

    assert ADRCalculator().calculate_adr_percent(price_data, period=5) == expected


def test_calculate_adr_percent_returns_none_when_too_many_rows_are_invalid():
    price_data = _make_price_data(
        highs=[105.0, 108.0, 99.0, np.nan, 110.0],
        lows=[100.0, 102.0, 100.0, 99.0, 104.0],
        closes=[102.0, 0.0, 101.0, 100.0, 0.0],
    )

    assert ADRCalculator().calculate_adr_percent(price_data, period=5) is None
