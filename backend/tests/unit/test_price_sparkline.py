"""Regression tests for PriceSparklineCalculator and sparkline schema validation.

CA and DE static-export pipelines crashed because price_sparkline_data contained
30 nulls — ScanResultItem requires List[float] and Pydantic rejected the row.
The regressions below cover the two upstream shapes that produced that output:
an all-None list and an all-NaN price input.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.scanners.criteria.price_sparkline import PriceSparklineCalculator
from app.schemas.scanning import ScanResultItem


def test_all_nan_price_input_returns_none_payload():
    series = pd.Series([float("nan")] * 30)

    result = PriceSparklineCalculator().calculate_price_sparkline(series)

    assert result["price_data"] is None
    assert result["price_trend"] == 0
    assert result["price_change_1d"] is None


def test_all_inf_price_input_returns_none_payload():
    series = pd.Series([math.inf] * 30)

    result = PriceSparklineCalculator().calculate_price_sparkline(series)

    assert result["price_data"] is None


def test_leading_nan_uses_first_finite_fill():
    values = [float("nan")] * 5 + [100.0 + i for i in range(25)]
    series = pd.Series(values)

    result = PriceSparklineCalculator().calculate_price_sparkline(series)

    assert result["price_data"] is not None
    assert len(result["price_data"]) == 30
    for value in result["price_data"]:
        assert math.isfinite(value)


def test_clean_input_still_produces_finite_normalized_series():
    closes = [10.0 + i * 0.5 for i in range(30)]
    series = pd.Series(closes)

    result = PriceSparklineCalculator().calculate_price_sparkline(series)

    assert result["price_data"] is not None
    assert len(result["price_data"]) == 30
    assert result["price_data"][0] == pytest.approx(1.0)
    assert result["price_trend"] == 1


def test_scan_result_item_collapses_all_none_sparkline_to_none():
    item = ScanResultItem(
        symbol="DE-FOO",
        rating="Watch",
        price_sparkline_data=[None] * 30,
        rs_sparkline_data=[None] * 30,
    )

    assert item.price_sparkline_data is None
    assert item.rs_sparkline_data is None


def test_scan_result_item_collapses_single_nan_element_to_none():
    item = ScanResultItem(
        symbol="CA-FOO",
        rating="Watch",
        price_sparkline_data=[1.0] * 29 + [float("nan")],
    )

    assert item.price_sparkline_data is None


def test_scan_result_item_accepts_clean_float_sparkline():
    clean = [1.0 + i * 0.01 for i in range(30)]

    item = ScanResultItem(
        symbol="OK",
        rating="Buy",
        price_sparkline_data=clean,
        rs_sparkline_data=clean,
    )

    assert item.price_sparkline_data == clean
    assert item.rs_sparkline_data == clean


def test_scan_result_item_collapses_numpy_nan_serialized_to_none():
    series = np.full(30, np.nan)
    # Mirror the convert_numpy_types path: NaN floats become None in the list.
    payload = [None if math.isnan(v) else float(v) for v in series]

    item = ScanResultItem(
        symbol="DE-NAN",
        rating="Watch",
        price_sparkline_data=payload,
    )

    assert item.price_sparkline_data is None
