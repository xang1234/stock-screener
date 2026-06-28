from __future__ import annotations

from datetime import date

import pytest

from app.infra.serialization import convert_numpy_types, json_safe


def test_json_safe_and_convert_numpy_types_share_numpy_and_date_conversion():
    np = pytest.importorskip("numpy")

    payload = {
        1: np.array([1.0, np.nan]),
        "date": date(2026, 6, 24),
        "nested": {"value": np.float64("inf")},
    }

    assert json_safe(payload) == {
        "1": [1.0, None],
        "date": "2026-06-24",
        "nested": {"value": None},
    }
    assert convert_numpy_types(payload) == {
        1: [1.0, None],
        "date": "2026-06-24",
        "nested": {"value": None},
    }
