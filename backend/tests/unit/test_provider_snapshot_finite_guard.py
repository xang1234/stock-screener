from __future__ import annotations

import math

from app.services.provider_snapshot_service import _finite_or_none


def test_finite_or_none_passes_regular_values():
    assert _finite_or_none(0) == 0
    assert _finite_or_none(123) == 123
    assert _finite_or_none(1.5) == 1.5
    assert _finite_or_none(None) is None
    assert _finite_or_none("not-a-number") == "not-a-number"


def test_finite_or_none_rejects_nan_and_inf():
    assert _finite_or_none(float("nan")) is None
    assert _finite_or_none(float("inf")) is None
    assert _finite_or_none(-math.inf) is None
