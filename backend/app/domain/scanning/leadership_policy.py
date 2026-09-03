"""Shared leader thresholds and deterministic ordering."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

LEADERS_MAX_GROUP_RANK = 40
LEADERS_MIN_RS_RATING = 80
LEADERS_PRIMARY_SORT_FIELD = "composite_score"


def _value(row: object, name: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(name)
    return getattr(row, name, None)


def leadership_order_key(row: object) -> tuple[float, str]:
    """Order strongest scores first and resolve ties by canonical symbol."""
    score = _value(row, LEADERS_PRIMARY_SORT_FIELD)
    numeric_score = float(score) if score is not None else float("-inf")
    if not math.isfinite(numeric_score):
        numeric_score = float("-inf")
    symbol = str(_value(row, "symbol") or "").strip().upper()
    return (-numeric_score, symbol)

