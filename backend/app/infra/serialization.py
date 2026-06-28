"""Shared serialization helpers for the infrastructure layer.

These utilities handle type conversions between external libraries (numpy,
pandas) and Python-native types suitable for JSON/SQLAlchemy persistence.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from numbers import Integral, Real
from typing import Any, Optional


def finite_float_or_none(value: Any) -> float | None:
    """Return a finite float, or ``None`` for missing/non-finite values."""
    if value is None:
        return None
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except (ImportError, TypeError, ValueError):
        pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _numpy_native(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def json_safe(value: Any, *, stringify_keys: bool = True) -> Any:
    """Convert a value tree into strict JSON-compatible primitives."""
    value = _numpy_native(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return {
            str(key) if stringify_keys else key: json_safe(item, stringify_keys=stringify_keys)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item, stringify_keys=stringify_keys) for item in value]
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except (ImportError, TypeError, ValueError):
        pass
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def sanitize_sparkline(value: Any) -> Optional[list[float]]:
    """Return a finite-float list, or ``None`` if any element is null/non-finite.

    Sparkline payloads serialised through ``convert_numpy_types`` get NaN/Inf
    rewritten to ``None``, which then fails ``List[float]`` validation in the
    response schemas.  Collapsing the whole sparkline to ``None`` keeps the
    surrounding row exportable.
    """
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return None
    sanitized: list[float] = []
    for element in value:
        if element is None:
            return None
        try:
            as_float = float(element)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(as_float):
            return None
        sanitized.append(as_float)
    return sanitized


def normalize_string_list(value: object) -> list[str]:
    """Normalize a scalar-or-sequence value into a clean list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        normalized: list[str] = []
        for item in value:
            if item is None:
                continue
            text = item.strip() if isinstance(item, str) else str(item).strip()
            if text:
                normalized.append(text)
        return normalized
    text = str(value).strip()
    return [text] if text else []


def coerce_bool_or_false(value: object) -> bool:
    """Normalize optional persisted/filter booleans to a strict bool."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def convert_numpy_types(obj: object) -> object:
    """Recursively convert numpy/pandas types to native Python types.

    Handles: numpy scalars (bool, int, float), ndarrays, NaN/Inf → None,
    datetime/date → ISO string.  Safe to call even when numpy is not
    installed — falls through to the identity return.
    """
    return json_safe(obj, stringify_keys=False)
