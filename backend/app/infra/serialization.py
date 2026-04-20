"""Shared serialization helpers for the infrastructure layer.

These utilities handle type conversions between external libraries (numpy,
pandas) and Python-native types suitable for JSON/SQLAlchemy persistence.
"""

from __future__ import annotations

from datetime import date, datetime


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


def convert_numpy_types(obj: object) -> object:
    """Recursively convert numpy/pandas types to native Python types.

    Handles: numpy scalars (bool, int, float), ndarrays, NaN/Inf → None,
    datetime/date → ISO string.  Safe to call even when numpy is not
    installed — falls through to the identity return.
    """
    try:
        import numpy as np
    except ImportError:
        # numpy not installed — nothing to convert
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_numpy_types(i) for i in obj]
        return obj

    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif obj is None:
        return None
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    else:
        return obj
