"""Canonical scalar normalization for scan filter conditions."""

from __future__ import annotations

from datetime import date
import math
from typing import Any

from .filter_capabilities import FIELD_CAPABILITIES


def normalize_range_bound(field: str, value: Any) -> int | float | str | None:
    """Normalize one range bound using the shared logical field contract."""

    if value is None:
        return None
    capability = FIELD_CAPABILITIES.get(field)
    if capability is None or capability.filter_kind != "range":
        raise ValueError(f"Unsupported range field: {field}")
    if capability.value_type == "date":
        if not isinstance(value, str):
            raise ValueError(
                f"{capability.field} bounds must use ISO YYYY-MM-DD strings"
            )
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError as exc:
            raise ValueError(
                f"{capability.field} bounds must use ISO YYYY-MM-DD strings"
            ) from exc
    if isinstance(value, bool):
        raise ValueError("Numeric range bounds cannot be booleans")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Numeric range bounds must be finite numbers") from exc
    if not math.isfinite(normalized):
        raise ValueError("Numeric range bounds must be finite numbers")
    return int(normalized) if normalized.is_integer() else normalized


def normalize_listing_min_volume(value: Any) -> int | float:
    if isinstance(value, bool):
        raise ValueError("Listing-discovery volume must be a positive number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Listing-discovery volume must be a positive number") from exc
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError("Listing-discovery volume must be a positive number")
    return int(normalized) if normalized.is_integer() else normalized


__all__ = ["normalize_listing_min_volume", "normalize_range_bound"]
