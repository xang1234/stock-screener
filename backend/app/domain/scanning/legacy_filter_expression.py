"""Compatibility conversion from flat scan filters to canonical expressions."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Mapping

from app.domain.common.query import (
    BooleanFilter,
    CategoricalFilter,
    FilterExpression,
    FilterGroup,
    FilterMode,
    ListingDiscoveryFilter,
    RangeFilter,
    TextSearchFilter,
)

from .filter_capabilities import (
    LEGACY_BOOLEAN_FILTER_FIELDS,
    LEGACY_RANGE_FILTER_FIELDS,
)


_IPO_PRESET_MONTHS = {"6m": 6, "1y": 12, "2y": 24, "3y": 36, "5y": 60}


def _ipo_cutoff(value: Any, today: date | None = None) -> str | None:
    if not value:
        return None
    text = str(value)
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        pass
    months = _IPO_PRESET_MONTHS.get(text)
    if months is None:
        return None
    current = today or date.today()
    year = current.year
    month = current.month - months
    while month <= 0:
        month += 12
        year -= 1
    # Mirror Date.setUTCMonth: retain the day and let an invalid target date
    # roll into the following month (for example, Sep 31 becomes Oct 1).
    return (date(year, month, 1) + timedelta(days=current.day - 1)).isoformat()


def legacy_filters_to_expression(
    filters: Mapping[str, Any] | None,
    *,
    today: date | None = None,
) -> FilterExpression:
    """Translate one legacy flat payload at its compatibility boundary."""

    values = filters or {}
    conditions = []

    for key, field in LEGACY_RANGE_FILTER_FIELDS.items():
        range_value = values.get(key)
        if not isinstance(range_value, Mapping):
            continue
        minimum = range_value.get("min")
        maximum = range_value.get("max")
        if minimum is not None or maximum is not None:
            conditions.append(RangeFilter(field, minimum, maximum))

    for key, field in LEGACY_BOOLEAN_FILTER_FIELDS.items():
        value = values.get(key)
        if value is not None:
            if not isinstance(value, bool):
                raise ValueError(f"Legacy boolean filter {key!r} must be a boolean")
            conditions.append(BooleanFilter(field, value))

    symbol_search = str(values.get("symbolSearch") or "").strip()
    if symbol_search:
        conditions.append(TextSearchFilter("listing_search", symbol_search))

    stage = values.get("stage")
    if stage is not None:
        conditions.append(RangeFilter("stage", stage, stage))

    categorical_values = (
        ("rating", values.get("ratings"), "include"),
        (
            "ibd_industry_group",
            (values.get("ibdIndustries") or {}).get("values"),
            (values.get("ibdIndustries") or {}).get("mode", "include"),
        ),
        (
            "gics_sector",
            (values.get("gicsSectors") or {}).get("values"),
            (values.get("gicsSectors") or {}).get("mode", "include"),
        ),
        ("market", values.get("markets"), "include"),
        ("se_pattern_primary", values.get("sePatternPrimary"), "include"),
    )
    for field, selected, mode in categorical_values:
        if selected:
            conditions.append(
                CategoricalFilter(
                    field,
                    tuple(str(item) for item in selected),
                    FilterMode(str(mode)),
                )
            )

    if values.get("passesTemplate") is True:
        conditions.append(
            CategoricalFilter("rating", ("Strong Buy", "Buy"))
        )

    min_volume = values.get("minVolume")
    if min_volume is not None:
        conditions.append(
            ListingDiscoveryFilter(min_volume)
            if symbol_search
            else RangeFilter("volume", min_volume, None)
        )

    min_market_cap = values.get("minMarketCap")
    if min_market_cap is not None:
        conditions.append(RangeFilter("market_cap", min_market_cap, None))

    cutoff = _ipo_cutoff(values.get("ipoAfter"), today)
    if cutoff:
        conditions.append(RangeFilter("ipo_date", cutoff, None))

    return FilterExpression(
        required=FilterGroup(
            id="required",
            name="Always require",
            conditions=tuple(conditions),
        )
    )


__all__ = ["legacy_filters_to_expression"]
