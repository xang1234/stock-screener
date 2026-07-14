"""Evaluation and result annotation for canonical scan-filter expressions."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping

from app.domain.common.query import (
    BooleanFilter,
    CategoricalFilter,
    FilterMode,
    RangeFilter,
    TextSearchFilter,
)

from .filter_expression_model import (
    FilterCondition,
    FilterExpression,
    FilterGroup,
    MatchOperator,
)
from .models import MatchedGroupDomain, ScanResultItemDomain


PASSES_ONLY_CONDITION = CategoricalFilter(
    field="rating",
    values=("Strong Buy", "Buy"),
)


def require_passing_ratings(
    expression: FilterExpression,
    *,
    enabled: bool,
) -> FilterExpression:
    """Apply the legacy passes-only policy to one canonical expression."""

    if not enabled or PASSES_ONLY_CONDITION in expression.required.conditions:
        return expression
    return expression.with_required_condition(PASSES_ONLY_CONDITION)


def scan_result_values(item: ScanResultItemDomain) -> dict[str, Any]:
    """Flatten a scan-result domain item into canonical filter field names."""

    values = dict(item.extended_fields or {})
    values.update(
        {
            "symbol": str(item.symbol),
            "composite_score": item.composite_score,
            "rating": item.rating,
            "current_price": item.current_price,
            "price": item.current_price,
        }
    )
    return values


def _row_value(row: Mapping[str, Any], field: str) -> Any:
    if field == "listing_aware_volume":
        return math.inf if row.get("scan_mode") == "listing_only" else row.get("volume")
    if field == "price":
        return row.get("price", row.get("current_price"))
    if field == "listing_search":
        return f"{row.get('symbol', '')} {row.get('company_name', '')}".strip()
    return row.get(field)


def evaluate_condition(row: Mapping[str, Any], condition: FilterCondition) -> bool:
    """Evaluate one leaf using the documented missing-value policy."""

    value = _row_value(row, condition.field)
    if isinstance(condition, RangeFilter):
        if value is None:
            return False
        try:
            if condition.min_value is not None and value < condition.min_value:
                return False
            if condition.max_value is not None and value > condition.max_value:
                return False
        except TypeError:
            return False
        return True
    if isinstance(condition, CategoricalFilter):
        if condition.mode == FilterMode.EXCLUDE:
            return value is None or value not in condition.values
        return value is not None and value in condition.values
    if isinstance(condition, BooleanFilter):
        if isinstance(value, bool):
            return value is condition.value
        if value in (0, 1):
            return bool(value) is condition.value
        return False
    if isinstance(condition, TextSearchFilter):
        return (
            value is not None and condition.pattern.casefold() in str(value).casefold()
        )
    raise TypeError(f"Unsupported filter condition: {type(condition)!r}")


def evaluate_group(row: Mapping[str, Any], group: FilterGroup) -> bool:
    """Evaluate one enabled group; ALL(empty) is true and ANY(empty) is false."""

    if not group.enabled:
        return False
    matches = (evaluate_condition(row, condition) for condition in group.conditions)
    return all(matches) if group.match == MatchOperator.ALL else any(matches)


def evaluate_expression(row: Mapping[str, Any], expression: FilterExpression) -> bool:
    if not evaluate_group(row, expression.required):
        return False
    groups = expression.enabled_groups
    if not groups:
        return True
    matches = (evaluate_group(row, group) for group in groups)
    return all(matches) if expression.group_join == MatchOperator.ALL else any(matches)


def matched_setup_groups(
    row: Mapping[str, Any], expression: FilterExpression
) -> tuple[MatchedGroupDomain, ...]:
    """Return enabled setup groups satisfied by an already-matching row."""

    return tuple(
        MatchedGroupDomain(id=group.id, name=group.name)
        for group in expression.enabled_groups
        if evaluate_group(row, group)
    )


def annotate_matched_groups(
    items: tuple[ScanResultItemDomain, ...], expression: FilterExpression
) -> tuple[ScanResultItemDomain, ...]:
    """Attach bounded page/export explainability without changing count SQL."""

    return tuple(
        replace(
            item,
            matched_groups=matched_setup_groups(scan_result_values(item), expression),
        )
        for item in items
    )


__all__ = [
    "PASSES_ONLY_CONDITION",
    "annotate_matched_groups",
    "evaluate_condition",
    "evaluate_expression",
    "evaluate_group",
    "matched_setup_groups",
    "require_passing_ratings",
    "scan_result_values",
]
