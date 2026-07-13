"""Pure grouped-filter semantics shared by live results and exports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any, Mapping

from app.domain.common.query import (
    BooleanFilter,
    CategoricalFilter,
    FilterCondition,
    FilterExpression,
    FilterGroup,
    FilterMode,
    MatchOperator,
    RangeFilter,
    TextSearchFilter,
)

from .models import ScanResultItemDomain


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
    if field == "price":
        return row.get("price", row.get("current_price"))
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
        if value is None:
            return False
        if isinstance(value, bool):
            return value is condition.value
        if value in (0, 1):
            return bool(value) is condition.value
        return False

    if isinstance(condition, TextSearchFilter):
        if value is None:
            return False
        return condition.pattern.casefold() in str(value).casefold()

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
) -> tuple[dict[str, str], ...]:
    """Return the enabled setup groups satisfied by an already-matching row."""

    return tuple(
        {"id": group.id, "name": group.name}
        for group in expression.enabled_groups
        if evaluate_group(row, group)
    )


def annotate_matched_groups(
    items: tuple[ScanResultItemDomain, ...], expression: FilterExpression
) -> tuple[ScanResultItemDomain, ...]:
    """Attach bounded page/export explainability without changing count SQL."""

    annotated: list[ScanResultItemDomain] = []
    for item in items:
        fields = dict(item.extended_fields or {})
        fields["matched_groups"] = list(
            matched_setup_groups(scan_result_values(item), expression)
        )
        annotated.append(replace(item, extended_fields=fields))
    return tuple(annotated)


def _condition_payload(condition: FilterCondition) -> dict[str, Any]:
    if isinstance(condition, RangeFilter):
        return {
            "kind": "range",
            "field": condition.field,
            "min": condition.min_value,
            "max": condition.max_value,
        }
    if isinstance(condition, CategoricalFilter):
        return {
            "kind": "categorical",
            "field": condition.field,
            "values": sorted(condition.values),
            "mode": condition.mode.value,
        }
    if isinstance(condition, BooleanFilter):
        return {"kind": "boolean", "field": condition.field, "value": condition.value}
    if isinstance(condition, TextSearchFilter):
        return {"kind": "text", "field": condition.field, "pattern": condition.pattern}
    raise TypeError(f"Unsupported filter condition: {type(condition)!r}")


def canonical_expression_payload(expression: FilterExpression) -> dict[str, Any]:
    def group_payload(group: FilterGroup) -> dict[str, Any]:
        return {
            "id": group.id,
            "name": group.name,
            "match": group.match.value,
            "enabled": group.enabled,
            "conditions": [_condition_payload(item) for item in group.conditions],
        }

    return {
        "expression_version": expression.version,
        "required": group_payload(expression.required),
        "group_join": expression.group_join.value,
        "groups": [group_payload(group) for group in expression.groups],
    }


def expression_fingerprint(expression: FilterExpression) -> str:
    payload = json.dumps(
        canonical_expression_payload(expression),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


__all__ = [
    "annotate_matched_groups",
    "canonical_expression_payload",
    "evaluate_condition",
    "evaluate_expression",
    "evaluate_group",
    "expression_fingerprint",
    "matched_setup_groups",
    "scan_result_values",
]
