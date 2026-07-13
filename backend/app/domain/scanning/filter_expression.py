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
    ListingDiscoveryFilter,
    MatchOperator,
    RangeFilter,
    TextSearchFilter,
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
    if field == "price":
        return row.get("price", row.get("current_price"))
    if field == "listing_search":
        return f"{row.get('symbol', '')} {row.get('company_name', '')}".strip()
    return row.get(field)


def evaluate_condition(row: Mapping[str, Any], condition: FilterCondition) -> bool:
    """Evaluate one leaf using the documented missing-value policy."""

    if isinstance(condition, ListingDiscoveryFilter):
        if row.get("scan_mode") == "listing_only":
            return True
        volume = row.get("volume")
        if volume is None:
            return False
        try:
            return volume >= condition.min_volume
        except TypeError:
            return False

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
) -> tuple[MatchedGroupDomain, ...]:
    """Return the enabled setup groups satisfied by an already-matching row."""

    return tuple(
        MatchedGroupDomain(id=group.id, name=group.name)
        for group in expression.enabled_groups
        if evaluate_group(row, group)
    )


def annotate_matched_groups(
    items: tuple[ScanResultItemDomain, ...], expression: FilterExpression
) -> tuple[ScanResultItemDomain, ...]:
    """Attach bounded page/export explainability without changing count SQL."""

    annotated: list[ScanResultItemDomain] = []
    for item in items:
        annotated.append(
            replace(
                item,
                matched_groups=matched_setup_groups(
                    scan_result_values(item), expression
                ),
            )
        )
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
    if isinstance(condition, ListingDiscoveryFilter):
        return {
            "kind": "listing_discovery",
            "min_volume": condition.min_volume,
        }
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


def expression_from_payload(payload: Mapping[str, Any]) -> FilterExpression:
    """Hydrate a trusted canonical payload, including static-only fields."""

    def condition_from_payload(item: Mapping[str, Any]) -> FilterCondition:
        kind = item.get("kind")
        if kind == "range":
            return RangeFilter(
                field=str(item["field"]),
                min_value=item.get("min"),
                max_value=item.get("max"),
            )
        if kind == "categorical":
            return CategoricalFilter(
                field=str(item["field"]),
                values=tuple(str(value) for value in item.get("values", ())),
                mode=FilterMode(str(item.get("mode", "include"))),
            )
        if kind == "boolean":
            return BooleanFilter(field=str(item["field"]), value=bool(item["value"]))
        if kind == "text":
            return TextSearchFilter(
                field=str(item["field"]),
                pattern=str(item["pattern"]),
            )
        if kind == "listing_discovery":
            return ListingDiscoveryFilter(min_volume=float(item["min_volume"]))
        raise ValueError(f"Unsupported filter condition kind: {kind!r}")

    def group_from_payload(item: Mapping[str, Any]) -> FilterGroup:
        return FilterGroup(
            id=str(item["id"]),
            name=str(item["name"]),
            match=MatchOperator(str(item.get("match", "all"))),
            enabled=bool(item.get("enabled", True)),
            conditions=tuple(
                condition_from_payload(condition)
                for condition in item.get("conditions", ())
            ),
        )

    required_payload = payload.get("required") or {
        "id": "required",
        "name": "Always require",
        "match": "all",
        "conditions": [],
    }
    return FilterExpression(
        required=group_from_payload(required_payload),
        group_join=MatchOperator(str(payload.get("group_join", "any"))),
        groups=tuple(group_from_payload(group) for group in payload.get("groups", ())),
        version=int(payload.get("expression_version", 1)),
    )


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
    "expression_from_payload",
    "matched_setup_groups",
    "require_passing_ratings",
    "scan_result_values",
]
