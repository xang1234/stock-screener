"""Pure grouped-filter semantics shared by live results and exports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from enum import Enum
from typing import Any, Mapping

from app.domain.common.query import (
    BooleanFilter,
    CategoricalFilter,
    FilterMode,
    RangeFilter,
    TextSearchFilter,
)

from .filter_capabilities import FIELD_CAPABILITIES
from .filter_expression_model import (
    FilterCondition,
    FilterExpression,
    FilterGroup,
    ListingDiscoveryFilter,
    MatchOperator,
)
from .filter_values import normalize_listing_min_volume, normalize_range_bound
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


class FilterExpressionDecodePolicy(str, Enum):
    """Select which fields one external expression boundary may accept."""

    API = "api"
    STATIC = "static"


def decode_filter_expression(
    payload: Mapping[str, Any],
    *,
    policy: FilterExpressionDecodePolicy,
) -> FilterExpression:
    """Decode one external expression through the canonical domain codec."""

    if not isinstance(payload, Mapping):
        raise ValueError("Filter expressions must be objects")

    missing = object()

    def string_value(
        item: Mapping[str, Any],
        key: str,
        label: str,
        *,
        default: object = missing,
    ) -> str:
        value = item.get(key, default)
        if not isinstance(value, str):
            raise ValueError(f"{label} must be a string")
        return value

    def filter_field(item: Mapping[str, Any], kind: str) -> str:
        field = string_value(item, "field", "Filter fields")
        capability = FIELD_CAPABILITIES.get(field)
        if capability is None or capability.filter_kind != kind:
            raise ValueError(f"Unsupported {kind} field: {field}")
        if policy == FilterExpressionDecodePolicy.API and not capability.api_filter:
            raise ValueError(f"Unsupported {kind} field: {field}")
        return field

    def condition_from_payload(item: Mapping[str, Any]) -> FilterCondition:
        if not isinstance(item, Mapping):
            raise ValueError("Filter conditions must be objects")
        kind = item.get("kind")
        if kind == "range":
            field = filter_field(item, kind)
            return RangeFilter(
                field=field,
                min_value=normalize_range_bound(field, item.get("min")),
                max_value=normalize_range_bound(field, item.get("max")),
            )
        if kind == "categorical":
            field = filter_field(item, kind)
            raw_values = item.get("values")
            if not isinstance(raw_values, (list, tuple)):
                raise ValueError("Categorical conditions require a values array")
            if any(not isinstance(value, str) for value in raw_values):
                raise ValueError("Categorical filter values must be strings")
            values = tuple(
                dict.fromkeys(value.strip() for value in raw_values if value.strip())
            )
            return CategoricalFilter(
                field=field,
                values=values,
                mode=FilterMode(
                    string_value(
                        item,
                        "mode",
                        "Filter modes",
                        default="include",
                    )
                ),
            )
        if kind == "boolean":
            field = filter_field(item, kind)
            value = item.get("value")
            if not isinstance(value, bool):
                raise ValueError("Boolean filter values must be booleans")
            return BooleanFilter(
                field=field,
                value=value,
            )
        if kind == "text":
            field = filter_field(item, kind)
            pattern = item.get("pattern")
            if not isinstance(pattern, str):
                raise ValueError("Text patterns must be strings")
            return TextSearchFilter(
                field=field,
                pattern=pattern.strip(),
            )
        if kind == "listing_discovery":
            return ListingDiscoveryFilter(
                min_volume=normalize_listing_min_volume(item.get("min_volume"))
            )
        raise ValueError(f"Unsupported filter condition kind: {kind!r}")

    def group_from_payload(item: Mapping[str, Any]) -> FilterGroup:
        if not isinstance(item, Mapping):
            raise ValueError("Filter groups must be objects")
        enabled = item.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("Filter group enabled values must be booleans")
        raw_conditions = item.get("conditions", ())
        if not isinstance(raw_conditions, (list, tuple)):
            raise ValueError("Filter group conditions must be arrays")
        return FilterGroup(
            id=string_value(item, "id", "Filter group IDs"),
            name=string_value(item, "name", "Filter group names").strip(),
            match=MatchOperator(
                string_value(
                    item,
                    "match",
                    "Group match operators",
                    default="all",
                )
            ),
            enabled=enabled,
            conditions=tuple(
                condition_from_payload(condition) for condition in raw_conditions
            ),
        )

    required_payload = payload.get("required")
    if required_payload is None:
        required_payload = {
            "id": "required",
            "name": "Always require",
            "match": "all",
            "conditions": [],
        }
    raw_groups = payload.get("groups", ())
    if not isinstance(raw_groups, (list, tuple)):
        raise ValueError("Filter expression groups must be an array")
    version = payload.get("expression_version", 1)
    if type(version) is not int:
        raise ValueError("Filter expression versions must be integers")
    return FilterExpression(
        required=group_from_payload(required_payload),
        group_join=MatchOperator(
            string_value(
                payload,
                "group_join",
                "Group join operators",
                default="any",
            )
        ),
        groups=tuple(group_from_payload(group) for group in raw_groups),
        version=version,
    )


def expression_from_payload(payload: Mapping[str, Any]) -> FilterExpression:
    """Compatibility name for decoding static/persisted expressions."""

    return decode_filter_expression(
        payload,
        policy=FilterExpressionDecodePolicy.STATIC,
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
    "decode_filter_expression",
    "evaluate_condition",
    "evaluate_expression",
    "evaluate_group",
    "expression_fingerprint",
    "expression_from_payload",
    "FilterExpressionDecodePolicy",
    "matched_setup_groups",
    "normalize_listing_min_volume",
    "normalize_range_bound",
    "require_passing_ratings",
    "scan_result_values",
]
