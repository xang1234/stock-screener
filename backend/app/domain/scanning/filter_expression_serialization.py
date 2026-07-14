"""Canonical serialization and identity for scan-filter expressions."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.domain.common.query import (
    BooleanFilter,
    CategoricalFilter,
    RangeFilter,
    TextSearchFilter,
)

from .filter_expression_model import (
    FilterCondition,
    FilterExpression,
    FilterGroup,
)


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


__all__ = ["canonical_expression_payload", "expression_fingerprint"]
