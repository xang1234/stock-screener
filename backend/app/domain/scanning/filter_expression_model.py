"""Scan-owned grouped filter aggregate and query specification."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
import math
import re
from typing import TypeAlias

from app.domain.common.query import (
    BooleanFilter,
    CategoricalFilter,
    FilterMode,
    FilterSpec,
    PageSpec,
    RangeFilter,
    SortSpec,
    TextSearchFilter,
)

from .filter_capabilities import FIELD_CAPABILITIES


class MatchOperator(str, Enum):
    """How conditions inside a group, or setup groups at the root, combine."""

    ALL = "all"
    ANY = "any"


@dataclass(frozen=True)
class ListingDiscoveryFilter:
    """Retain listing-only discovery rows without weakening normal liquidity."""

    min_volume: float | int

    def is_empty(self) -> bool:
        return False


FilterCondition: TypeAlias = (
    RangeFilter
    | CategoricalFilter
    | BooleanFilter
    | TextSearchFilter
    | ListingDiscoveryFilter
)


@dataclass(frozen=True)
class FilterGroup:
    """A named, bounded set of leaf conditions."""

    id: str
    name: str
    match: MatchOperator = MatchOperator.ALL
    conditions: tuple[FilterCondition, ...] = ()
    enabled: bool = True


def _required_group() -> FilterGroup:
    return FilterGroup(id="required", name="Always require")


@dataclass(frozen=True)
class FilterExpression:
    """Bounded grouped filter expression used by every scan-result read path."""

    required: FilterGroup = field(default_factory=_required_group)
    group_join: MatchOperator = MatchOperator.ANY
    groups: tuple[FilterGroup, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        validate_filter_expression(self)

    @property
    def enabled_groups(self) -> tuple[FilterGroup, ...]:
        return tuple(group for group in self.groups if group.enabled)

    @property
    def condition_count(self) -> int:
        return len(self.required.conditions) + sum(
            len(group.conditions) for group in self.groups
        )

    @property
    def is_required_only(self) -> bool:
        return not self.enabled_groups

    def with_required_condition(self, condition: FilterCondition) -> FilterExpression:
        return FilterExpression(
            required=FilterGroup(
                id=self.required.id,
                name=self.required.name,
                match=MatchOperator.ALL,
                conditions=(*self.required.conditions, condition),
                enabled=True,
            ),
            group_join=self.group_join,
            groups=self.groups,
            version=self.version,
        )


MAX_EXPRESSION_GROUPS = 8
MAX_GROUP_CONDITIONS = 20
MAX_EXPRESSION_CONDITIONS = 100
MAX_GROUP_ID_LENGTH = 64
MAX_GROUP_NAME_LENGTH = 60
MAX_TEXT_PATTERN_LENGTH = 100
MAX_CATEGORICAL_VALUES = 100
GROUP_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]*$"
_GROUP_ID_PATTERN = re.compile(GROUP_ID_PATTERN)


def _require_filter_field(field_name: object) -> str:
    if not isinstance(field_name, str) or not field_name:
        raise ValueError("Filter fields must be non-empty strings")
    return field_name


def _validate_filter_condition(condition: FilterCondition) -> None:
    if isinstance(condition, RangeFilter):
        field_name = _require_filter_field(condition.field)
        if condition.is_empty():
            raise ValueError("Range conditions require a minimum or maximum")
        capability = FIELD_CAPABILITIES.get(field_name)
        for bound in (condition.min_value, condition.max_value):
            if bound is None:
                continue
            if capability is not None and capability.value_type == "date":
                if not isinstance(bound, str):
                    raise ValueError(
                        f"{field_name} bounds must use ISO YYYY-MM-DD strings"
                    )
                try:
                    date.fromisoformat(bound)
                except ValueError as exc:
                    raise ValueError(
                        f"{field_name} bounds must use ISO YYYY-MM-DD strings"
                    ) from exc
                continue
            if isinstance(bound, bool):
                raise ValueError("Numeric range bounds cannot be booleans")
            if not isinstance(bound, (int, float)):
                raise ValueError("Numeric range bounds must be numbers")
            if not math.isfinite(float(bound)):
                raise ValueError("Numeric range bounds must be finite numbers")
        if condition.min_value is not None and condition.max_value is not None:
            try:
                if condition.min_value > condition.max_value:
                    raise ValueError("Range minimum cannot exceed maximum")
            except TypeError as exc:
                raise ValueError(
                    "Range bounds must use comparable value types"
                ) from exc
        return

    if isinstance(condition, CategoricalFilter):
        _require_filter_field(condition.field)
        if not isinstance(condition.mode, FilterMode):
            raise ValueError("Categorical modes must be FilterMode values")
        if condition.is_empty():
            raise ValueError("Categorical conditions require at least one value")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in condition.values
        ):
            raise ValueError("Categorical values must be non-empty strings")
        if len(condition.values) > MAX_CATEGORICAL_VALUES:
            raise ValueError(
                f"Categorical conditions allow at most {MAX_CATEGORICAL_VALUES} values"
            )
        return

    if isinstance(condition, BooleanFilter):
        _require_filter_field(condition.field)
        if not isinstance(condition.value, bool):
            raise ValueError("Boolean filter values must be booleans")
        return

    if isinstance(condition, TextSearchFilter):
        _require_filter_field(condition.field)
        if not isinstance(condition.pattern, str) or not condition.pattern.strip():
            raise ValueError("Text patterns cannot be blank")
        if len(condition.pattern) > MAX_TEXT_PATTERN_LENGTH:
            raise ValueError(
                f"Text patterns allow at most {MAX_TEXT_PATTERN_LENGTH} characters"
            )
        return

    if isinstance(condition, ListingDiscoveryFilter):
        if isinstance(condition.min_volume, bool):
            raise ValueError("Listing-discovery volume must be a positive number")
        try:
            value = float(condition.min_volume)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Listing-discovery volume must be a positive number"
            ) from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Listing-discovery volume must be a positive number")
        return

    raise TypeError(f"Unsupported filter condition: {type(condition)!r}")


def validate_filter_expression(expression: FilterExpression) -> None:
    """Enforce canonical scan-expression invariants for every construction path."""

    if type(expression.version) is not int or expression.version != 1:
        raise ValueError("Unsupported filter expression version")
    if not isinstance(expression.group_join, MatchOperator):
        raise ValueError("Group joins must be MatchOperator values")
    required = expression.required
    if not isinstance(required, FilterGroup):
        raise ValueError("The required group must be a FilterGroup")
    if required.id != "required" or required.match != MatchOperator.ALL:
        raise ValueError("The required group must use id='required' and match='all'")
    if not required.enabled:
        raise ValueError("The required group cannot be disabled")
    if len(expression.groups) > MAX_EXPRESSION_GROUPS:
        raise ValueError(
            f"An expression can contain at most {MAX_EXPRESSION_GROUPS} setup groups"
        )

    group_ids: set[str] = set()
    for group in (required, *expression.groups):
        if not isinstance(group, FilterGroup):
            raise ValueError("Expression groups must be FilterGroup values")
        if not isinstance(group.match, MatchOperator):
            raise ValueError("Group match operators must be MatchOperator values")
        if not isinstance(group.enabled, bool):
            raise ValueError("Group enabled flags must be booleans")
        if (
            not isinstance(group.id, str)
            or len(group.id) > MAX_GROUP_ID_LENGTH
            or not _GROUP_ID_PATTERN.fullmatch(group.id)
        ):
            raise ValueError("Group IDs must be 1-64 URL-safe characters")
        if (
            not isinstance(group.name, str)
            or not group.name.strip()
            or len(group.name) > MAX_GROUP_NAME_LENGTH
        ):
            raise ValueError("Group names must be 1-60 non-blank characters")
        if len(group.conditions) > MAX_GROUP_CONDITIONS:
            raise ValueError(
                f"A filter group can contain at most {MAX_GROUP_CONDITIONS} conditions"
            )
        for condition in group.conditions:
            _validate_filter_condition(condition)

    for group in expression.groups:
        if group.id == "required" or group.id in group_ids:
            raise ValueError("Setup group IDs must be unique and cannot use 'required'")
        group_ids.add(group.id)
        if group.enabled and not group.conditions:
            raise ValueError("Enabled setup groups cannot be empty")

    if expression.condition_count > MAX_EXPRESSION_CONDITIONS:
        raise ValueError(
            f"An expression can contain at most {MAX_EXPRESSION_CONDITIONS} conditions"
        )


def filter_spec_to_expression(filters: FilterSpec) -> FilterExpression:
    """Preserve existing flat-filter semantics as one required ALL group."""

    conditions: tuple[FilterCondition, ...] = (
        *filters.range_filters,
        *filters.categorical_filters,
        *filters.boolean_filters,
        *filters.text_searches,
    )
    return FilterExpression(
        required=FilterGroup(
            id="required",
            name="Always require",
            match=MatchOperator.ALL,
            conditions=conditions,
        )
    )


@dataclass(frozen=True)
class QuerySpec:
    """Complete scan-result query using the canonical filter expression."""

    expression: FilterExpression = field(default_factory=FilterExpression)
    sort: SortSpec = field(default_factory=SortSpec)
    page: PageSpec = field(default_factory=PageSpec)

    @classmethod
    def from_filter_spec(
        cls,
        filters: FilterSpec,
        *,
        sort: SortSpec | None = None,
        page: PageSpec | None = None,
    ) -> QuerySpec:
        return cls(
            expression=filter_spec_to_expression(filters),
            sort=sort or SortSpec(),
            page=page or PageSpec(),
        )


__all__ = [
    "FilterCondition",
    "FilterExpression",
    "FilterGroup",
    "ListingDiscoveryFilter",
    "MatchOperator",
    "GROUP_ID_PATTERN",
    "MAX_CATEGORICAL_VALUES",
    "MAX_EXPRESSION_CONDITIONS",
    "MAX_EXPRESSION_GROUPS",
    "MAX_GROUP_CONDITIONS",
    "MAX_GROUP_ID_LENGTH",
    "MAX_GROUP_NAME_LENGTH",
    "MAX_TEXT_PATTERN_LENGTH",
    "QuerySpec",
    "filter_spec_to_expression",
    "validate_filter_expression",
]
