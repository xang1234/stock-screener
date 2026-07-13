"""Filter, sort, and pagination specifications for domain queries.

These types express query intent in domain terms, independent of
any persistence mechanism.  Adapters translate them into SQL WHERE
clauses, in-memory predicates, or whatever the infra layer requires.

Canonical location: ``app.domain.common.query``.
Backward-compatible re-export from ``app.domain.scanning.filter_spec``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
import math
import re
from typing import TypeAlias


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"


class FilterMode(str, Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"


class MatchOperator(str, Enum):
    """How conditions inside a group, or setup groups at the root, combine."""

    ALL = "all"
    ANY = "any"


# ---------------------------------------------------------------------------
# Individual Filter Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RangeFilter:
    """Numeric range constraint on a single field."""

    field: str
    min_value: float | int | str | None = None
    max_value: float | int | str | None = None

    def is_empty(self) -> bool:
        return self.min_value is None and self.max_value is None


@dataclass(frozen=True)
class CategoricalFilter:
    """Include or exclude rows matching specific categorical values."""

    field: str
    values: tuple[str, ...]  # tuple for hashability
    mode: FilterMode = FilterMode.INCLUDE

    def is_empty(self) -> bool:
        return len(self.values) == 0


@dataclass(frozen=True)
class BooleanFilter:
    """Boolean flag constraint on a single field."""

    field: str
    value: bool

    def is_empty(self) -> bool:
        return False  # a boolean filter is never "empty"


@dataclass(frozen=True)
class TextSearchFilter:
    """Substring / pattern search on a text field."""

    field: str
    pattern: str

    def is_empty(self) -> bool:
        return not self.pattern


@dataclass(frozen=True)
class ListingDiscoveryFilter:
    """Retain listing-only discovery rows without weakening normal liquidity."""

    min_volume: float | int

    def is_empty(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Composite Specifications
# ---------------------------------------------------------------------------


@dataclass
class FilterSpec:
    """Holds all active filters.  Mutable for builder-pattern construction.

    Builder methods return ``self`` for fluent chaining and silently
    skip empty / None values so callers don't need guard clauses.
    """

    range_filters: list[RangeFilter] = field(default_factory=list)
    categorical_filters: list[CategoricalFilter] = field(default_factory=list)
    boolean_filters: list[BooleanFilter] = field(default_factory=list)
    text_searches: list[TextSearchFilter] = field(default_factory=list)

    # -- Builder helpers ---------------------------------------------------

    def add_range(
        self,
        field_name: str,
        min_value: float | int | str | None = None,
        max_value: float | int | str | None = None,
    ) -> FilterSpec:
        if min_value is not None or max_value is not None:
            self.range_filters.append(
                RangeFilter(field=field_name, min_value=min_value, max_value=max_value)
            )
        return self

    def add_categorical(
        self,
        field_name: str,
        values: tuple[str, ...] | list[str],
        mode: FilterMode = FilterMode.INCLUDE,
    ) -> FilterSpec:
        vals = tuple(values) if isinstance(values, list) else values
        if vals:
            self.categorical_filters.append(
                CategoricalFilter(field=field_name, values=vals, mode=mode)
            )
        return self

    def add_boolean(self, field_name: str, value: bool) -> FilterSpec:
        self.boolean_filters.append(BooleanFilter(field=field_name, value=value))
        return self

    def add_text_search(self, field_name: str, pattern: str) -> FilterSpec:
        if pattern:
            self.text_searches.append(
                TextSearchFilter(field=field_name, pattern=pattern)
            )
        return self

    def to_expression(self) -> FilterExpression:
        """Convert legacy flat AND filters into the canonical expression."""

        return filter_spec_to_expression(self)


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
    """Bounded grouped filter expression used by every result read path."""

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
        required = FilterGroup(
            id=self.required.id,
            name=self.required.name,
            match=MatchOperator.ALL,
            conditions=(*self.required.conditions, condition),
            enabled=True,
        )
        return FilterExpression(
            required=required,
            group_join=self.group_join,
            groups=self.groups,
            version=self.version,
        )


_GROUP_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _validate_filter_condition(condition: FilterCondition) -> None:
    field_name = getattr(condition, "field", None)
    if field_name is not None and (not isinstance(field_name, str) or not field_name):
        raise ValueError("Filter fields must be non-empty strings")

    if isinstance(condition, RangeFilter):
        if condition.is_empty():
            raise ValueError("Range conditions require a minimum or maximum")
        for bound in (condition.min_value, condition.max_value):
            if bound is None:
                continue
            if condition.field == "ipo_date":
                if not isinstance(bound, str):
                    raise ValueError("IPO date bounds must use ISO YYYY-MM-DD strings")
                try:
                    date.fromisoformat(bound)
                except ValueError as exc:
                    raise ValueError(
                        "IPO date bounds must use ISO YYYY-MM-DD strings"
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
                raise ValueError("Range bounds must use comparable value types") from exc
        return

    if isinstance(condition, CategoricalFilter):
        if condition.is_empty():
            raise ValueError("Categorical conditions require at least one value")
        if any(not isinstance(value, str) or not value.strip() for value in condition.values):
            raise ValueError("Categorical values must be non-empty strings")
        if len(condition.values) > 100:
            raise ValueError("Categorical conditions allow at most 100 values")
        return

    if isinstance(condition, BooleanFilter):
        if not isinstance(condition.value, bool):
            raise ValueError("Boolean filter values must be booleans")
        return

    if isinstance(condition, TextSearchFilter):
        if not isinstance(condition.pattern, str) or not condition.pattern.strip():
            raise ValueError("Text patterns cannot be blank")
        if len(condition.pattern) > 100:
            raise ValueError("Text patterns allow at most 100 characters")
        return

    if isinstance(condition, ListingDiscoveryFilter):
        if isinstance(condition.min_volume, bool):
            raise ValueError("Listing-discovery volume must be a positive number")
        try:
            value = float(condition.min_volume)
        except (TypeError, ValueError) as exc:
            raise ValueError("Listing-discovery volume must be a positive number") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Listing-discovery volume must be a positive number")
        return

    raise TypeError(f"Unsupported filter condition: {type(condition)!r}")


def validate_filter_expression(expression: FilterExpression) -> None:
    """Enforce canonical expression invariants for every construction path."""

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
    if len(expression.groups) > 8:
        raise ValueError("An expression can contain at most 8 setup groups")

    group_ids: set[str] = set()
    for group in (required, *expression.groups):
        if not isinstance(group, FilterGroup):
            raise ValueError("Expression groups must be FilterGroup values")
        if not isinstance(group.match, MatchOperator):
            raise ValueError("Group match operators must be MatchOperator values")
        if not isinstance(group.enabled, bool):
            raise ValueError("Group enabled flags must be booleans")
        if not isinstance(group.id, str) or len(group.id) > 64 or not _GROUP_ID_PATTERN.fullmatch(group.id):
            raise ValueError("Group IDs must be 1-64 URL-safe characters")
        if not isinstance(group.name, str) or not group.name.strip() or len(group.name) > 60:
            raise ValueError("Group names must be 1-60 non-blank characters")
        if len(group.conditions) > 20:
            raise ValueError("A filter group can contain at most 20 conditions")
        for condition in group.conditions:
            _validate_filter_condition(condition)

    for group in expression.groups:
        if group.id == "required" or group.id in group_ids:
            raise ValueError("Setup group IDs must be unique and cannot use 'required'")
        group_ids.add(group.id)
        if group.enabled and not group.conditions:
            raise ValueError("Enabled setup groups cannot be empty")

    if expression.condition_count > 100:
        raise ValueError("An expression can contain at most 100 conditions")


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
class SortSpec:
    """Sort directive for query results."""

    field: str = "composite_score"
    order: SortOrder = SortOrder.DESC


@dataclass
class PageSpec:
    """Pagination parameters with validation."""

    page: int = 1
    per_page: int = 50

    def __post_init__(self) -> None:
        if self.page < 1:
            raise ValueError(f"page must be >= 1, got {self.page}")
        if not (1 <= self.per_page <= 100):
            raise ValueError(f"per_page must be 1-100, got {self.per_page}")

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.per_page

    @property
    def limit(self) -> int:
        return self.per_page


@dataclass(frozen=True)
class QuerySpec:
    """Complete result query using the canonical filter expression."""

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
        """Convert a legacy flat filter at the transport boundary."""

        return cls(
            expression=filter_spec_to_expression(filters),
            sort=sort or SortSpec(),
            page=page or PageSpec(),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "SortOrder",
    "FilterMode",
    "MatchOperator",
    "RangeFilter",
    "CategoricalFilter",
    "BooleanFilter",
    "TextSearchFilter",
    "ListingDiscoveryFilter",
    "FilterSpec",
    "FilterCondition",
    "FilterGroup",
    "FilterExpression",
    "validate_filter_expression",
    "filter_spec_to_expression",
    "SortSpec",
    "PageSpec",
    "QuerySpec",
]
