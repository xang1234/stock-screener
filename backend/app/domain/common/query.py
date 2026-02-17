"""Filter, sort, and pagination specifications for domain queries.

These types express query intent in domain terms, independent of
any persistence mechanism.  Adapters translate them into SQL WHERE
clauses, in-memory predicates, or whatever the infra layer requires.

Canonical location: ``app.domain.common.query``.
Backward-compatible re-export from ``app.domain.scanning.filter_spec``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"


class FilterMode(str, Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"


# ---------------------------------------------------------------------------
# Individual Filter Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RangeFilter:
    """Numeric range constraint on a single field."""

    field: str
    min_value: float | int | None = None
    max_value: float | int | None = None

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
        min_value: float | int | None = None,
        max_value: float | int | None = None,
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


@dataclass
class QuerySpec:
    """Complete query = filters + sort + pagination."""

    filters: FilterSpec = field(default_factory=FilterSpec)
    sort: SortSpec = field(default_factory=SortSpec)
    page: PageSpec = field(default_factory=PageSpec)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "SortOrder",
    "FilterMode",
    "RangeFilter",
    "CategoricalFilter",
    "BooleanFilter",
    "TextSearchFilter",
    "FilterSpec",
    "SortSpec",
    "PageSpec",
    "QuerySpec",
]
