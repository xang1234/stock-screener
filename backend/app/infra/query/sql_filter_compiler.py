"""Shared SQL leaf compiler for logical scan-filter fields."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field as dataclass_field
from datetime import date
from typing import Any, Mapping

from sqlalchemy import and_, asc, desc, false, func, or_, true
from sqlalchemy.orm import Query

from app.domain.common.query import (
    BooleanFilter,
    CategoricalFilter,
    FilterMode,
    RangeFilter,
    SortOrder,
    SortSpec,
    TextSearchFilter,
)
from app.domain.scanning.filter_capabilities import FIELD_CAPABILITIES
from app.domain.scanning.filter_expression_model import (
    FilterCondition,
    FilterExpression,
    FilterGroup,
    MatchOperator,
)
from app.infra.db.portability import is_postgres, json_number, json_text
from app.infra.query.like_pattern import literal_contains_pattern


RangePredicateCompiler = Callable[
    [Query, RangeFilter, "SqlFilterFieldResolver"],
    Any,
]


@dataclass(frozen=True, slots=True)
class SqlFieldBinding:
    """One adapter-owned physical binding for a logical scan field."""

    column: Any = None
    json_path: tuple[str, ...] | None = None
    numeric_sort: bool = False

    def __post_init__(self) -> None:
        if (self.column is None) == (self.json_path is None):
            raise ValueError("SQL field bindings require exactly one physical source")


def column_bindings(columns: Mapping[str, Any]) -> dict[str, SqlFieldBinding]:
    return {field: SqlFieldBinding(column=column) for field, column in columns.items()}


def json_bindings(
    paths: Mapping[str, tuple[str, ...]],
) -> dict[str, SqlFieldBinding]:
    def is_numeric(field: str) -> bool:
        capability = FIELD_CAPABILITIES.get(field)
        return capability is not None and capability.value_type == "number"

    return {
        field: SqlFieldBinding(
            json_path=path,
            numeric_sort=is_numeric(field),
        )
        for field, path in paths.items()
    }


@dataclass(frozen=True, slots=True)
class SqlFilterFieldResolver:
    """Resolve logical fields against one persistence representation."""

    source_name: str
    bindings: Mapping[str, SqlFieldBinding]
    json_column: Any
    symbol_column: Any
    company_name_column: Any
    scan_mode_path: tuple[str, ...] = ("scan_mode",)
    range_predicates: Mapping[str, RangePredicateCompiler] = dataclass_field(
        default_factory=dict
    )

    @property
    def supported_filter_fields(self) -> frozenset[str]:
        return frozenset((*self.bindings, *self.range_predicates, "listing_search"))

    @property
    def supported_sort_fields(self) -> frozenset[str]:
        return frozenset(self.bindings)


def _unsupported(resolver: SqlFilterFieldResolver, kind: str, field: str) -> ValueError:
    return ValueError(f"Unsupported {resolver.source_name} {kind} field: {field}")


def _json_value(
    query: Query, resolver: SqlFilterFieldResolver, field: str, *, numeric: bool
):
    path = resolver.bindings[field].json_path
    if path is None:
        raise _unsupported(resolver, "JSON", field)
    extractor = json_number if numeric else json_text
    return extractor(resolver.json_column, path, bind_or_session=query)


def _range_predicate(
    query: Query,
    condition: RangeFilter,
    resolver: SqlFilterFieldResolver,
):
    custom_predicate = resolver.range_predicates.get(condition.field)
    if custom_predicate is not None:
        return custom_predicate(query, condition, resolver)

    binding = resolver.bindings.get(condition.field)
    column = binding.column if binding is not None else None
    minimum = condition.min_value
    maximum = condition.max_value
    if column is not None and condition.field == "ipo_date":
        minimum = date.fromisoformat(minimum) if isinstance(minimum, str) else minimum
        maximum = date.fromisoformat(maximum) if isinstance(maximum, str) else maximum

    if column is not None:
        value = column
    elif binding is not None and binding.json_path is not None:
        value = _json_value(
            query,
            resolver,
            condition.field,
            numeric=condition.field != "ipo_date",
        )
    else:
        raise _unsupported(resolver, "range", condition.field)

    predicates = []
    if minimum is not None:
        predicates.extend((value.isnot(None), value >= minimum))
    if maximum is not None:
        predicates.extend((value.isnot(None), value <= maximum))
    return and_(*predicates) if predicates else true()


def _categorical_predicate(
    query: Query,
    condition: CategoricalFilter,
    resolver: SqlFilterFieldResolver,
):
    binding = resolver.bindings.get(condition.field)
    value = binding.column if binding is not None else None
    if value is None and binding is not None and binding.json_path is not None:
        value = _json_value(query, resolver, condition.field, numeric=False)
    if value is None:
        raise _unsupported(resolver, "categorical", condition.field)
    if condition.mode == FilterMode.EXCLUDE:
        return or_(value.is_(None), ~value.in_(condition.values))
    return value.in_(condition.values)


def _boolean_predicate(
    query: Query,
    condition: BooleanFilter,
    resolver: SqlFilterFieldResolver,
):
    binding = resolver.bindings.get(condition.field)
    value = binding.column if binding is not None else None
    if value is not None:
        return and_(value.isnot(None), value == condition.value)
    if binding is None or binding.json_path is None:
        raise _unsupported(resolver, "boolean", condition.field)

    value = _json_value(query, resolver, condition.field, numeric=False)
    if is_postgres(query):
        value = func.lower(value)
        expected = "true" if condition.value else "false"
    else:
        expected = 1 if condition.value else 0
    return and_(value.isnot(None), value == expected)


def _text_predicate(
    query: Query,
    condition: TextSearchFilter,
    resolver: SqlFilterFieldResolver,
):
    pattern = literal_contains_pattern(condition.pattern)
    if condition.field == "listing_search":
        return or_(
            resolver.symbol_column.ilike(pattern, escape="\\"),
            resolver.company_name_column.ilike(pattern, escape="\\"),
        )

    binding = resolver.bindings.get(condition.field)
    value = binding.column if binding is not None else None
    if value is None and binding is not None and binding.json_path is not None:
        value = _json_value(query, resolver, condition.field, numeric=False)
    if value is None:
        raise _unsupported(resolver, "text", condition.field)
    return and_(value.isnot(None), value.ilike(pattern, escape="\\"))


def listing_aware_volume_predicate(
    query: Query,
    condition: RangeFilter,
    resolver: SqlFilterFieldResolver,
):
    scan_mode = json_text(
        resolver.json_column,
        resolver.scan_mode_path,
        bind_or_session=query,
    )
    volume_binding = resolver.bindings.get("volume")
    volume = volume_binding.column if volume_binding is not None else None
    if (
        volume is None
        and volume_binding is not None
        and volume_binding.json_path is not None
    ):
        volume = _json_value(query, resolver, "volume", numeric=True)
    if volume is None:
        raise _unsupported(resolver, "listing-discovery", "volume")
    predicates = [volume.isnot(None)]
    if condition.min_value is not None:
        predicates.append(volume >= condition.min_value)
    if condition.max_value is not None:
        predicates.append(volume <= condition.max_value)
    is_listing_only = scan_mode == "listing_only"
    listing_match = is_listing_only if condition.max_value is None else false()
    regular_match = and_(
        or_(scan_mode.is_(None), scan_mode != "listing_only"),
        *predicates,
    )
    return or_(listing_match, regular_match)


def apply_sql_sort(
    query: Query,
    sort: SortSpec,
    resolver: SqlFilterFieldResolver,
) -> Query:
    """Apply adapter-owned primary sorting with a stable symbol tie-breaker."""

    binding = resolver.bindings.get(sort.field)
    if binding is None:
        raise _unsupported(resolver, "sort", sort.field)

    if binding.column is not None:
        value = binding.column
    else:
        value = _json_value(
            query,
            resolver,
            sort.field,
            numeric=binding.numeric_sort,
        )

    order_fn = asc if sort.order == SortOrder.ASC else desc
    ordered = order_fn(value)
    if binding.json_path is not None or sort.field == "composite_score":
        ordered = ordered.nullslast()
    if sort.field == "symbol":
        return query.order_by(ordered)
    return query.order_by(ordered, asc(resolver.symbol_column))


def compile_sql_condition(
    query: Query,
    condition: FilterCondition,
    resolver: SqlFilterFieldResolver,
):
    """Compile one canonical leaf while the adapter supplies physical fields."""

    if isinstance(condition, RangeFilter):
        return _range_predicate(query, condition, resolver)
    if isinstance(condition, CategoricalFilter):
        return _categorical_predicate(query, condition, resolver)
    if isinstance(condition, BooleanFilter):
        return _boolean_predicate(query, condition, resolver)
    if isinstance(condition, TextSearchFilter):
        return _text_predicate(query, condition, resolver)
    raise TypeError(f"Unsupported filter condition: {type(condition)!r}")


def _compile_sql_conditions(
    query: Query,
    conditions: tuple[FilterCondition, ...],
    match: MatchOperator,
    resolver: SqlFilterFieldResolver,
):
    predicates = [
        compile_sql_condition(query, condition, resolver)
        for condition in conditions
    ]
    if not predicates:
        return true() if match == MatchOperator.ALL else false()
    return and_(*predicates) if match == MatchOperator.ALL else or_(*predicates)


def _compile_sql_group(
    query: Query,
    group: FilterGroup,
    resolver: SqlFilterFieldResolver,
):
    return _compile_sql_conditions(query, group.conditions, group.match, resolver)


def compile_sql_expression(
    query: Query,
    expression: FilterExpression,
    resolver: SqlFilterFieldResolver,
):
    """Compile one canonical expression against an adapter's SQL bindings."""

    required = _compile_sql_conditions(
        query,
        expression.required_conditions,
        MatchOperator.ALL,
        resolver,
    )
    groups = expression.enabled_groups
    if not groups:
        return required
    predicates = [_compile_sql_group(query, group, resolver) for group in groups]
    joined = (
        and_(*predicates)
        if expression.group_join == MatchOperator.ALL
        else or_(*predicates)
    )
    return and_(required, joined)


__all__ = [
    "SqlFieldBinding",
    "SqlFilterFieldResolver",
    "apply_sql_sort",
    "column_bindings",
    "compile_sql_condition",
    "compile_sql_expression",
    "json_bindings",
    "listing_aware_volume_predicate",
]
