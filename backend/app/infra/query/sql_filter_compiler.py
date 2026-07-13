"""Shared SQL leaf compiler for logical scan-filter fields."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

from sqlalchemy import and_, func, or_, true
from sqlalchemy.orm import Query

from app.domain.common.query import (
    BooleanFilter,
    CategoricalFilter,
    FilterCondition,
    FilterMode,
    ListingDiscoveryFilter,
    RangeFilter,
    TextSearchFilter,
)
from app.infra.db.portability import is_postgres, json_number, json_text
from app.infra.query.like_pattern import literal_contains_pattern


@dataclass(frozen=True, slots=True)
class SqlFilterFieldResolver:
    """Resolve logical fields against one persistence representation."""

    source_name: str
    columns: Mapping[str, Any]
    json_paths: Mapping[str, tuple[str, ...]]
    json_column: Any
    symbol_column: Any
    company_name_column: Any
    scan_mode_path: tuple[str, ...] = ("scan_mode",)

    @property
    def supported_filter_fields(self) -> frozenset[str]:
        return frozenset((*self.columns, *self.json_paths, "listing_search"))


def _unsupported(resolver: SqlFilterFieldResolver, kind: str, field: str) -> ValueError:
    return ValueError(f"Unsupported {resolver.source_name} {kind} field: {field}")


def _json_value(query: Query, resolver: SqlFilterFieldResolver, field: str, *, numeric: bool):
    path = resolver.json_paths[field]
    extractor = json_number if numeric else json_text
    return extractor(resolver.json_column, path, bind_or_session=query)


def _range_predicate(
    query: Query,
    condition: RangeFilter,
    resolver: SqlFilterFieldResolver,
):
    column = resolver.columns.get(condition.field)
    minimum = condition.min_value
    maximum = condition.max_value
    if column is not None and condition.field == "ipo_date":
        minimum = date.fromisoformat(minimum) if isinstance(minimum, str) else minimum
        maximum = date.fromisoformat(maximum) if isinstance(maximum, str) else maximum

    if column is not None:
        value = column
    elif condition.field in resolver.json_paths:
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
    value = resolver.columns.get(condition.field)
    if value is None and condition.field in resolver.json_paths:
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
    value = resolver.columns.get(condition.field)
    if value is not None:
        return and_(value.isnot(None), value == condition.value)
    if condition.field not in resolver.json_paths:
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

    value = resolver.columns.get(condition.field)
    if value is None and condition.field in resolver.json_paths:
        value = _json_value(query, resolver, condition.field, numeric=False)
    if value is None:
        raise _unsupported(resolver, "text", condition.field)
    return and_(value.isnot(None), value.ilike(pattern, escape="\\"))


def _listing_discovery_predicate(
    query: Query,
    condition: ListingDiscoveryFilter,
    resolver: SqlFilterFieldResolver,
):
    scan_mode = json_text(
        resolver.json_column,
        resolver.scan_mode_path,
        bind_or_session=query,
    )
    volume = resolver.columns.get("volume")
    if volume is None and "volume" in resolver.json_paths:
        volume = _json_value(query, resolver, "volume", numeric=True)
    if volume is None:
        raise _unsupported(resolver, "listing-discovery", "volume")
    return or_(
        scan_mode == "listing_only",
        and_(volume.isnot(None), volume >= condition.min_volume),
    )


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
    if isinstance(condition, ListingDiscoveryFilter):
        return _listing_discovery_predicate(query, condition, resolver)
    raise TypeError(f"Unsupported filter condition: {type(condition)!r}")


__all__ = ["SqlFilterFieldResolver", "compile_sql_condition"]
