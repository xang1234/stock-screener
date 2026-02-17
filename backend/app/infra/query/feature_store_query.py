"""SQLAlchemy query builder for feature store results.

Translates domain FilterSpec / SortSpec / PageSpec into SQLAlchemy
WHERE, ORDER BY, and LIMIT/OFFSET clauses for StockFeatureDaily.
Handles both indexed SQL columns and json_extract() for fields
stored in the details_json blob.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Float as SAFloat, and_, asc, cast, desc, func
from sqlalchemy.orm import Query

from app.domain.common.query import (
    BooleanFilter,
    CategoricalFilter,
    FilterMode,
    FilterSpec,
    PageSpec,
    RangeFilter,
    SortOrder,
    SortSpec,
    TextSearchFilter,
)
from app.infra.db.models.feature_store import StockFeatureDaily

# ── Column resolution ───────────────────────────────────────────────────

# Maps domain filter/sort field names to StockFeatureDaily column attributes.
_COLUMN_MAP: dict[str, Any] = {
    "symbol": StockFeatureDaily.symbol,
    "composite_score": StockFeatureDaily.composite_score,
    "overall_rating": StockFeatureDaily.overall_rating,
    "passes_count": StockFeatureDaily.passes_count,
    "as_of_date": StockFeatureDaily.as_of_date,
}

# JSON details paths for fields stored in details_json.
# Forward-compatible with G2 dual-source query — these fields come from
# the scan orchestrator output stored in the details blob.
_JSON_FIELD_MAP: dict[str, str] = {
    "minervini_score": "$.minervini_score",
    "canslim_score": "$.canslim_score",
    "rs_rating": "$.rs_rating",
    "stage": "$.stage",
    "rating": "$.rating",
}


# ── Public API ──────────────────────────────────────────────────────────


def apply_filters(query: Query, filters: FilterSpec) -> Query:
    """Apply all FilterSpec constraints as SQLAlchemy WHERE clauses."""
    for rf in filters.range_filters:
        query = _apply_range_filter(query, rf)
    for cf in filters.categorical_filters:
        query = _apply_categorical_filter(query, cf)
    for bf in filters.boolean_filters:
        query = _apply_boolean_filter(query, bf)
    for ts in filters.text_searches:
        query = _apply_text_search(query, ts)
    return query


def apply_sort_and_paginate(
    query: Query,
    sort: SortSpec,
    page: PageSpec,
) -> tuple[list, int]:
    """Apply sort + pagination.  Returns (rows, total_count)."""
    total = query.count()

    col = _COLUMN_MAP.get(sort.field)
    if col is not None:
        order_fn = asc if sort.order == SortOrder.ASC else desc
        query = query.order_by(order_fn(col))
    elif sort.field in _JSON_FIELD_MAP:
        json_path = _JSON_FIELD_MAP[sort.field]
        json_val = func.json_extract(StockFeatureDaily.details_json, json_path)
        order_fn = asc if sort.order == SortOrder.ASC else desc
        query = query.order_by(order_fn(json_val))
    else:
        # Unknown sort field — fall back to composite_score desc
        query = query.order_by(desc(StockFeatureDaily.composite_score))

    query = query.offset(page.offset).limit(page.limit)
    rows = query.all()

    return rows, total


# ── Private helpers ─────────────────────────────────────────────────────


def _apply_range_filter(query: Query, rf: RangeFilter) -> Query:
    """Apply a numeric range filter — SQL column or json_extract."""
    col = _COLUMN_MAP.get(rf.field)
    if col is not None:
        if rf.min_value is not None:
            query = query.filter(col >= rf.min_value)
        if rf.max_value is not None:
            query = query.filter(col <= rf.max_value)
    elif rf.field in _JSON_FIELD_MAP:
        json_path = _JSON_FIELD_MAP[rf.field]
        json_val = func.json_extract(StockFeatureDaily.details_json, json_path)
        if rf.min_value is not None:
            query = query.filter(and_(
                json_val.isnot(None),
                cast(json_val, SAFloat) >= rf.min_value,
            ))
        if rf.max_value is not None:
            query = query.filter(and_(
                json_val.isnot(None),
                cast(json_val, SAFloat) <= rf.max_value,
            ))
    return query


def _apply_categorical_filter(query: Query, cf: CategoricalFilter) -> Query:
    """Apply an include/exclude categorical filter on a SQL column."""
    col = _COLUMN_MAP.get(cf.field)
    if col is None:
        return query  # unknown field — skip
    if cf.mode == FilterMode.EXCLUDE:
        query = query.filter(~col.in_(cf.values))
    else:
        query = query.filter(col.in_(cf.values))
    return query


def _apply_boolean_filter(query: Query, bf: BooleanFilter) -> Query:
    """Apply a boolean filter — SQL column or json_extract."""
    col = _COLUMN_MAP.get(bf.field)
    if col is not None:
        query = query.filter(col == bf.value)
    elif bf.field in _JSON_FIELD_MAP:
        json_path = _JSON_FIELD_MAP[bf.field]
        json_val = func.json_extract(StockFeatureDaily.details_json, json_path)
        query = query.filter(and_(
            json_val.isnot(None),
            json_val == (1 if bf.value else 0),
        ))
    return query


def _apply_text_search(query: Query, ts: TextSearchFilter) -> Query:
    """Apply a LIKE text search on a SQL column."""
    col = _COLUMN_MAP.get(ts.field)
    if col is not None:
        query = query.filter(col.ilike(f"%{ts.pattern}%"))
    return query
