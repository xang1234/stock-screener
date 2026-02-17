"""SQLAlchemy query builder for scan results.

Translates domain FilterSpec / SortSpec / PageSpec into SQLAlchemy
WHERE, ORDER BY, and LIMIT/OFFSET clauses.  Handles both indexed
SQL columns and json_extract() for fields stored in the details blob.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Float as SAFloat, and_, asc, cast, desc, func
from sqlalchemy.orm import Query

from app.domain.scanning.filter_spec import (
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
from app.models.scan_result import ScanResult

# ── Column resolution ───────────────────────────────────────────────────

# Maps domain filter/sort field names to ScanResult column attributes.
# Fields NOT in this map are treated as JSON details fields.
_COLUMN_MAP: dict[str, Any] = {
    "symbol": ScanResult.symbol,
    "composite_score": ScanResult.composite_score,
    "minervini_score": ScanResult.minervini_score,
    "canslim_score": ScanResult.canslim_score,
    "ipo_score": ScanResult.ipo_score,
    "custom_score": ScanResult.custom_score,
    "volume_breakthrough_score": ScanResult.volume_breakthrough_score,
    "price": ScanResult.price,
    "current_price": ScanResult.price,  # alias
    "volume": ScanResult.volume,
    "market_cap": ScanResult.market_cap,
    "stage": ScanResult.stage,
    "rating": ScanResult.rating,
    "rs_rating": ScanResult.rs_rating,
    "rs_rating_1m": ScanResult.rs_rating_1m,
    "rs_rating_3m": ScanResult.rs_rating_3m,
    "rs_rating_12m": ScanResult.rs_rating_12m,
    "eps_growth_qq": ScanResult.eps_growth_qq,
    "sales_growth_qq": ScanResult.sales_growth_qq,
    "eps_growth_yy": ScanResult.eps_growth_yy,
    "sales_growth_yy": ScanResult.sales_growth_yy,
    "peg_ratio": ScanResult.peg_ratio,
    "peg": ScanResult.peg_ratio,  # alias
    "adr_percent": ScanResult.adr_percent,
    "eps_rating": ScanResult.eps_rating,
    "ibd_industry_group": ScanResult.ibd_industry_group,
    "ibd_group_rank": ScanResult.ibd_group_rank,
    "gics_sector": ScanResult.gics_sector,
    "gics_industry": ScanResult.gics_industry,
    "rs_trend": ScanResult.rs_trend,
    "price_change_1d": ScanResult.price_change_1d,
    "price_trend": ScanResult.price_trend,
    "perf_week": ScanResult.perf_week,
    "perf_month": ScanResult.perf_month,
    "perf_3m": ScanResult.perf_3m,
    "perf_6m": ScanResult.perf_6m,
    "gap_percent": ScanResult.gap_percent,
    "volume_surge": ScanResult.volume_surge,
    "ema_10_distance": ScanResult.ema_10_distance,
    "ema_20_distance": ScanResult.ema_20_distance,
    "ema_50_distance": ScanResult.ema_50_distance,
    "week_52_high_distance": ScanResult.week_52_high_distance,
    "week_52_low_distance": ScanResult.week_52_low_distance,
    "ipo_date": ScanResult.ipo_date,
    "beta": ScanResult.beta,
    "beta_adj_rs": ScanResult.beta_adj_rs,
    "beta_adj_rs_1m": ScanResult.beta_adj_rs_1m,
    "beta_adj_rs_3m": ScanResult.beta_adj_rs_3m,
    "beta_adj_rs_12m": ScanResult.beta_adj_rs_12m,
}

# JSON details paths for fields stored in the details blob.
# Maps domain field name → json_extract path.
_JSON_FIELD_MAP: dict[str, str] = {
    "vcp_score": "$.vcp_score",
    "vcp_pivot": "$.vcp_pivot",
    "vcp_detected": "$.vcp_detected",
    "vcp_ready_for_breakout": "$.vcp_ready_for_breakout",
    "ma_alignment": "$.ma_alignment",
}

# Sort fields that must be fetched and sorted in Python (not in SQL).
_PYTHON_SORT_FIELDS = frozenset({
    "stage_name",
    "ma_alignment",
    "vcp_detected",
    "passes_template",
})

# Cap for Python-sorted queries to prevent memory issues.
_PYTHON_SORT_LIMIT = 1000


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
) -> tuple[list, int, bool]:
    """Apply sort + pagination.  Returns (rows, total_count, was_python_sorted).

    If the sort field is a SQL column, sorting and pagination happen in SQL.
    If it's a JSON details field, we fetch up to _PYTHON_SORT_LIMIT rows,
    sort in Python, and slice for the requested page.
    """
    total = query.count()
    python_sorted = sort.field in _PYTHON_SORT_FIELDS

    if python_sorted:
        rows = query.limit(_PYTHON_SORT_LIMIT).all()
        rows = _sort_in_python(rows, sort)
        rows = rows[page.offset : page.offset + page.limit]
    else:
        col = _COLUMN_MAP.get(sort.field)
        if col is not None:
            order_fn = asc if sort.order == SortOrder.ASC else desc
            query = query.order_by(order_fn(col))
        else:
            # Unknown sort field — fall back to composite_score desc
            query = query.order_by(desc(ScanResult.composite_score))
        query = query.offset(page.offset).limit(page.limit)
        rows = query.all()

    return rows, total, python_sorted


def apply_sort_all(query: Query, sort: SortSpec) -> list:
    """Apply sort and return ALL matching rows (no pagination).

    Used by export-style queries that need every row.
    """
    if sort.field in _PYTHON_SORT_FIELDS:
        rows = query.all()
        return _sort_in_python(rows, sort)

    col = _COLUMN_MAP.get(sort.field)
    if col is not None:
        order_fn = asc if sort.order == SortOrder.ASC else desc
        query = query.order_by(order_fn(col))
    else:
        query = query.order_by(desc(ScanResult.composite_score))
    return query.all()


# ── Private helpers ─────────────────────────────────────────────────────


def _apply_range_filter(query: Query, rf: RangeFilter) -> Query:
    """Apply a numeric range filter — SQL column or json_extract."""
    col = _COLUMN_MAP.get(rf.field)
    if col is not None:
        # Regular SQL column
        if rf.min_value is not None:
            query = query.filter(col >= rf.min_value)
        if rf.max_value is not None:
            query = query.filter(col <= rf.max_value)
    elif rf.field in _JSON_FIELD_MAP:
        # JSON-extracted numeric field
        json_path = _JSON_FIELD_MAP[rf.field]
        json_val = func.json_extract(ScanResult.details, json_path)
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
        json_val = func.json_extract(ScanResult.details, json_path)
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


def _sort_in_python(
    rows: list[tuple],
    sort: SortSpec,
) -> list[tuple]:
    """Sort (ScanResult, company_name) tuples by a details JSON field."""

    def get_sort_key(row_tuple):
        result = row_tuple[0]
        detail_value = (
            result.details.get(sort.field) if result.details else None
        )
        if detail_value is None:
            return float("-inf") if sort.order == SortOrder.DESC else float("inf")
        return detail_value

    return sorted(rows, key=get_sort_key, reverse=(sort.order == SortOrder.DESC))
