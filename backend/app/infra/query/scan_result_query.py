"""SQLAlchemy query builder for scan results."""

from __future__ import annotations

from sqlalchemy.orm import Query

from app.domain.common.query import (
    FilterSpec,
    PageSpec,
    SortOrder,
    SortSpec,
)
from app.domain.scanning.filter_expression_model import (
    FilterExpression,
    filter_spec_to_expression,
)
from app.infra.db.portability import lean_count
from app.infra.query.sql_filter_compiler import (
    SqlFilterFieldResolver,
    apply_sql_sort,
    column_bindings,
    compile_sql_expression,
    json_bindings,
    listing_aware_volume_predicate,
)
from app.models.scan_result import ScanResult
from app.models.stock import StockFundamental
from app.models.stock_universe import StockUniverse

# ── Column resolution ───────────────────────────────────────────────────

# One adapter-owned registry maps each logical field to its physical source.
_FIELD_BINDINGS = column_bindings({
    "symbol": ScanResult.symbol,
    "symbol_exact": ScanResult.symbol,
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
    "ibd_industry_group_search": ScanResult.ibd_industry_group,
    "ibd_group_rank": ScanResult.ibd_group_rank,
    "gics_sector": ScanResult.gics_sector,
    "gics_industry": ScanResult.gics_industry,
    "rs_trend": ScanResult.rs_trend,
    "rs_line_new_high": ScanResult.rs_line_new_high,
    "rs_line_new_high_before_price": ScanResult.rs_line_new_high_before_price,
    "rs_line_blue_dot_recent": ScanResult.rs_line_blue_dot_recent,
    "rs_line_new_high_date": ScanResult.rs_line_new_high_date,
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
    # Joined columns: scan_result_repo always applies the StockUniverse +
    # StockFundamental outer joins, so these resolve at SQL time even though
    # they don't live on ScanResult itself.
    "market": StockUniverse.market,
    "exchange": StockUniverse.exchange,
    "currency": StockUniverse.currency,
    "market_cap_usd": StockFundamental.market_cap_usd,
    "adv_usd": StockFundamental.adv_usd,
}) | json_bindings({
    # VCP
    "vcp_score": ("vcp_score",),
    "vcp_pivot": ("vcp_pivot",),
    "vcp_detected": ("vcp_detected",),
    "vcp_ready_for_breakout": ("vcp_ready_for_breakout",),
    "ma_alignment": ("ma_alignment",),
    # Pocket Pivot / Power Trend
    "pocket_pivot": ("pocket_pivot",),
    "power_trend": ("power_trend",),
    "passes_template": ("passes_template",),
    # Setup Engine (numeric)
    "se_setup_score": ("setup_engine", "setup_score"),
    "se_quality_score": ("setup_engine", "quality_score"),
    "se_readiness_score": ("setup_engine", "readiness_score"),
    "se_pattern_confidence": ("setup_engine", "pattern_confidence"),
    "se_pivot_price": ("setup_engine", "pivot_price"),
    "se_distance_to_pivot_pct": ("setup_engine", "distance_to_pivot_pct"),
    "se_base_length_weeks": ("setup_engine", "base_length_weeks"),
    "se_base_depth_pct": ("setup_engine", "base_depth_pct"),
    "se_support_tests_count": ("setup_engine", "support_tests_count"),
    "se_tight_closes_count": ("setup_engine", "tight_closes_count"),
    "se_atr14_pct": ("setup_engine", "atr14_pct"),
    "se_atr14_pct_trend": ("setup_engine", "atr14_pct_trend"),
    "se_bb_width_pct": ("setup_engine", "bb_width_pct"),
    "se_bb_width_pctile_252": ("setup_engine", "bb_width_pctile_252"),
    "se_volume_vs_50d": ("setup_engine", "volume_vs_50d"),
    "se_up_down_volume_ratio_10d": ("setup_engine", "up_down_volume_ratio_10d"),
    "se_quiet_days_10d": ("setup_engine", "quiet_days_10d"),
    "se_rs": ("setup_engine", "rs"),
    "se_rs_vs_spy_65d": ("setup_engine", "rs_vs_spy_65d"),
    "se_rs_vs_spy_trend_20d": ("setup_engine", "rs_vs_spy_trend_20d"),
    # Setup Engine (boolean)
    "se_setup_ready": ("setup_engine", "setup_ready"),
    "se_rs_line_new_high": ("setup_engine", "rs_line_new_high"),
    "se_rs_line_blue_dot": ("setup_engine", "rs_line_blue_dot"),
    "se_in_early_zone": ("setup_engine", "in_early_zone"),
    "se_extended_from_pivot": ("setup_engine", "extended_from_pivot"),
    "se_bb_squeeze": ("setup_engine", "bb_squeeze"),
    # Setup Engine (string)
    "se_pattern_primary": ("setup_engine", "pattern_primary"),
    "se_pivot_type": ("setup_engine", "pivot_type"),
})

_FILTER_FIELD_RESOLVER = SqlFilterFieldResolver(
    source_name="scan-result",
    bindings=_FIELD_BINDINGS,
    json_column=ScanResult.details,
    symbol_column=ScanResult.symbol,
    company_name_column=StockUniverse.name,
    range_predicates={"listing_aware_volume": listing_aware_volume_predicate},
)

# Sort fields that must be fetched and sorted in Python (not in SQL).
_PYTHON_SORT_FIELDS = frozenset({
    "stage_name",
    "ma_alignment",
    "vcp_detected",
    "passes_template",
})

# Cap for Python-sorted queries to prevent memory issues.
_PYTHON_SORT_LIMIT = 1000


def requires_python_sort(field: str) -> bool:
    """Return True when a sort field needs in-Python sorting."""
    return field in _PYTHON_SORT_FIELDS


# ── Public API ──────────────────────────────────────────────────────────


def apply_filters(query: Query, filters: FilterSpec) -> Query:
    """Backward-compatible adapter for existing flat-filter callers."""

    return apply_filter_expression(query, filter_spec_to_expression(filters))


def apply_filter_expression(query: Query, expression: FilterExpression) -> Query:
    """Apply one bounded grouped expression as a single WHERE predicate."""

    return query.filter(compile_filter_expression(query, expression))


def compile_filter_expression(query: Query, expression: FilterExpression):
    return compile_sql_expression(query, expression, _FILTER_FIELD_RESOLVER)


def supported_filter_fields() -> frozenset[str]:
    return _FILTER_FIELD_RESOLVER.supported_filter_fields


def supported_sort_fields() -> frozenset[str]:
    return _FILTER_FIELD_RESOLVER.supported_sort_fields | _PYTHON_SORT_FIELDS


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
    total = lean_count(query)
    python_sorted = sort.field in _PYTHON_SORT_FIELDS

    if python_sorted:
        rows = query.limit(_PYTHON_SORT_LIMIT).all()
        rows = _sort_in_python(rows, sort)
        rows = rows[page.offset : page.offset + page.limit]
    else:
        query = apply_sql_sort(query, sort, _FILTER_FIELD_RESOLVER)
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

    return apply_sql_sort(query, sort, _FILTER_FIELD_RESOLVER).all()


def _sort_in_python(
    rows: list,
    sort: SortSpec,
) -> list:
    """Sort ScanResult rows (or (ScanResult, ...) tuples) by details JSON field."""

    def get_sort_key(row_obj):
        result = row_obj[0] if isinstance(row_obj, tuple) else row_obj
        detail_value = (
            result.details.get(sort.field) if result.details else None
        )
        if detail_value is None:
            return float("-inf") if sort.order == SortOrder.DESC else float("inf")
        return detail_value

    return sorted(rows, key=get_sort_key, reverse=(sort.order == SortOrder.DESC))
