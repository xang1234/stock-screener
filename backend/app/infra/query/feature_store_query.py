"""SQLAlchemy query builder for feature store results."""

from __future__ import annotations

from sqlalchemy.orm import Query

from app.domain.common.query import (
    FilterSpec,
    PageSpec,
    SortSpec,
)
from app.domain.scanning.filter_expression_model import (
    FilterExpression,
    filter_spec_to_expression,
)
from app.infra.db.portability import lean_count
from app.infra.db.models.feature_store import StockFeatureDaily
from app.infra.query.sql_filter_compiler import (
    SqlFilterFieldResolver,
    apply_sql_sort,
    column_bindings,
    compile_sql_expression,
    json_bindings,
    listing_aware_volume_predicate,
)
from app.models.stock import StockFundamental
from app.models.stock_universe import StockUniverse

# ── Column resolution ───────────────────────────────────────────────────

# One adapter-owned registry maps each logical field to its physical source.
_FIELD_BINDINGS = column_bindings({
    "symbol": StockFeatureDaily.symbol,
    "symbol_exact": StockFeatureDaily.symbol,
    "composite_score": StockFeatureDaily.composite_score,
    "overall_rating": StockFeatureDaily.overall_rating,
    "passes_count": StockFeatureDaily.passes_count,
    "as_of_date": StockFeatureDaily.as_of_date,
    "rs_line_new_high": StockFeatureDaily.rs_line_new_high,
    "rs_line_new_high_before_price": StockFeatureDaily.rs_line_new_high_before_price,
    "rs_line_blue_dot_recent": StockFeatureDaily.rs_line_blue_dot_recent,
    "rs_line_new_high_date": StockFeatureDaily.rs_line_new_high_date,
    # Joined columns: feature_store_repo applies the matching StockUniverse +
    # StockFundamental outer joins on every read path, so these resolve at
    # SQL time even though they don't live on StockFeatureDaily itself.
    "market": StockUniverse.market,
    "exchange": StockUniverse.exchange,
    "currency": StockUniverse.currency,
    "market_cap_usd": StockFundamental.market_cap_usd,
    "adv_usd": StockFundamental.adv_usd,
}) | json_bindings({
    # Scores
    "minervini_score": ("minervini_score",),
    "canslim_score": ("canslim_score",),
    "ipo_score": ("ipo_score",),
    "custom_score": ("custom_score",),
    "volume_breakthrough_score": ("volume_breakthrough_score",),
    # Price / volume
    "price": ("current_price",),
    "current_price": ("current_price",),
    "volume": ("avg_dollar_volume",),
    "market_cap": ("market_cap",),
    # Technicals
    "stage": ("stage",),
    "rating": ("rating",),
    "rs_rating": ("rs_rating",),
    "rs_rating_1m": ("rs_rating_1m",),
    "rs_rating_3m": ("rs_rating_3m",),
    "rs_rating_12m": ("rs_rating_12m",),
    "adr_percent": ("adr_percent",),
    # Fundamentals
    "eps_growth_qq": ("eps_growth_qq",),
    "sales_growth_qq": ("sales_growth_qq",),
    "eps_growth_yy": ("eps_growth_yy",),
    "sales_growth_yy": ("sales_growth_yy",),
    "peg_ratio": ("peg_ratio",),
    "peg": ("peg_ratio",),
    "eps_rating": ("eps_rating",),
    # Classification
    "ibd_industry_group": ("ibd_industry_group",),
    "ibd_industry_group_search": ("ibd_industry_group",),
    "ibd_group_rank": ("ibd_group_rank",),
    "gics_sector": ("gics_sector",),
    "gics_industry": ("gics_industry",),
    # Performance
    "perf_week": ("perf_week",),
    "perf_month": ("perf_month",),
    "perf_3m": ("perf_3m",),
    "perf_6m": ("perf_6m",),
    # Sparkline meta
    "rs_trend": ("rs_trend",),
    "price_change_1d": ("price_change_1d",),
    "price_trend": ("price_trend",),
    # Beta
    "beta": ("beta",),
    "beta_adj_rs": ("beta_adj_rs",),
    "beta_adj_rs_1m": ("beta_adj_rs_1m",),
    "beta_adj_rs_3m": ("beta_adj_rs_3m",),
    "beta_adj_rs_12m": ("beta_adj_rs_12m",),
    # Distances
    "ema_10_distance": ("ema_10_distance",),
    "ema_20_distance": ("ema_20_distance",),
    "ema_50_distance": ("ema_50_distance",),
    "week_52_high_distance": ("from_52w_high_pct",),
    "week_52_low_distance": ("above_52w_low_pct",),
    # Episodic pivot
    "gap_percent": ("gap_percent",),
    "volume_surge": ("volume_surge",),
    # IPO / dates
    "ipo_date": ("ipo_date",),
    # VCP / details-only fields
    "vcp_score": ("vcp_score",),
    "vcp_pivot": ("vcp_pivot",),
    "vcp_detected": ("vcp_detected",),
    "vcp_ready_for_breakout": ("vcp_ready_for_breakout",),
    "ma_alignment": ("ma_alignment",),
    "stage_name": ("stage_name",),
    "passes_template": ("passes_template",),
    # Pocket Pivot / Power Trend
    "pocket_pivot": ("pocket_pivot",),
    "power_trend": ("power_trend",),
    "vcp_contraction_ratio": ("vcp_contraction_ratio",),
    "vcp_atr_score": ("vcp_atr_score",),
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
    source_name="feature-store",
    bindings=_FIELD_BINDINGS,
    json_column=StockFeatureDaily.details_json,
    symbol_column=StockFeatureDaily.symbol,
    company_name_column=StockUniverse.name,
    range_predicates={"listing_aware_volume": listing_aware_volume_predicate},
)


# ── Public API ──────────────────────────────────────────────────────────


def apply_filters(query: Query, filters: FilterSpec) -> Query:
    """Backward-compatible adapter for existing flat-filter callers."""

    return apply_filter_expression(query, filter_spec_to_expression(filters))


def apply_filter_expression(query: Query, expression: FilterExpression) -> Query:
    return query.filter(compile_filter_expression(query, expression))


def compile_filter_expression(query: Query, expression: FilterExpression):
    return compile_sql_expression(query, expression, _FILTER_FIELD_RESOLVER)


def supported_filter_fields() -> frozenset[str]:
    return _FILTER_FIELD_RESOLVER.supported_filter_fields


def supported_sort_fields() -> frozenset[str]:
    return _FILTER_FIELD_RESOLVER.supported_sort_fields


def apply_sort_and_paginate(
    query: Query,
    sort: SortSpec,
    page: PageSpec,
) -> tuple[list, int]:
    """Apply sort + pagination.  Returns (rows, total_count)."""
    total = lean_count(query)

    query = apply_sql_sort(query, sort, _FILTER_FIELD_RESOLVER)
    query = query.offset(page.offset).limit(page.limit)
    rows = query.all()

    return rows, total


def apply_sort_all(query: Query, sort: SortSpec) -> list:
    """Apply sort and return ALL matching rows (no pagination).

    Used by export-style queries that need every row.
    Unlike the legacy scan_result_query, no Python-sort fallback is needed
    because all feature store fields are SQL-sortable via json_extract.
    """
    return apply_sql_sort(query, sort, _FILTER_FIELD_RESOLVER).all()
