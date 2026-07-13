"""Server-owned scan-filter field and sort capabilities."""

from __future__ import annotations

from typing import Final


RANGE_FIELDS: Final = frozenset(
    {
        "minervini_score", "composite_score", "canslim_score", "ipo_score",
        "custom_score", "volume_breakthrough_score", "se_setup_score",
        "se_distance_to_pivot_pct", "se_base_length_weeks", "se_base_depth_pct",
        "se_support_tests_count", "se_tight_closes_count",
        "se_bb_width_pctile_252", "se_volume_vs_50d",
        "se_up_down_volume_ratio_10d", "se_quiet_days_10d", "rs_rating",
        "rs_rating_1m", "rs_rating_3m", "rs_rating_12m", "price",
        "adr_percent", "eps_growth_qq", "sales_growth_qq", "eps_growth_yy",
        "sales_growth_yy", "peg_ratio", "eps_rating", "ibd_group_rank",
        "volume", "market_cap", "market_cap_usd", "adv_usd", "vcp_score",
        "vcp_pivot", "price_change_1d", "perf_week", "perf_month", "perf_3m",
        "perf_6m", "ema_10_distance", "ema_20_distance", "ema_50_distance",
        "week_52_high_distance", "week_52_low_distance", "ipo_date", "beta",
        "beta_adj_rs", "beta_adj_rs_1m", "beta_adj_rs_3m", "beta_adj_rs_12m",
        "gap_percent", "volume_surge", "stage",
    }
)
CATEGORICAL_FIELDS: Final = frozenset(
    {"rating", "ibd_industry_group", "gics_sector", "market", "se_pattern_primary"}
)
BOOLEAN_FIELDS: Final = frozenset(
    {
        "vcp_detected", "vcp_ready_for_breakout", "ma_alignment", "pocket_pivot",
        "power_trend", "passes_template", "se_setup_ready", "se_rs_line_new_high",
        "se_rs_line_blue_dot", "rs_line_new_high", "rs_line_new_high_before_price",
        "rs_line_blue_dot_recent", "se_in_early_zone", "se_extended_from_pivot",
        "se_bb_squeeze",
    }
)
TEXT_FIELDS: Final = frozenset({"symbol", "listing_search"})

FILTER_FIELD_KINDS: Final = {
    **{field: "range" for field in RANGE_FIELDS},
    **{field: "categorical" for field in CATEGORICAL_FIELDS},
    **{field: "boolean" for field in BOOLEAN_FIELDS},
    **{field: "text" for field in TEXT_FIELDS},
}

LEGACY_RANGE_FILTER_FIELDS: Final = {
    "compositeScore": "composite_score",
    "minerviniScore": "minervini_score",
    "canslimScore": "canslim_score",
    "ipoScore": "ipo_score",
    "customScore": "custom_score",
    "volBreakthroughScore": "volume_breakthrough_score",
    "seSetupScore": "se_setup_score",
    "seDistanceToPivot": "se_distance_to_pivot_pct",
    "seBbSqueeze": "se_bb_width_pctile_252",
    "seVolumeVs50d": "se_volume_vs_50d",
    "seUpDownVolume": "se_up_down_volume_ratio_10d",
    "rsRating": "rs_rating",
    "rs1m": "rs_rating_1m",
    "rs3m": "rs_rating_3m",
    "rs12m": "rs_rating_12m",
    "epsRating": "eps_rating",
    "ibdGroupRank": "ibd_group_rank",
    "price": "price",
    "adrPercent": "adr_percent",
    "epsGrowth": "eps_growth_qq",
    "salesGrowth": "sales_growth_qq",
    "vcpScore": "vcp_score",
    "vcpPivot": "vcp_pivot",
    "perfDay": "price_change_1d",
    "perfWeek": "perf_week",
    "perfMonth": "perf_month",
    "perf3m": "perf_3m",
    "perf6m": "perf_6m",
    "gapPercent": "gap_percent",
    "volumeSurge": "volume_surge",
    "ema10Distance": "ema_10_distance",
    "ema20Distance": "ema_20_distance",
    "ema50Distance": "ema_50_distance",
    "week52HighDistance": "week_52_high_distance",
    "week52LowDistance": "week_52_low_distance",
    "beta": "beta",
    "betaAdjRs": "beta_adj_rs",
    "marketCapUsd": "market_cap_usd",
    "advUsd": "adv_usd",
    # Static percentile ranks are distinct from raw performance fields.
    "pctDay": "pct_day",
    "pctWeek": "pct_week",
    "pctMonth": "pct_month",
}

LEGACY_BOOLEAN_FILTER_FIELDS: Final = {
    "seSetupReady": "se_setup_ready",
    "seRsLineNewHigh": "se_rs_line_new_high",
    "seRsLineBlueDot": "se_rs_line_blue_dot",
    "rsLineBlueDotRecent": "rs_line_blue_dot_recent",
    "vcpDetected": "vcp_detected",
    "vcpReady": "vcp_ready_for_breakout",
    "maAlignment": "ma_alignment",
    "pocketPivot": "pocket_pivot",
    "powerTrend": "power_trend",
}

# Keep this bounded to fields supported by both persistence adapters and exposed
# by current result-table/preset workflows. Unknown values must fail at the API
# boundary instead of silently changing to composite-score ordering.
SORT_FIELDS: Final = frozenset(
    {
        "symbol", "rs_trend", "price_change_1d", "gics_sector",
        "ibd_industry_group", "ibd_group_rank", "composite_score",
        "minervini_score", "canslim_score", "ipo_score", "custom_score",
        "volume_breakthrough_score", "se_setup_score", "se_pattern_primary",
        "se_distance_to_pivot_pct", "se_bb_width_pctile_252",
        "se_volume_vs_50d", "se_up_down_volume_ratio_10d",
        "rs_line_blue_dot_recent", "se_pivot_price", "rs_rating",
        "rs_rating_1m", "rs_rating_3m", "rs_rating_12m", "beta",
        "beta_adj_rs", "eps_rating", "stage", "price", "current_price",
        "volume", "market_cap", "market_cap_usd", "adv_usd", "ipo_date",
        "eps_growth_qq", "sales_growth_qq", "adr_percent", "vcp_score",
        "vcp_pivot", "gap_percent", "volume_surge", "perf_week", "perf_month",
        "perf_3m", "perf_6m", "week_52_high_distance",
    }
)

_BUILDER_FIELDS: Final = (
    ("composite_score", "Composite score", "Scores"),
    ("minervini_score", "Minervini score", "Scores"),
    ("canslim_score", "CANSLIM score", "Scores"),
    ("se_setup_score", "Setup score", "Setups"),
    ("se_distance_to_pivot_pct", "Distance to pivot %", "Setups"),
    ("se_pattern_primary", "Setup pattern", "Setups"),
    ("se_setup_ready", "Setup ready", "Setups"),
    ("vcp_detected", "VCP detected", "Setups"),
    ("vcp_ready_for_breakout", "VCP ready", "Setups"),
    ("pocket_pivot", "Pocket pivot", "Setups"),
    ("power_trend", "Power trend", "Setups"),
    ("rating", "Rating", "Ratings"),
    ("rs_rating", "RS rating", "Ratings"),
    ("rs_rating_1m", "RS rating 1M", "Ratings"),
    ("rs_rating_3m", "RS rating 3M", "Ratings"),
    ("eps_rating", "EPS rating", "Fundamentals"),
    ("eps_growth_qq", "EPS growth Q/Q %", "Fundamentals"),
    ("sales_growth_qq", "Sales growth Q/Q %", "Fundamentals"),
    ("ibd_group_rank", "IBD group rank", "Fundamentals"),
    ("ibd_industry_group", "IBD industry", "Classification"),
    ("gics_sector", "GICS sector", "Classification"),
    ("market", "Market", "Classification"),
    ("price", "Price", "Liquidity"),
    ("market_cap_usd", "Market cap USD", "Liquidity"),
    ("adv_usd", "Average dollar volume USD", "Liquidity"),
    ("adr_percent", "ADR %", "Technicals"),
    ("perf_month", "1-month performance %", "Technicals"),
    ("perf_3m", "3-month performance %", "Technicals"),
    ("ma_alignment", "MA alignment", "Technicals"),
    ("stage", "Stage", "Technicals"),
    ("listing_search", "Symbol or company contains", "Identity"),
)


def filter_field_catalog_payload() -> list[dict[str, object]]:
    """Return the guided-builder catalogue safe for API/static serialization."""

    return [
        {
            "field": field,
            "label": label,
            "type": FILTER_FIELD_KINDS[field],
            "category": category,
            "sortable": field in SORT_FIELDS,
        }
        for field, label, category in _BUILDER_FIELDS
    ]


__all__ = [
    "BOOLEAN_FIELDS",
    "CATEGORICAL_FIELDS",
    "FILTER_FIELD_KINDS",
    "LEGACY_BOOLEAN_FILTER_FIELDS",
    "LEGACY_RANGE_FILTER_FIELDS",
    "RANGE_FIELDS",
    "SORT_FIELDS",
    "TEXT_FIELDS",
    "filter_field_catalog_payload",
]
