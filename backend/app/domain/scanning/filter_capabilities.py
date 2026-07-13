"""Canonical logical field descriptors for scan filtering and sorting."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal


FilterKind = Literal["range", "categorical", "boolean", "text"]


@dataclass(frozen=True, slots=True)
class ScanFieldCapability:
    """One logical scan field, independent of its persistence representation."""

    field: str
    filter_kind: FilterKind | None = None
    sortable: bool = False
    api_filter: bool = True
    legacy_key: str | None = None
    builder_label: str | None = None
    builder_category: str | None = None


def _field(
    field: str,
    filter_kind: FilterKind | None = None,
    *,
    sortable: bool = False,
    api_filter: bool = True,
    legacy_key: str | None = None,
    builder: tuple[str, str] | None = None,
) -> ScanFieldCapability:
    label, category = builder or (None, None)
    return ScanFieldCapability(
        field=field,
        filter_kind=filter_kind,
        sortable=sortable,
        api_filter=api_filter,
        legacy_key=legacy_key,
        builder_label=label,
        builder_category=category,
    )


# This is the single logical catalogue. Persistence adapters own only physical
# resolution (column vs JSON path); API sets, legacy aliases, sort allowlists,
# and guided-builder metadata are all derived below.
SCAN_FIELD_CAPABILITIES: Final = (
    _field("composite_score", "range", sortable=True, legacy_key="compositeScore", builder=("Composite score", "Scores")),
    _field("minervini_score", "range", sortable=True, legacy_key="minerviniScore", builder=("Minervini score", "Scores")),
    _field("canslim_score", "range", sortable=True, legacy_key="canslimScore", builder=("CANSLIM score", "Scores")),
    _field("ipo_score", "range", sortable=True, legacy_key="ipoScore"),
    _field("custom_score", "range", sortable=True, legacy_key="customScore"),
    _field("volume_breakthrough_score", "range", sortable=True, legacy_key="volBreakthroughScore"),
    _field("se_setup_score", "range", sortable=True, legacy_key="seSetupScore", builder=("Setup score", "Setups")),
    _field("se_distance_to_pivot_pct", "range", sortable=True, legacy_key="seDistanceToPivot", builder=("Distance to pivot %", "Setups")),
    _field("se_base_length_weeks", "range"),
    _field("se_base_depth_pct", "range"),
    _field("se_support_tests_count", "range"),
    _field("se_tight_closes_count", "range"),
    _field("se_bb_width_pctile_252", "range", sortable=True, legacy_key="seBbSqueeze"),
    _field("se_volume_vs_50d", "range", sortable=True, legacy_key="seVolumeVs50d"),
    _field("se_up_down_volume_ratio_10d", "range", sortable=True, legacy_key="seUpDownVolume"),
    _field("se_quiet_days_10d", "range"),
    _field("rs_rating", "range", sortable=True, legacy_key="rsRating", builder=("RS rating", "Ratings")),
    _field("rs_rating_1m", "range", sortable=True, legacy_key="rs1m", builder=("RS rating 1M", "Ratings")),
    _field("rs_rating_3m", "range", sortable=True, legacy_key="rs3m", builder=("RS rating 3M", "Ratings")),
    _field("rs_rating_12m", "range", sortable=True, legacy_key="rs12m"),
    _field("price", "range", sortable=True, legacy_key="price", builder=("Price", "Liquidity")),
    _field("adr_percent", "range", sortable=True, legacy_key="adrPercent", builder=("ADR %", "Technicals")),
    _field("eps_growth_qq", "range", sortable=True, legacy_key="epsGrowth", builder=("EPS growth Q/Q %", "Fundamentals")),
    _field("sales_growth_qq", "range", sortable=True, legacy_key="salesGrowth", builder=("Sales growth Q/Q %", "Fundamentals")),
    _field("eps_growth_yy", "range"),
    _field("sales_growth_yy", "range"),
    _field("peg_ratio", "range"),
    _field("eps_rating", "range", sortable=True, legacy_key="epsRating", builder=("EPS rating", "Fundamentals")),
    _field("ibd_group_rank", "range", sortable=True, legacy_key="ibdGroupRank", builder=("IBD group rank", "Fundamentals")),
    _field("volume", "range", sortable=True),
    _field("market_cap", "range", sortable=True),
    _field("market_cap_usd", "range", sortable=True, legacy_key="marketCapUsd", builder=("Market cap USD", "Liquidity")),
    _field("adv_usd", "range", sortable=True, legacy_key="advUsd", builder=("Average dollar volume USD", "Liquidity")),
    _field("vcp_score", "range", sortable=True, legacy_key="vcpScore"),
    _field("vcp_pivot", "range", sortable=True, legacy_key="vcpPivot"),
    _field("price_change_1d", "range", sortable=True, legacy_key="perfDay"),
    _field("perf_week", "range", sortable=True, legacy_key="perfWeek"),
    _field("perf_month", "range", sortable=True, legacy_key="perfMonth", builder=("1-month performance %", "Technicals")),
    _field("perf_3m", "range", sortable=True, legacy_key="perf3m", builder=("3-month performance %", "Technicals")),
    _field("perf_6m", "range", sortable=True, legacy_key="perf6m"),
    _field("ema_10_distance", "range", legacy_key="ema10Distance"),
    _field("ema_20_distance", "range", legacy_key="ema20Distance"),
    _field("ema_50_distance", "range", legacy_key="ema50Distance"),
    _field("week_52_high_distance", "range", sortable=True, legacy_key="week52HighDistance"),
    _field("week_52_low_distance", "range", legacy_key="week52LowDistance"),
    _field("ipo_date", "range", sortable=True),
    _field("beta", "range", sortable=True, legacy_key="beta"),
    _field("beta_adj_rs", "range", sortable=True, legacy_key="betaAdjRs"),
    _field("beta_adj_rs_1m", "range"),
    _field("beta_adj_rs_3m", "range"),
    _field("beta_adj_rs_12m", "range"),
    _field("gap_percent", "range", sortable=True, legacy_key="gapPercent"),
    _field("volume_surge", "range", sortable=True, legacy_key="volumeSurge"),
    _field("stage", "range", sortable=True, builder=("Stage", "Technicals")),
    _field("rating", "categorical", builder=("Rating", "Ratings")),
    _field("ibd_industry_group", "categorical", sortable=True, builder=("IBD industry", "Classification")),
    _field("gics_sector", "categorical", sortable=True, builder=("GICS sector", "Classification")),
    _field("market", "categorical", builder=("Market", "Classification")),
    _field("se_pattern_primary", "categorical", sortable=True, builder=("Setup pattern", "Setups")),
    _field("vcp_detected", "boolean", legacy_key="vcpDetected", builder=("VCP detected", "Setups")),
    _field("vcp_ready_for_breakout", "boolean", legacy_key="vcpReady", builder=("VCP ready", "Setups")),
    _field("ma_alignment", "boolean", legacy_key="maAlignment", builder=("MA alignment", "Technicals")),
    _field("pocket_pivot", "boolean", legacy_key="pocketPivot", builder=("Pocket pivot", "Setups")),
    _field("power_trend", "boolean", legacy_key="powerTrend", builder=("Power trend", "Setups")),
    _field("passes_template", "boolean"),
    _field("se_setup_ready", "boolean", legacy_key="seSetupReady", builder=("Setup ready", "Setups")),
    _field("se_rs_line_new_high", "boolean", legacy_key="seRsLineNewHigh"),
    _field("se_rs_line_blue_dot", "boolean", legacy_key="seRsLineBlueDot"),
    _field("rs_line_new_high", "boolean"),
    _field("rs_line_new_high_before_price", "boolean"),
    _field("rs_line_blue_dot_recent", "boolean", sortable=True, legacy_key="rsLineBlueDotRecent"),
    _field("se_in_early_zone", "boolean"),
    _field("se_extended_from_pivot", "boolean"),
    _field("se_bb_squeeze", "boolean"),
    _field("symbol", "text", sortable=True),
    _field("listing_search", "text", builder=("Symbol or company contains", "Identity")),
    # Static-only percentile ranks remain valid compatibility fields but are
    # deliberately absent from the live API allowlist and guided builder.
    _field("pct_day", "range", api_filter=False, legacy_key="pctDay"),
    _field("pct_week", "range", api_filter=False, legacy_key="pctWeek"),
    _field("pct_month", "range", api_filter=False, legacy_key="pctMonth"),
    # Sort-only aliases/metrics.
    _field("current_price", sortable=True),
    _field("rs_trend", sortable=True),
    _field("se_pivot_price", sortable=True),
)


_capability_map = {item.field: item for item in SCAN_FIELD_CAPABILITIES}
if len(_capability_map) != len(SCAN_FIELD_CAPABILITIES):
    raise RuntimeError("Duplicate logical scan field capability")
FIELD_CAPABILITIES: Final = MappingProxyType(_capability_map)

FILTER_FIELD_KINDS: Final = MappingProxyType(
    {
        item.field: item.filter_kind
        for item in SCAN_FIELD_CAPABILITIES
        if item.api_filter and item.filter_kind is not None
    }
)
RANGE_FIELDS: Final = frozenset(
    field for field, kind in FILTER_FIELD_KINDS.items() if kind == "range"
)
CATEGORICAL_FIELDS: Final = frozenset(
    field for field, kind in FILTER_FIELD_KINDS.items() if kind == "categorical"
)
BOOLEAN_FIELDS: Final = frozenset(
    field for field, kind in FILTER_FIELD_KINDS.items() if kind == "boolean"
)
TEXT_FIELDS: Final = frozenset(
    field for field, kind in FILTER_FIELD_KINDS.items() if kind == "text"
)
SORT_FIELDS: Final = frozenset(
    item.field for item in SCAN_FIELD_CAPABILITIES if item.sortable
)
LEGACY_RANGE_FILTER_FIELDS: Final = MappingProxyType(
    {
        item.legacy_key: item.field
        for item in SCAN_FIELD_CAPABILITIES
        if item.legacy_key and item.filter_kind == "range"
    }
)
LEGACY_BOOLEAN_FILTER_FIELDS: Final = MappingProxyType(
    {
        item.legacy_key: item.field
        for item in SCAN_FIELD_CAPABILITIES
        if item.legacy_key and item.filter_kind == "boolean"
    }
)


def filter_field_catalog_payload() -> list[dict[str, object]]:
    """Serialize builder metadata derived from the canonical descriptors."""

    return [
        {
            "field": item.field,
            "label": item.builder_label,
            "type": item.filter_kind,
            "category": item.builder_category,
            "sortable": item.sortable,
        }
        for item in SCAN_FIELD_CAPABILITIES
        if item.builder_label is not None
    ]


__all__ = [
    "BOOLEAN_FIELDS",
    "CATEGORICAL_FIELDS",
    "FIELD_CAPABILITIES",
    "FILTER_FIELD_KINDS",
    "LEGACY_BOOLEAN_FILTER_FIELDS",
    "LEGACY_RANGE_FILTER_FIELDS",
    "RANGE_FIELDS",
    "SCAN_FIELD_CAPABILITIES",
    "SORT_FIELDS",
    "TEXT_FIELDS",
    "ScanFieldCapability",
    "filter_field_catalog_payload",
]
