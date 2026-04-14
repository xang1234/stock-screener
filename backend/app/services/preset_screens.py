"""Preset stock screen definitions for the static site.

Each preset is a named filter configuration using the frontend's camelCase
filter key names (matching scanClient.js and defaultFilters.js). Presets are
embedded in the scan manifest and consumed client-side — no backend scanner
logic is involved at runtime.
"""

from __future__ import annotations

RANGE_FILTER_TO_FIELD: dict[str, str] = {
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
    "rsRating": "rs_rating",
    "rs1m": "rs_rating_1m",
    "rs3m": "rs_rating_3m",
    "rs12m": "rs_rating_12m",
    "epsRating": "eps_rating",
    "price": "current_price",
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
    "pctDay": "pct_day",
    "pctWeek": "pct_week",
    "pctMonth": "pct_month",
}

SCALAR_FILTER_TO_FIELD: dict[str, str] = {
    "minVolume": "volume",
    "minMarketCap": "market_cap",
}

BOOLEAN_FILTER_TO_FIELD: dict[str, str] = {
    "seSetupReady": "se_setup_ready",
    "seRsLineNewHigh": "se_rs_line_new_high",
    "vcpDetected": "vcp_detected",
    "vcpReady": "vcp_ready_for_breakout",
    "maAlignment": "ma_alignment",
    "passesTemplate": "passes_template",
}

# ---------------------------------------------------------------------------
# Preset screen definitions
# ---------------------------------------------------------------------------

PRESET_SCREENS: list[dict] = [
    # -- Tier 1: Score-based (existing scanners) --
    {
        "id": "minervini",
        "name": "Minervini Trend Template",
        "short_name": "Minervini",
        "description": "Stage 2 uptrend stocks passing the 8-point trend template",
        "tier": 1,
        "filters": {
            "minerviniScore": {"min": 70, "max": None},
            "stage": 2,
            "maAlignment": True,
            "rsRating": {"min": 70, "max": None},
        },
        "sort_by": "minervini_score",
        "sort_order": "desc",
    },
    {
        "id": "canslim",
        "name": "CANSLIM",
        "short_name": "CANSLIM",
        "description": "William O'Neil growth screen: strong earnings, RS, and institutional demand",
        "tier": 1,
        "filters": {
            "canslimScore": {"min": 70, "max": None},
            "epsGrowth": {"min": 25, "max": None},
            "rsRating": {"min": 80, "max": None},
        },
        "sort_by": "canslim_score",
        "sort_order": "desc",
    },
    {
        "id": "vcp",
        "name": "VCP Setups",
        "short_name": "VCP",
        "description": "Volatility contraction patterns with Minervini trend confirmation",
        "tier": 1,
        "filters": {
            "vcpDetected": True,
            "minerviniScore": {"min": 50, "max": None},
        },
        "sort_by": "vcp_score",
        "sort_order": "desc",
    },
    {
        "id": "vol_break",
        "name": "Volume Breakthrough",
        "short_name": "Vol Breakout",
        "description": "Record-volume breakouts across 1-year and 5-year lookbacks",
        "tier": 1,
        "filters": {
            "volBreakthroughScore": {"min": 33, "max": None},
        },
        "sort_by": "volume_breakthrough_score",
        "sort_order": "desc",
    },
    # -- Tier 2: Filter-based (composable from existing fields) --
    {
        "id": "episodic_pivot",
        "name": "Episodic Pivot",
        "short_name": "Episodic Pivot",
        "description": "Qullamaggie-style gap-ups on massive volume",
        "tier": 2,
        "filters": {
            "gapPercent": {"min": 10, "max": None},
            "volumeSurge": {"min": 2.0, "max": None},
            "rsRating": {"min": 70, "max": None},
        },
        "sort_by": "gap_percent",
        "sort_order": "desc",
    },
    {
        "id": "momentum",
        "name": "Momentum Leaders",
        "short_name": "Momentum",
        "description": "Top performers over 3-6 months in confirmed Stage 2 uptrends",
        "tier": 2,
        "filters": {
            "perf3m": {"min": 30, "max": None},
            "perf6m": {"min": 80, "max": None},
            "rsRating": {"min": 85, "max": None},
            "stage": 2,
        },
        "sort_by": "perf_6m",
        "sort_order": "desc",
    },
    {
        "id": "kell_growth",
        "name": "Oliver Kell Growth",
        "short_name": "Kell Growth",
        "description": "Growth stocks near highs with strong earnings and sales acceleration",
        "tier": 2,
        "filters": {
            "price": {"min": 20, "max": None},
            "epsGrowth": {"min": 25, "max": None},
            "salesGrowth": {"min": 15, "max": None},
            "rsRating": {"min": 80, "max": None},
            "week52HighDistance": {"min": -15, "max": None},
        },
        "sort_by": "composite_score",
        "sort_order": "desc",
    },
    {
        "id": "rs_power",
        "name": "RS Power Play",
        "short_name": "RS Power",
        "description": "Elite relative strength leaders in Stage 2 uptrends",
        "tier": 2,
        "filters": {
            "rsRating": {"min": 90, "max": None},
            "rs3m": {"min": 85, "max": None},
            "stage": 2,
        },
        "sort_by": "rs_rating",
        "sort_order": "desc",
    },
    {
        "id": "new_highs",
        "name": "New Highs + Volume",
        "short_name": "New Highs",
        "description": "Stocks at or near 52-week highs with above-average volume",
        "tier": 2,
        "filters": {
            "week52HighDistance": {"min": -5, "max": None},
            "volumeSurge": {"min": 1.3, "max": None},
            "rsRating": {"min": 70, "max": None},
        },
        "sort_by": "week_52_high_distance",
        "sort_order": "desc",
    },
    {
        "id": "growth",
        "name": "Growth Rockets",
        "short_name": "Growth",
        "description": "Triple-digit EPS and sales growth with strong relative strength",
        "tier": 2,
        "filters": {
            "epsGrowth": {"min": 40, "max": None},
            "salesGrowth": {"min": 30, "max": None},
            "rsRating": {"min": 70, "max": None},
        },
        "sort_by": "eps_growth_qq",
        "sort_order": "desc",
    },
    {
        "id": "tight",
        "name": "Tight Setups",
        "short_name": "Tight",
        "description": "Low-volatility bases with VCP characteristics in strong uptrends",
        "tier": 2,
        "filters": {
            "adrPercent": {"min": None, "max": 4},
            "rsRating": {"min": 80, "max": None},
            "vcpScore": {"min": 30, "max": None},
            "stage": 2,
        },
        "sort_by": "vcp_score",
        "sort_order": "desc",
    },
    {
        "id": "ipo",
        "name": "Recent IPOs",
        "short_name": "IPOs",
        "description": "High-scoring recent IPOs with strong early price action",
        "tier": 2,
        "filters": {
            "ipoScore": {"min": 50, "max": None},
        },
        "sort_by": "ipo_score",
        "sort_order": "desc",
    },
    # -- Tier 3: Super Scanners (Market-Metrics inspired) --
    {
        "id": "gainers_4pct",
        "name": "4% Daily Gainers",
        "short_name": "4% Gainers",
        "description": "Stocks advancing 4%+ intraday — quick broad-market momentum scan",
        "tier": 3,
        "filters": {
            "perfDay": {"min": 4, "max": None},
        },
        "sort_by": "price_change_1d",
        "sort_order": "desc",
    },
    {
        "id": "movers_9m",
        "name": "9M Movers",
        "short_name": "9M Movers",
        "description": "Heavy-volume movers: $100M+ dollar volume with 1.25x+ relative volume surge",
        "tier": 3,
        "filters": {
            # minVolume is dollar volume (avg_volume * price), not share count.
            # 9M-shares proxy ≈ $100M dollar-volume floor (liquid large-caps).
            "minVolume": 100_000_000,
            "volumeSurge": {"min": 1.25, "max": None},
        },
        "sort_by": "price_change_1d",
        "sort_order": "desc",
    },
    {
        "id": "movers_20_weekly",
        "name": "20% Weekly Movers",
        "short_name": "20% Weekly",
        "description": "Stocks up 20%+ over the past 5 sessions",
        "tier": 3,
        "filters": {
            "perfWeek": {"min": 20, "max": None},
        },
        "sort_by": "perf_week",
        "sort_order": "desc",
    },
    {
        "id": "club_97",
        "name": "97 Club",
        "short_name": "97 Club",
        "description": "Top 3% percentile movers across day/week/month horizons simultaneously",
        "tier": 3,
        "filters": {
            "pctDay": {"min": 97, "max": None},
            "pctWeek": {"min": 97, "max": None},
            "pctMonth": {"min": 97, "max": None},
        },
        "sort_by": "price_change_1d",
        "sort_order": "desc",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _matches_preset_filters(row: dict, filters: dict) -> bool:
    """Check if a serialized scan row matches a preset's filter criteria.

    Mirrors the logic in frontend/src/static/scanClient.js filterStaticScanRows.
    """
    for key, value in filters.items():
        # Stage filter (integer equality)
        if key == "stage":
            if row.get("stage") != value:
                return False
            continue

        if key in SCALAR_FILTER_TO_FIELD:
            field = SCALAR_FILTER_TO_FIELD[key]
            row_val = row.get(field)
            if row_val is None or row_val < value:
                return False
            continue

        # Boolean filter
        if key in BOOLEAN_FILTER_TO_FIELD:
            field = BOOLEAN_FILTER_TO_FIELD[key]
            if bool(row.get(field)) != value:
                return False
            continue

        # Range filter
        if key in RANGE_FILTER_TO_FIELD:
            field = RANGE_FILTER_TO_FIELD[key]
            row_val = row.get(field)
            if isinstance(value, dict):
                if row_val is None:
                    return False
                if value.get("min") is not None and row_val < value["min"]:
                    return False
                if value.get("max") is not None and row_val > value["max"]:
                    return False
            continue

    return True


def get_preset_chart_symbols(
    serialized_rows: list[dict],
    presets: list[dict] | None = None,
    top_n: int = 50,
) -> set[str]:
    """Return the union of top-N symbols per preset screen.

    Used by the chart export to expand chart coverage beyond the default
    top-200 composite-score ranking.
    """
    if presets is None:
        presets = PRESET_SCREENS

    symbols: set[str] = set()
    for preset in presets:
        matching = [
            row for row in serialized_rows
            if _matches_preset_filters(row, preset["filters"])
        ]
        sort_field = preset.get("sort_by", "composite_score")
        reverse = preset.get("sort_order", "desc") == "desc"
        # Always sort None values last regardless of direction: a None row
        # should never "win" a slot over a real row.
        matching.sort(
            key=lambda r, f=sort_field, d=reverse: (
                r.get(f) is None,
                -(r.get(f) or 0) if d else (r.get(f) or 0),
            ),
        )
        for row in matching[:top_n]:
            sym = row.get("symbol")
            if sym:
                symbols.add(sym)

    return symbols
