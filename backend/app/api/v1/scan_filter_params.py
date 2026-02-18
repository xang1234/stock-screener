"""Reusable FastAPI dependencies for scan result filter/sort parsing.

Extracts the 40+ query parameters into shared Depends() functions
so that both get_scan_results and export_scan_results use the
same FilterSpec + SortSpec construction.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from dateutil.relativedelta import relativedelta
from fastapi import Query

from app.domain.scanning.filter_spec import (
    FilterMode,
    FilterSpec,
    PageSpec,
    SortOrder,
    SortSpec,
)


# ---------------------------------------------------------------------------
# IPO date preset parsing
# ---------------------------------------------------------------------------


def parse_ipo_after_preset(preset: str) -> Optional[str]:
    """Parse IPO date preset to a date string (YYYY-MM-DD).

    Presets: 6m, 1y, 2y, 3y, 5y or explicit YYYY-MM-DD.
    """
    if not preset:
        return None

    preset = preset.strip().lower()
    today = datetime.now().date()

    preset_map = {
        "6m": relativedelta(months=6),
        "1y": relativedelta(years=1),
        "2y": relativedelta(years=2),
        "3y": relativedelta(years=3),
        "5y": relativedelta(years=5),
    }

    delta = preset_map.get(preset)
    if delta:
        cutoff = today - delta
    else:
        try:
            cutoff = datetime.strptime(preset, "%Y-%m-%d").date()
        except ValueError:
            return None

    return cutoff.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Filter parsing dependency
# ---------------------------------------------------------------------------


def parse_scan_filters(
    # Text search
    symbol_search: Optional[str] = Query(None, description="Symbol search pattern"),
    # Minervini score range
    min_score: Optional[float] = Query(None, description="Minimum Minervini score"),
    max_score: Optional[float] = Query(None, description="Maximum Minervini score"),
    # Stage
    stage: Optional[int] = Query(None, description="Stage filter (1-4)"),
    # Categorical
    ratings: Optional[str] = Query(None, description="Rating filter (comma-separated)"),
    ibd_industries: Optional[str] = Query(None, description="IBD industry filter (comma-separated)"),
    ibd_industries_mode: Optional[str] = Query(None, description="IBD industry filter mode: include or exclude"),
    gics_sectors: Optional[str] = Query(None, description="GICS sector filter (comma-separated)"),
    gics_sectors_mode: Optional[str] = Query(None, description="GICS sector filter mode: include or exclude"),
    # Composite score
    min_composite: Optional[float] = Query(None, description="Minimum composite score"),
    max_composite: Optional[float] = Query(None, description="Maximum composite score"),
    # Individual screener scores
    min_canslim: Optional[float] = Query(None, description="Minimum CANSLIM score"),
    max_canslim: Optional[float] = Query(None, description="Maximum CANSLIM score"),
    min_ipo: Optional[float] = Query(None, description="Minimum IPO score"),
    max_ipo: Optional[float] = Query(None, description="Maximum IPO score"),
    min_custom: Optional[float] = Query(None, description="Minimum custom score"),
    max_custom: Optional[float] = Query(None, description="Maximum custom score"),
    min_vol_breakthrough: Optional[float] = Query(None, description="Minimum vol breakthrough score"),
    max_vol_breakthrough: Optional[float] = Query(None, description="Maximum vol breakthrough score"),
    # RS ratings
    min_rs: Optional[float] = Query(None, ge=0, le=100, description="Minimum RS Rating"),
    max_rs: Optional[float] = Query(None, ge=0, le=100, description="Maximum RS Rating"),
    min_rs_1m: Optional[float] = Query(None, ge=0, le=100, description="Minimum RS 1M"),
    max_rs_1m: Optional[float] = Query(None, ge=0, le=100, description="Maximum RS 1M"),
    min_rs_3m: Optional[float] = Query(None, ge=0, le=100, description="Minimum RS 3M"),
    max_rs_3m: Optional[float] = Query(None, ge=0, le=100, description="Maximum RS 3M"),
    min_rs_12m: Optional[float] = Query(None, ge=0, le=100, description="Minimum RS 12M"),
    max_rs_12m: Optional[float] = Query(None, ge=0, le=100, description="Maximum RS 12M"),
    # Price & Growth
    min_price: Optional[float] = Query(None, description="Minimum price"),
    max_price: Optional[float] = Query(None, description="Maximum price"),
    min_adr: Optional[float] = Query(None, description="Minimum ADR %"),
    max_adr: Optional[float] = Query(None, description="Maximum ADR %"),
    min_eps_growth: Optional[float] = Query(None, description="Minimum EPS growth Q/Q (%)"),
    max_eps_growth: Optional[float] = Query(None, description="Maximum EPS growth Q/Q (%)"),
    min_sales_growth: Optional[float] = Query(None, description="Minimum sales growth Q/Q (%)"),
    max_sales_growth: Optional[float] = Query(None, description="Maximum sales growth Q/Q (%)"),
    min_eps_growth_yy: Optional[float] = Query(None, description="Minimum EPS growth Y/Y (%)"),
    min_sales_growth_yy: Optional[float] = Query(None, description="Minimum sales growth Y/Y (%)"),
    max_peg: Optional[float] = Query(None, description="Maximum PEG ratio"),
    # EPS Rating
    min_eps_rating: Optional[int] = Query(None, ge=0, le=99, description="Minimum EPS Rating"),
    max_eps_rating: Optional[int] = Query(None, ge=0, le=99, description="Maximum EPS Rating"),
    # Volume & Market Cap
    min_volume: Optional[int] = Query(None, description="Minimum volume"),
    min_market_cap: Optional[int] = Query(None, description="Minimum market cap"),
    # VCP
    min_vcp_score: Optional[float] = Query(None, description="Minimum VCP score"),
    max_vcp_score: Optional[float] = Query(None, description="Maximum VCP score"),
    min_vcp_pivot: Optional[float] = Query(None, description="Minimum VCP pivot price"),
    max_vcp_pivot: Optional[float] = Query(None, description="Maximum VCP pivot price"),
    vcp_detected: Optional[bool] = Query(None, description="VCP detected filter"),
    vcp_ready: Optional[bool] = Query(None, description="VCP ready for breakout filter"),
    # Boolean
    ma_alignment: Optional[bool] = Query(None, description="MA alignment filter"),
    # Performance
    min_perf_day: Optional[float] = Query(None, description="Minimum 1-day % change"),
    max_perf_day: Optional[float] = Query(None, description="Maximum 1-day % change"),
    min_perf_week: Optional[float] = Query(None, description="Minimum 5-day % change"),
    max_perf_week: Optional[float] = Query(None, description="Maximum 5-day % change"),
    min_perf_month: Optional[float] = Query(None, description="Minimum 21-day % change"),
    max_perf_month: Optional[float] = Query(None, description="Maximum 21-day % change"),
    # EMA distances
    min_ema_10: Optional[float] = Query(None, description="Minimum % from EMA10"),
    max_ema_10: Optional[float] = Query(None, description="Maximum % from EMA10"),
    min_ema_20: Optional[float] = Query(None, description="Minimum % from EMA20"),
    max_ema_20: Optional[float] = Query(None, description="Maximum % from EMA20"),
    min_ema_50: Optional[float] = Query(None, description="Minimum % from EMA50"),
    max_ema_50: Optional[float] = Query(None, description="Maximum % from EMA50"),
    # 52-week distances
    min_52w_high: Optional[float] = Query(None, description="Minimum % below 52-week high"),
    max_52w_high: Optional[float] = Query(None, description="Maximum % below 52-week high"),
    min_52w_low: Optional[float] = Query(None, description="Minimum % above 52-week low"),
    max_52w_low: Optional[float] = Query(None, description="Maximum % above 52-week low"),
    # IPO date
    ipo_after: Optional[str] = Query(None, description="IPO after date (presets: 6m, 1y, 2y, 3y, 5y or YYYY-MM-DD)"),
    # Beta
    min_beta: Optional[float] = Query(None, description="Minimum Beta"),
    max_beta: Optional[float] = Query(None, description="Maximum Beta"),
    min_beta_adj_rs: Optional[float] = Query(None, ge=0, le=100, description="Minimum Beta-adjusted RS"),
    max_beta_adj_rs: Optional[float] = Query(None, ge=0, le=100, description="Maximum Beta-adjusted RS"),
    min_beta_adj_rs_1m: Optional[float] = Query(None, ge=0, le=100, description="Minimum Beta-adjusted RS 1M"),
    min_beta_adj_rs_3m: Optional[float] = Query(None, ge=0, le=100, description="Minimum Beta-adjusted RS 3M"),
    min_beta_adj_rs_12m: Optional[float] = Query(None, ge=0, le=100, description="Minimum Beta-adjusted RS 12M"),
    # Extended performance
    min_perf_3m: Optional[float] = Query(None, description="Minimum 3-month % change"),
    max_perf_3m: Optional[float] = Query(None, description="Maximum 3-month % change"),
    min_perf_6m: Optional[float] = Query(None, description="Minimum 6-month % change"),
    max_perf_6m: Optional[float] = Query(None, description="Maximum 6-month % change"),
    # Episodic Pivot
    min_gap_percent: Optional[float] = Query(None, description="Minimum gap up %"),
    max_gap_percent: Optional[float] = Query(None, description="Maximum gap up %"),
    min_volume_surge: Optional[float] = Query(None, description="Minimum volume surge ratio"),
    max_volume_surge: Optional[float] = Query(None, description="Maximum volume surge ratio"),
) -> FilterSpec:
    """Build a FilterSpec from HTTP query parameters."""
    f = FilterSpec()

    # Text search
    if symbol_search:
        f.add_text_search("symbol", symbol_search)

    # Range filters — using domain field names matching _COLUMN_MAP
    f.add_range("minervini_score", min_score, max_score)
    f.add_range("composite_score", min_composite, max_composite)
    f.add_range("canslim_score", min_canslim, max_canslim)
    f.add_range("ipo_score", min_ipo, max_ipo)
    f.add_range("custom_score", min_custom, max_custom)
    f.add_range("volume_breakthrough_score", min_vol_breakthrough, max_vol_breakthrough)
    f.add_range("rs_rating", min_rs, max_rs)
    f.add_range("rs_rating_1m", min_rs_1m, max_rs_1m)
    f.add_range("rs_rating_3m", min_rs_3m, max_rs_3m)
    f.add_range("rs_rating_12m", min_rs_12m, max_rs_12m)
    f.add_range("price", min_price, max_price)
    f.add_range("adr_percent", min_adr, max_adr)
    f.add_range("eps_growth_qq", min_eps_growth, max_eps_growth)
    f.add_range("sales_growth_qq", min_sales_growth, max_sales_growth)
    f.add_range("eps_growth_yy", min_eps_growth_yy, None)
    f.add_range("sales_growth_yy", min_sales_growth_yy, None)
    f.add_range("peg_ratio", None, max_peg)
    f.add_range("eps_rating", min_eps_rating, max_eps_rating)
    f.add_range("volume", min_volume, None)
    f.add_range("market_cap", min_market_cap, None)
    f.add_range("vcp_score", min_vcp_score, max_vcp_score)
    f.add_range("vcp_pivot", min_vcp_pivot, max_vcp_pivot)
    f.add_range("price_change_1d", min_perf_day, max_perf_day)
    f.add_range("perf_week", min_perf_week, max_perf_week)
    f.add_range("perf_month", min_perf_month, max_perf_month)
    f.add_range("ema_10_distance", min_ema_10, max_ema_10)
    f.add_range("ema_20_distance", min_ema_20, max_ema_20)
    f.add_range("ema_50_distance", min_ema_50, max_ema_50)
    f.add_range("week_52_high_distance", min_52w_high, max_52w_high)
    f.add_range("week_52_low_distance", min_52w_low, max_52w_low)
    f.add_range("beta", min_beta, max_beta)
    f.add_range("beta_adj_rs", min_beta_adj_rs, max_beta_adj_rs)
    f.add_range("beta_adj_rs_1m", min_beta_adj_rs_1m, None)
    f.add_range("beta_adj_rs_3m", min_beta_adj_rs_3m, None)
    f.add_range("beta_adj_rs_12m", min_beta_adj_rs_12m, None)
    f.add_range("perf_3m", min_perf_3m, max_perf_3m)
    f.add_range("perf_6m", min_perf_6m, max_perf_6m)
    f.add_range("gap_percent", min_gap_percent, max_gap_percent)
    f.add_range("volume_surge", min_volume_surge, max_volume_surge)

    # Stage (exact integer match via range)
    if stage is not None:
        f.add_range("stage", stage, stage)

    # Categorical filters
    if ratings:
        rating_list = tuple(r.strip() for r in ratings.split(","))
        f.add_categorical("rating", rating_list)

    if ibd_industries:
        industry_list = tuple(i.strip() for i in ibd_industries.split(","))
        mode = FilterMode.EXCLUDE if ibd_industries_mode == "exclude" else FilterMode.INCLUDE
        f.add_categorical("ibd_industry_group", industry_list, mode)

    if gics_sectors:
        sector_list = tuple(s.strip() for s in gics_sectors.split(","))
        mode = FilterMode.EXCLUDE if gics_sectors_mode == "exclude" else FilterMode.INCLUDE
        f.add_categorical("gics_sector", sector_list, mode)

    # Boolean filters
    if vcp_detected is not None:
        f.add_boolean("vcp_detected", vcp_detected)
    if vcp_ready is not None:
        f.add_boolean("vcp_ready_for_breakout", vcp_ready)
    if ma_alignment is not None:
        f.add_boolean("ma_alignment", ma_alignment)

    # IPO date (range on string column)
    if ipo_after:
        ipo_cutoff = parse_ipo_after_preset(ipo_after)
        if ipo_cutoff:
            f.add_range("ipo_date", ipo_cutoff, None)

    return f


# ---------------------------------------------------------------------------
# Sort parsing dependency
# ---------------------------------------------------------------------------


def parse_scan_sort(
    sort_by: str = Query("composite_score", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order: asc or desc"),
) -> SortSpec:
    """Build a SortSpec from HTTP query parameters."""
    order = SortOrder.ASC if sort_order.lower() == "asc" else SortOrder.DESC
    return SortSpec(field=sort_by, order=order)


# ---------------------------------------------------------------------------
# Pagination parsing dependency
# ---------------------------------------------------------------------------


def parse_page_spec(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=100, description="Results per page"),
) -> PageSpec:
    """Build a PageSpec from HTTP query parameters."""
    return PageSpec(page=page, per_page=per_page)
