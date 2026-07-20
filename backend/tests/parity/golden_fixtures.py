"""Golden fixture factory for parity tests.

Generates deterministic orchestrator output dicts with named profiles that
exercise specific edge cases in the legacy and feature-store write/read paths.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Ticker profiles — each tests a specific condition
# ---------------------------------------------------------------------------

GOLDEN_PROFILES: dict[str, dict[str, Any]] = {
    # High score, multi-screener, Strong Buy
    "AAPL": {"profile": "high_score_multi", "idx": 0},
    "NVDA": {"profile": "high_score_multi", "idx": 1},
    # Mid score, Buy rating, Stage 2
    "MSFT": {"profile": "mid_score", "idx": 0},
    "AMZN": {"profile": "mid_score", "idx": 1},
    "META": {"profile": "mid_score", "idx": 2},
    # Watch boundary (~60.5)
    "GOOGL": {"profile": "watch_boundary", "idx": 0},
    "NFLX": {"profile": "watch_boundary", "idx": 1},
    # Pass rating (~45)
    "INTC": {"profile": "pass_score", "idx": 0},
    "PFE": {"profile": "pass_score", "idx": 1},
    # Boundary: 79.999 — just below Strong Buy threshold
    "AVGO": {"profile": "boundary_strong_buy", "idx": 0},
    # Floor score: 0.0
    "GE": {"profile": "floor_score", "idx": 0},
    # Ceiling score: 100.0
    "TSLA": {"profile": "ceiling_score", "idx": 0},
    # None-heavy — most optional fields None
    "PLTR": {"profile": "none_heavy", "idx": 0},
    "COIN": {"profile": "none_heavy", "idx": 1},
    # All 5 screener scores populated
    "AMD": {"profile": "all_screeners", "idx": 0},
    # Single screener only
    "JPM": {"profile": "single_screener", "idx": 0},
    "V": {"profile": "single_screener", "idx": 1},
    "WMT": {"profile": "single_screener", "idx": 2},
    # Has sparklines + trends
    "UNH": {"profile": "has_sparklines", "idx": 0},
    "MA": {"profile": "has_sparklines", "idx": 1},
    # Empty sparklines edge case
    "CRM": {"profile": "empty_sparklines", "idx": 0},
}

GOLDEN_TICKERS: list[str] = list(GOLDEN_PROFILES.keys())

CANONICAL_STOCK_RS_FIELDS: tuple[str, ...] = (
    "rs_rating",
    "rs_rating_1m",
    "rs_rating_3m",
    "rs_rating_12m",
    "rs_formula_version",
    "market_rs_run_id",
    "rs_universe_size",
)

CANONICAL_GROUP_RS_FIELDS: tuple[str, ...] = (
    "rank",
    "avg_rs_rating",
    "avg_rs_rating_1m",
    "avg_rs_rating_3m",
    "num_stocks",
    "top_symbol",
    "rs_formula_version",
    "market_rs_run_id",
)


def _base_dict(symbol: str) -> dict[str, Any]:
    """Common fields shared by all profiles, ensuring no key is missed."""
    return {
        # Core
        "composite_score": 70.0,
        "rating": "Buy",
        "current_price": 150.0 + hash(symbol) % 500,
        # Screener scores
        "minervini_score": 65.0,
        "canslim_score": None,
        "ipo_score": None,
        "custom_score": None,
        "volume_breakthrough_score": None,
        # RS ratings
        "rs_rating": 75.0,
        "rs_rating_1m": 70.0,
        "rs_rating_3m": 72.0,
        "rs_rating_12m": 68.0,
        # Stage
        "stage": 2,
        "stage_name": "Stage 2 - Uptrend",
        # Volume / market
        "avg_dollar_volume": 50_000_000,
        "market_cap": 100_000_000_000,
        # Technical
        "ma_alignment": True,
        "vcp_detected": False,
        "vcp_score": None,
        "vcp_pivot": None,
        "vcp_ready_for_breakout": None,
        "vcp_contraction_ratio": None,
        "vcp_atr_score": None,
        "passes_template": True,
        "adr_percent": 2.0,
        # Fundamentals
        "eps_growth_qq": 25.0,
        "sales_growth_qq": 18.0,
        "eps_growth_yy": 30.0,
        "sales_growth_yy": 22.0,
        "peg_ratio": 1.5,
        "eps_rating": 85,
        # Industry
        "ibd_industry_group": "Comp-Software",
        "ibd_group_rank": 15,
        "gics_sector": "Technology",
        "gics_industry": "Software",
        # Sparklines
        "rs_sparkline_data": [70, 72, 74, 76, 78],
        "rs_trend": 1,
        "price_sparkline_data": [145, 148, 150, 152, 155],
        "price_change_1d": 1.2,
        "price_trend": 1,
        # IPO / Beta
        "ipo_date": "2010-06-29",
        "beta": 1.1,
        "beta_adj_rs": 72.0,
        "beta_adj_rs_1m": 68.0,
        "beta_adj_rs_3m": 70.0,
        "beta_adj_rs_12m": 66.0,
        # Performance
        "perf_week": 2.5,
        "perf_month": 5.0,
        "perf_3m": 12.0,
        "perf_6m": 20.0,
        # Episodic pivot
        "gap_percent": 0.5,
        "volume_surge": 1.2,
        # EMA distances
        "ema_10_distance": 1.0,
        "ema_20_distance": 2.0,
        "ema_50_distance": 5.0,
        # 52-week distances (orchestrator keys, translated on write)
        "from_52w_high_pct": -8.0,
        "above_52w_low_pct": 45.0,
        # Multi-screener metadata
        "screeners_run": ["minervini"],
        "composite_method": "weighted_average",
        "screeners_passed": 1,
        "screeners_total": 1,
    }


def build_golden_result(symbol: str, profile: str, idx: int) -> dict[str, Any]:
    """Build a deterministic orchestrator output dict for the given profile."""
    d = _base_dict(symbol)

    if profile == "high_score_multi":
        d.update(
            composite_score=91.5 + idx * 1.0,
            rating="Strong Buy",
            minervini_score=90.0 + idx,
            canslim_score=88.0 + idx,
            rs_rating=94.0 + idx,
            stage=2,
            stage_name="Stage 2 - Uptrend",
            passes_template=True,
            screeners_run=["minervini", "canslim"],
            composite_method="weighted_average",
            screeners_passed=2,
            screeners_total=2,
            gics_sector="Technology",
            gics_industry="Semiconductors" if idx == 1 else "Consumer Electronics",
            ibd_industry_group="Comp-Peripherals" if idx == 0 else "Elec-Semiconductor",
            ibd_group_rank=5 + idx,
        )

    elif profile == "mid_score":
        d.update(
            composite_score=74.0 + idx * 1.5,
            rating="Buy",
            minervini_score=72.0 + idx,
            canslim_score=68.0 + idx,
            rs_rating=78.0 + idx,
            stage=2,
            passes_template=True,
            screeners_run=["minervini", "canslim"],
            screeners_passed=1,
            screeners_total=2,
            gics_sector="Technology",
            gics_industry=["Software", "E-Commerce", "Social Media"][idx],
            ibd_industry_group=["Comp-Software", "Internet-Retail", "Internet-Content"][idx],
            ibd_group_rank=10 + idx * 3,
        )

    elif profile == "watch_boundary":
        d.update(
            composite_score=60.5 + idx * 0.3,
            rating="Watch",
            minervini_score=58.0 + idx,
            canslim_score=None,
            rs_rating=62.0 + idx,
            stage=2,
            passes_template=False,
            screeners_run=["minervini"],
            screeners_passed=0,
            screeners_total=1,
            gics_sector="Communication Services",
            gics_industry=["Interactive Media", "Entertainment"][idx],
            ibd_industry_group=["Internet-Content", "Media-Entertainment"][idx],
            ibd_group_rank=30 + idx * 5,
        )

    elif profile == "pass_score":
        d.update(
            composite_score=44.0 + idx * 2.0,
            rating="Pass",
            minervini_score=42.0 + idx,
            canslim_score=None,
            rs_rating=40.0 + idx * 3,
            stage=3,
            stage_name="Stage 3 - Top",
            passes_template=False,
            screeners_run=["minervini"],
            screeners_passed=0,
            screeners_total=1,
            gics_sector="Healthcare" if idx == 1 else "Technology",
            gics_industry="Pharmaceuticals" if idx == 1 else "Semiconductors",
            ibd_industry_group="Medical-Pharma" if idx == 1 else "Elec-Semiconductor",
            ibd_group_rank=50 + idx * 10,
        )

    elif profile == "boundary_strong_buy":
        d.update(
            composite_score=79.999,
            rating="Buy",
            minervini_score=78.0,
            canslim_score=76.0,
            rs_rating=82.0,
            stage=2,
            passes_template=True,
            screeners_run=["minervini", "canslim"],
            screeners_passed=2,
            screeners_total=2,
            gics_sector="Technology",
            gics_industry="Semiconductors",
            ibd_industry_group="Elec-Semiconductor",
            ibd_group_rank=3,
        )

    elif profile == "floor_score":
        d.update(
            composite_score=0.0,
            rating="Pass",
            minervini_score=0.0,
            canslim_score=None,
            rs_rating=10.0,
            stage=4,
            stage_name="Stage 4 - Downtrend",
            passes_template=False,
            eps_growth_qq=-15.0,
            sales_growth_qq=-8.0,
            screeners_run=["minervini"],
            screeners_passed=0,
            screeners_total=1,
            gics_sector="Industrials",
            gics_industry="Aerospace & Defense",
            ibd_industry_group="Aero-Defense",
            ibd_group_rank=80,
        )

    elif profile == "ceiling_score":
        d.update(
            composite_score=100.0,
            rating="Strong Buy",
            minervini_score=100.0,
            canslim_score=100.0,
            rs_rating=99.0,
            stage=2,
            passes_template=True,
            screeners_run=["minervini", "canslim"],
            screeners_passed=2,
            screeners_total=2,
            gics_sector="Consumer Discretionary",
            gics_industry="Automobiles",
            ibd_industry_group="Auto-Cars",
            ibd_group_rank=1,
        )

    elif profile == "none_heavy":
        # Most optional fields None — simulates newly-IPO'd or thin data
        d.update(
            composite_score=55.0 + idx * 2.0,
            rating="Watch",
            minervini_score=50.0 + idx,
            canslim_score=None,
            ipo_score=None,
            custom_score=None,
            volume_breakthrough_score=None,
            rs_rating=55.0 + idx,
            rs_rating_1m=None,
            rs_rating_3m=None,
            rs_rating_12m=None,
            stage=None,
            stage_name=None,
            ma_alignment=None,
            vcp_detected=None,
            vcp_score=None,
            vcp_pivot=None,
            vcp_ready_for_breakout=None,
            vcp_contraction_ratio=None,
            vcp_atr_score=None,
            passes_template=False,
            adr_percent=None,
            eps_growth_qq=None,
            sales_growth_qq=None,
            eps_growth_yy=None,
            sales_growth_yy=None,
            peg_ratio=None,
            eps_rating=None,
            ibd_industry_group=None,
            ibd_group_rank=None,
            gics_sector=None,
            gics_industry=None,
            rs_sparkline_data=None,
            rs_trend=None,
            price_sparkline_data=None,
            price_change_1d=None,
            price_trend=None,
            ipo_date=None,
            beta=None,
            beta_adj_rs=None,
            beta_adj_rs_1m=None,
            beta_adj_rs_3m=None,
            beta_adj_rs_12m=None,
            perf_week=None,
            perf_month=None,
            perf_3m=None,
            perf_6m=None,
            gap_percent=None,
            volume_surge=None,
            ema_10_distance=None,
            ema_20_distance=None,
            ema_50_distance=None,
            from_52w_high_pct=None,
            above_52w_low_pct=None,
            screeners_run=["minervini"],
            screeners_passed=0,
            screeners_total=1,
        )

    elif profile == "all_screeners":
        d.update(
            composite_score=85.0,
            rating="Strong Buy",
            minervini_score=88.0,
            canslim_score=82.0,
            ipo_score=75.0,
            custom_score=80.0,
            volume_breakthrough_score=90.0,
            rs_rating=91.0,
            stage=2,
            passes_template=True,
            vcp_detected=True,
            vcp_score=78.5,
            vcp_pivot=155.0,
            vcp_ready_for_breakout=True,
            vcp_contraction_ratio=0.65,
            vcp_atr_score=82.0,
            screeners_run=["minervini", "canslim", "ipo", "custom", "volume_breakthrough"],
            composite_method="weighted_average",
            screeners_passed=5,
            screeners_total=5,
            gics_sector="Technology",
            gics_industry="Semiconductors",
            ibd_industry_group="Elec-Semiconductor",
            ibd_group_rank=2,
        )

    elif profile == "single_screener":
        d.update(
            composite_score=68.0 + idx * 3.5,
            rating="Buy" if idx < 2 else "Watch",
            minervini_score=68.0 + idx * 3.5,
            canslim_score=None,
            ipo_score=None,
            custom_score=None,
            volume_breakthrough_score=None,
            rs_rating=70.0 + idx * 2.5,
            stage=2,
            passes_template=idx < 2,
            screeners_run=["minervini"],
            composite_method="weighted_average",
            screeners_passed=1 if idx < 2 else 0,
            screeners_total=1,
            gics_sector=["Financials", "Financials", "Consumer Staples"][idx],
            gics_industry=["Banks", "Payment Processing", "Retail"][idx],
            ibd_industry_group=["Banks-Major", "Finance-Card", "Retail-Major"][idx],
            ibd_group_rank=20 + idx * 5,
        )

    elif profile == "has_sparklines":
        d.update(
            composite_score=82.0 + idx * 1.5,
            rating="Strong Buy",
            minervini_score=80.0 + idx,
            canslim_score=78.0 + idx,
            rs_rating=88.0 + idx,
            rs_sparkline_data=[75, 78, 80, 83, 85, 87, 88 + idx],
            rs_trend=1,
            price_sparkline_data=[400, 410, 415, 420, 425, 430 + idx * 10],
            price_change_1d=0.8 + idx * 0.3,
            price_trend=1,
            screeners_run=["minervini", "canslim"],
            screeners_passed=2,
            screeners_total=2,
            gics_sector="Healthcare" if idx == 0 else "Financials",
            gics_industry="Managed Health Care" if idx == 0 else "Payment Processing",
            ibd_industry_group="Medical-HMO" if idx == 0 else "Finance-Card",
            ibd_group_rank=8 + idx * 4,
        )

    elif profile == "empty_sparklines":
        d.update(
            composite_score=72.0,
            rating="Buy",
            minervini_score=70.0,
            canslim_score=None,
            rs_rating=73.5,
            rs_sparkline_data=[],
            rs_trend=0,
            price_sparkline_data=[],
            price_change_1d=0.0,
            price_trend=0,
            screeners_run=["minervini"],
            screeners_passed=1,
            screeners_total=1,
            gics_sector="Technology",
            gics_industry="Application Software",
            ibd_industry_group="Comp-Software",
            ibd_group_rank=12,
        )

    return d


def build_all_golden_results() -> dict[str, dict[str, Any]]:
    """Build all 20 golden orchestrator output dicts keyed by symbol."""
    return {
        symbol: build_golden_result(symbol, meta["profile"], meta["idx"])
        for symbol, meta in GOLDEN_PROFILES.items()
    }
