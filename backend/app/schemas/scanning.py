"""Pydantic schemas for bulk scan API endpoints.

Contains request/response models for scan creation, status polling,
paginated results, filter options, and score explanations.
"""

from datetime import datetime
from typing import Any, List, Optional, Self

from pydantic import BaseModel, Field

from ..domain.scanning.models import ScanResultItemDomain, StockExplanation
from .universe import UniverseDefinition


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ScanCreateRequest(BaseModel):
    """Request model for creating a new scan."""

    universe: str = Field(
        default="all",
        description="Legacy universe selector. Accepts: all, test, custom, nyse, nasdaq, amex, sp500. "
        "Prefer universe_def for new integrations.",
    )
    symbols: Optional[List[str]] = Field(
        default=None, description="Custom symbol list (if universe=custom/test)"
    )
    universe_def: Optional[UniverseDefinition] = Field(
        default=None,
        description="Structured universe definition. Takes precedence over legacy universe field.",
    )
    criteria: Optional[dict] = Field(default=None, description="Scan criteria")

    # Multi-screener fields
    screeners: List[str] = Field(
        default=["minervini"],
        description="Screeners to run: minervini, canslim, ipo, custom, volume_breakthrough, setup_engine",
    )
    composite_method: str = Field(
        default="weighted_average",
        description="How to combine scores: weighted_average, maximum, minimum",
    )

    # Idempotency
    idempotency_key: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional idempotency key. Repeated POSTs with the same key return the existing scan.",
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ScanCreateResponse(BaseModel):
    """Response model for scan creation."""

    scan_id: str
    status: str
    total_stocks: int
    message: str
    feature_run_id: Optional[int] = None


class ScanStatusResponse(BaseModel):
    """Response model for scan status."""

    scan_id: str
    status: str
    progress: float
    total_stocks: int
    completed_stocks: int
    passed_stocks: int
    started_at: datetime
    eta_seconds: Optional[int] = None


class ScanResultItem(BaseModel):
    """Individual scan result item."""

    symbol: str
    company_name: Optional[str] = None
    composite_score: float
    rating: str

    # Individual screener scores
    minervini_score: Optional[float] = None
    canslim_score: Optional[float] = None
    ipo_score: Optional[float] = None
    custom_score: Optional[float] = None
    volume_breakthrough_score: Optional[float] = None

    # Setup Engine fields
    se_setup_score: Optional[float] = None
    se_pattern_primary: Optional[str] = None
    se_distance_to_pivot_pct: Optional[float] = None
    se_in_early_zone: Optional[bool] = None
    se_extended_from_pivot: Optional[bool] = None
    se_base_length_weeks: Optional[float] = None
    se_base_depth_pct: Optional[float] = None
    se_support_tests_count: Optional[float] = None
    se_tight_closes_count: Optional[float] = None
    se_bb_width_pctile_252: Optional[float] = None
    se_bb_squeeze: Optional[bool] = None
    se_volume_vs_50d: Optional[float] = None
    se_up_down_volume_ratio_10d: Optional[float] = None
    se_quiet_days_10d: Optional[float] = None
    se_rs_line_new_high: Optional[bool] = None
    se_pivot_price: Optional[float] = None
    se_setup_ready: Optional[bool] = None
    se_quality_score: Optional[float] = None
    se_readiness_score: Optional[float] = None
    se_pattern_confidence: Optional[float] = None
    se_pivot_type: Optional[str] = None
    se_pivot_date: Optional[str] = None
    se_timeframe: Optional[str] = None
    se_atr14_pct: Optional[float] = None
    se_explain: Optional[dict] = None
    se_candidates: Optional[list] = None

    # Minervini fields
    rs_rating: Optional[float] = None
    rs_rating_1m: Optional[float] = None
    rs_rating_3m: Optional[float] = None
    rs_rating_12m: Optional[float] = None
    stage: Optional[int] = None
    stage_name: Optional[str] = None
    current_price: Optional[float] = None
    volume: Optional[float] = None
    market_cap: Optional[float] = None
    ma_alignment: Optional[bool] = None
    vcp_detected: Optional[bool] = None
    vcp_score: Optional[float] = None
    vcp_pivot: Optional[float] = None
    vcp_ready_for_breakout: Optional[bool] = None
    vcp_contraction_ratio: Optional[float] = None
    vcp_atr_score: Optional[float] = None
    passes_template: Optional[bool] = None

    # Growth fields
    adr_percent: Optional[float] = None
    eps_growth_qq: Optional[float] = None
    sales_growth_qq: Optional[float] = None
    eps_growth_yy: Optional[float] = None
    sales_growth_yy: Optional[float] = None

    # Valuation fields
    peg_ratio: Optional[float] = None

    # EPS Rating (IBD-style 0-99 percentile)
    eps_rating: Optional[int] = None

    # Industry classifications
    ibd_industry_group: Optional[str] = None
    ibd_group_rank: Optional[int] = None
    gics_sector: Optional[str] = None
    gics_industry: Optional[str] = None

    # RS Sparkline data (30-day stock/SPY ratio trend)
    rs_sparkline_data: Optional[List[float]] = None
    rs_trend: Optional[int] = None  # -1=declining, 0=flat, 1=improving

    # Price Sparkline data (30-day normalized price trend)
    price_sparkline_data: Optional[List[float]] = None
    price_change_1d: Optional[float] = None
    price_trend: Optional[int] = None  # -1=down, 0=flat, 1=up overall

    # IPO date for age filtering
    ipo_date: Optional[str] = None  # Format: YYYY-MM-DD

    # Beta and Beta-Adjusted RS metrics
    beta: Optional[float] = None
    beta_adj_rs: Optional[float] = None
    beta_adj_rs_1m: Optional[float] = None
    beta_adj_rs_3m: Optional[float] = None
    beta_adj_rs_12m: Optional[float] = None

    # Multi-screener metadata
    screeners_run: Optional[List[str]] = None

    @classmethod
    def from_domain(
        cls,
        item: ScanResultItemDomain,
        *,
        include_setup_payload: bool = True,
    ) -> Self:
        """Map a domain scan result to the HTTP response model.

        This is the canonical domain-to-HTTP mapper.  All field unpacking
        from ``extended_fields`` happens here so that endpoint handlers
        stay thin and don't duplicate the 40+ field mapping.
        """
        ef = item.extended_fields
        return cls(
            symbol=item.symbol,
            company_name=ef.get("company_name"),
            composite_score=item.composite_score,
            rating=item.rating,
            # Individual screener scores
            minervini_score=ef.get("minervini_score"),
            canslim_score=ef.get("canslim_score"),
            ipo_score=ef.get("ipo_score"),
            custom_score=ef.get("custom_score"),
            volume_breakthrough_score=ef.get("volume_breakthrough_score"),
            # Setup Engine fields
            se_setup_score=ef.get("se_setup_score"),
            se_pattern_primary=ef.get("se_pattern_primary"),
            se_distance_to_pivot_pct=ef.get("se_distance_to_pivot_pct"),
            se_in_early_zone=ef.get("se_in_early_zone"),
            se_extended_from_pivot=ef.get("se_extended_from_pivot"),
            se_base_length_weeks=ef.get("se_base_length_weeks"),
            se_base_depth_pct=ef.get("se_base_depth_pct"),
            se_support_tests_count=ef.get("se_support_tests_count"),
            se_tight_closes_count=ef.get("se_tight_closes_count"),
            se_bb_width_pctile_252=ef.get("se_bb_width_pctile_252"),
            se_bb_squeeze=ef.get("se_bb_squeeze"),
            se_volume_vs_50d=ef.get("se_volume_vs_50d"),
            se_up_down_volume_ratio_10d=ef.get("se_up_down_volume_ratio_10d"),
            se_quiet_days_10d=ef.get("se_quiet_days_10d"),
            se_rs_line_new_high=ef.get("se_rs_line_new_high"),
            se_pivot_price=ef.get("se_pivot_price"),
            se_setup_ready=ef.get("se_setup_ready"),
            se_quality_score=ef.get("se_quality_score"),
            se_readiness_score=ef.get("se_readiness_score"),
            se_pattern_confidence=ef.get("se_pattern_confidence"),
            se_pivot_type=ef.get("se_pivot_type"),
            se_pivot_date=ef.get("se_pivot_date"),
            se_timeframe=ef.get("se_timeframe"),
            se_atr14_pct=ef.get("se_atr14_pct"),
            se_explain=ef.get("se_explain") if include_setup_payload else None,
            se_candidates=ef.get("se_candidates") if include_setup_payload else None,
            # Minervini fields
            rs_rating=ef.get("rs_rating"),
            rs_rating_1m=ef.get("rs_rating_1m"),
            rs_rating_3m=ef.get("rs_rating_3m"),
            rs_rating_12m=ef.get("rs_rating_12m"),
            stage=ef.get("stage"),
            stage_name=ef.get("stage_name"),
            current_price=item.current_price,
            volume=ef.get("volume"),
            market_cap=ef.get("market_cap"),
            ma_alignment=ef.get("ma_alignment"),
            vcp_detected=ef.get("vcp_detected"),
            vcp_score=ef.get("vcp_score"),
            vcp_pivot=ef.get("vcp_pivot"),
            vcp_ready_for_breakout=ef.get("vcp_ready_for_breakout"),
            vcp_contraction_ratio=ef.get("vcp_contraction_ratio"),
            vcp_atr_score=ef.get("vcp_atr_score"),
            passes_template=ef.get("passes_template"),
            # Growth fields
            adr_percent=ef.get("adr_percent"),
            eps_growth_qq=ef.get("eps_growth_qq"),
            sales_growth_qq=ef.get("sales_growth_qq"),
            eps_growth_yy=ef.get("eps_growth_yy"),
            sales_growth_yy=ef.get("sales_growth_yy"),
            # Valuation
            peg_ratio=ef.get("peg_ratio"),
            # EPS Rating
            eps_rating=ef.get("eps_rating"),
            # Industry classifications
            ibd_industry_group=ef.get("ibd_industry_group"),
            ibd_group_rank=ef.get("ibd_group_rank"),
            gics_sector=ef.get("gics_sector"),
            gics_industry=ef.get("gics_industry"),
            # Sparklines
            rs_sparkline_data=ef.get("rs_sparkline_data"),
            rs_trend=ef.get("rs_trend"),
            price_sparkline_data=ef.get("price_sparkline_data"),
            price_change_1d=ef.get("price_change_1d"),
            price_trend=ef.get("price_trend"),
            # IPO date
            ipo_date=ef.get("ipo_date"),
            # Beta and Beta-Adjusted RS
            beta=ef.get("beta"),
            beta_adj_rs=ef.get("beta_adj_rs"),
            beta_adj_rs_1m=ef.get("beta_adj_rs_1m"),
            beta_adj_rs_3m=ef.get("beta_adj_rs_3m"),
            beta_adj_rs_12m=ef.get("beta_adj_rs_12m"),
            # Multi-screener metadata
            screeners_run=item.screeners_run,
        )


class ScanResultsResponse(BaseModel):
    """Response model for paginated scan results."""

    scan_id: str
    total: int
    page: int
    per_page: int
    pages: int
    results: List[ScanResultItem]


class ScanSymbolsResponse(BaseModel):
    """Response model for lightweight filtered symbol lists."""

    scan_id: str
    total: int
    symbols: List[str]
    page: Optional[int] = None
    per_page: Optional[int] = None
    next_cursor: Optional[str] = None


class SetupDetailsResponse(BaseModel):
    """Response model for setup-engine explain drawer payload."""

    scan_id: str
    symbol: str
    se_explain: Optional[dict[str, Any]] = None
    se_candidates: Optional[list[Any]] = None


class ScanListItem(BaseModel):
    """Individual scan in the list."""

    scan_id: str
    status: str
    universe: str  # Legacy label (backward compat)
    universe_type: Optional[str] = None
    universe_exchange: Optional[str] = None
    universe_index: Optional[str] = None
    universe_symbols_count: Optional[int] = None
    total_stocks: int
    passed_stocks: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    source: Optional[str] = None


class ScanListResponse(BaseModel):
    """Response model for list of scans."""

    scans: List[ScanListItem]


class FilterOptionsResponse(BaseModel):
    """Response model for filter options."""

    ibd_industries: List[str]
    gics_sectors: List[str]
    ratings: List[str]


# ---------------------------------------------------------------------------
# Explain endpoint schemas
# ---------------------------------------------------------------------------


class CriterionResultResponse(BaseModel):
    """One criterion's contribution within a screener."""

    name: str
    score: float
    max_score: float
    passed: bool


class ScreenerExplanationResponse(BaseModel):
    """Full explanation for one screener's evaluation."""

    screener_name: str
    score: float
    passes: bool
    rating: str
    criteria: List[CriterionResultResponse]


class ExplainResponse(BaseModel):
    """Complete explanation of a stock's composite score."""

    symbol: str
    composite_score: float
    rating: str
    composite_method: str
    screeners_passed: int
    screeners_total: int
    screener_explanations: List[ScreenerExplanationResponse]
    rating_thresholds: dict

    @classmethod
    def from_domain(cls, explanation: StockExplanation) -> Self:
        """Convert domain StockExplanation to Pydantic response."""
        return cls(
            symbol=explanation.symbol,
            composite_score=explanation.composite_score,
            rating=explanation.rating,
            composite_method=explanation.composite_method,
            screeners_passed=explanation.screeners_passed,
            screeners_total=explanation.screeners_total,
            screener_explanations=[
                ScreenerExplanationResponse(
                    screener_name=se.screener_name,
                    score=se.score,
                    passes=se.passes,
                    rating=se.rating,
                    criteria=[
                        CriterionResultResponse(
                            name=c.name,
                            score=c.score,
                            max_score=c.max_score,
                            passed=c.passed,
                        )
                        for c in se.criteria
                    ],
                )
                for se in explanation.screener_explanations
            ],
            rating_thresholds=explanation.rating_thresholds,
        )
