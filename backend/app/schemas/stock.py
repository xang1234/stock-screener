"""Stock data schemas"""
from pydantic import BaseModel
from typing import Any, List, Optional
from datetime import datetime

from .scanning import ScanResultItem, ScreenerExplanationResponse


class StockInfo(BaseModel):
    """Basic stock information"""
    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    current_price: Optional[float] = None
    market_cap: Optional[int] = None


class StockFundamentals(BaseModel):
    """Stock fundamental data"""
    symbol: str
    market_cap: Optional[int] = None
    pe_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    eps_current: Optional[float] = None
    eps_growth_quarterly: Optional[float] = None
    eps_growth_annual: Optional[float] = None
    revenue_current: Optional[int] = None
    revenue_growth: Optional[float] = None
    profit_margin: Optional[float] = None
    institutional_ownership: Optional[float] = None
    description: Optional[str] = None


class StockTechnicals(BaseModel):
    """Stock technical indicators"""
    symbol: str
    current_price: Optional[float] = None
    ma_50: Optional[float] = None
    ma_150: Optional[float] = None
    ma_200: Optional[float] = None
    rs_rating: Optional[float] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    stage: Optional[int] = None
    vcp_score: Optional[float] = None


class StockData(BaseModel):
    """Complete stock data"""
    info: StockInfo
    fundamentals: Optional[StockFundamentals] = None
    technicals: Optional[StockTechnicals] = None


class StockSearchResult(BaseModel):
    """Global symbol search result."""

    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None


class StockPriceHistoryPoint(BaseModel):
    """Single OHLCV bar."""

    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class StockChartSnapshot(BaseModel):
    """Chart payload for the decision workspace."""

    price_history: List[StockPriceHistoryPoint]
    chart_data: dict[str, Any]


class StockDecisionFactor(BaseModel):
    """Deterministic strength/weakness factor."""

    screener_name: str
    criterion_name: str
    score: float
    max_score: float
    passed: bool


class StockDecisionFreshness(BaseModel):
    """Freshness metadata for the workspace."""

    feature_run_id: Optional[int] = None
    feature_as_of_date: Optional[str] = None
    feature_completed_at: Optional[str] = None
    breadth_date: Optional[str] = None
    has_price_history: bool = False


class StockDecisionSummary(BaseModel):
    """Compact deterministic decision summary."""

    composite_score: Optional[float] = None
    rating: Optional[str] = None
    screeners_passed: int = 0
    screeners_total: int = 0
    composite_method: Optional[str] = None
    top_strengths: List[StockDecisionFactor]
    top_weaknesses: List[StockDecisionFactor]
    freshness: StockDecisionFreshness


class StockThemeSummary(BaseModel):
    """Theme linkage for a stock."""

    theme_id: int
    display_name: str
    pipeline: Optional[str] = None
    category: Optional[str] = None
    lifecycle_state: Optional[str] = None
    is_emerging: bool = False
    confidence: Optional[float] = None
    mention_count: Optional[int] = None
    correlation_to_theme: Optional[float] = None
    momentum_score: Optional[float] = None
    mention_velocity: Optional[float] = None
    basket_return_1m: Optional[float] = None
    status: Optional[str] = None


class StockRegimeSummary(BaseModel):
    """Simple deterministic market-regime summary."""

    label: str
    summary: str
    breadth_date: Optional[str] = None
    up_4pct: Optional[int] = None
    down_4pct: Optional[int] = None
    ratio_5day: Optional[float] = None
    ratio_10day: Optional[float] = None
    total_stocks_scanned: Optional[int] = None
    feature_run_stale: bool = False


class StockEventRiskSummary(BaseModel):
    """Upcoming earnings and ownership-derived event context."""

    next_earnings_date: Optional[str] = None
    days_until_earnings: Optional[int] = None
    earnings_window_risk: str = "safe"
    recent_earnings_count: int = 0
    beat_count_last_4: int = 0
    miss_count_last_4: int = 0
    avg_post_earnings_gap_pct: Optional[float] = None
    avg_post_earnings_5s_return_pct: Optional[float] = None
    institutional_ownership_current: Optional[float] = None
    institutional_ownership_delta_90d: Optional[float] = None
    notes: List[str]


class StockRegimeActions(BaseModel):
    """Profile-aware action guidance based on regime and event context."""

    stance: str
    sizing_guidance: str
    avoid_new_entries: bool = False
    preferred_setups: List[str]
    caution_flags: List[str]
    summary: str


class StockDecisionDashboardResponse(BaseModel):
    """Full stock decision workspace payload."""

    symbol: str
    as_of_date: Optional[str] = None
    freshness: StockDecisionFreshness
    info: Optional[StockInfo] = None
    fundamentals: Optional[StockFundamentals] = None
    technicals: Optional[StockTechnicals] = None
    chart: StockChartSnapshot
    decision_summary: StockDecisionSummary
    screener_explanations: List[ScreenerExplanationResponse]
    peers: List[ScanResultItem]
    themes: List[StockThemeSummary]
    regime: StockRegimeSummary
    event_risk: StockEventRiskSummary
    regime_actions: StockRegimeActions
    degraded_reasons: List[str]
