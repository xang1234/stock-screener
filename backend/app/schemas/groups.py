"""Pydantic schemas for IBD Industry Group Rankings API endpoints"""
from pydantic import BaseModel, Field
from datetime import date as Date
from typing import Optional, List

from ..schemas.scope import ScopedResponseMixin


class GroupRankResponse(BaseModel):
    """Response model for a single group ranking"""

    industry_group: str = Field(..., description="IBD industry group name")
    date: str = Field(..., description="Ranking date (YYYY-MM-DD)")
    rank: int = Field(..., description="Current rank (1 = best)")
    avg_rs_rating: float = Field(..., description="Average RS rating of stocks in group")
    median_rs_rating: Optional[float] = Field(None, description="Median RS rating of stocks in group")
    weighted_avg_rs_rating: Optional[float] = Field(None, description="Market-cap weighted average RS rating")
    rs_std_dev: Optional[float] = Field(None, description="RS rating dispersion (std dev)")
    num_stocks: int = Field(..., description="Number of stocks with valid RS")
    num_stocks_rs_above_80: Optional[int] = Field(None, description="Stocks with RS > 80")
    pct_rs_above_80: Optional[float] = Field(None, description="Percent of stocks with RS > 80")

    # Top performer
    top_symbol: Optional[str] = Field(None, description="Best performing stock in group")
    top_rs_rating: Optional[float] = Field(None, description="RS rating of top stock")

    # Rank changes (positive = improved, negative = declined)
    rank_change_1w: Optional[int] = Field(None, description="Rank change vs 1 week ago")
    rank_change_1m: Optional[int] = Field(None, description="Rank change vs 1 month ago")
    rank_change_3m: Optional[int] = Field(None, description="Rank change vs 3 months ago")
    rank_change_6m: Optional[int] = Field(None, description="Rank change vs 6 months ago")

    class Config:
        from_attributes = True


class GroupRankingsResponse(ScopedResponseMixin):
    """Response model for list of group rankings"""

    date: str = Field(..., description="Date of rankings")
    total_groups: int = Field(..., description="Total number of ranked groups")
    rankings: List[GroupRankResponse] = Field(..., description="List of group rankings")


class HistoricalDataPoint(BaseModel):
    """Single historical data point for a group"""

    date: str = Field(..., description="Date (YYYY-MM-DD)")
    rank: int = Field(..., description="Rank on this date")
    avg_rs_rating: float = Field(..., description="Average RS rating on this date")
    num_stocks: Optional[int] = Field(None, description="Number of stocks")


class ConstituentStock(BaseModel):
    """Stock within an industry group with metrics"""

    symbol: str = Field(..., description="Stock ticker symbol")
    price: Optional[float] = Field(None, description="Current price")
    rs_rating: Optional[float] = Field(None, description="RS Rating (weighted)")
    rs_rating_1m: Optional[float] = Field(None, description="1-month RS Rating")
    rs_rating_3m: Optional[float] = Field(None, description="3-month RS Rating")
    rs_rating_12m: Optional[float] = Field(None, description="12-month RS Rating")
    eps_growth_qq: Optional[float] = Field(None, description="EPS growth quarter-over-quarter %")
    eps_growth_yy: Optional[float] = Field(None, description="EPS growth year-over-year %")
    sales_growth_qq: Optional[float] = Field(None, description="Sales growth quarter-over-quarter %")
    sales_growth_yy: Optional[float] = Field(None, description="Sales growth year-over-year %")
    composite_score: Optional[float] = Field(None, description="Composite screener score")
    stage: Optional[int] = Field(None, description="Weinstein stage (1-4)")
    price_sparkline_data: Optional[list] = Field(None, description="30-day normalized price trend")
    price_trend: Optional[int] = Field(None, description="Price trend: -1=down, 0=flat, 1=up")
    price_change_1d: Optional[float] = Field(None, description="1-day price change %")
    rs_sparkline_data: Optional[list] = Field(None, description="30-day RS ratio trend")
    rs_trend: Optional[int] = Field(None, description="RS trend: -1=declining, 0=flat, 1=improving")


class GroupDetailResponse(BaseModel):
    """Detailed response for a single industry group"""

    industry_group: str = Field(..., description="IBD industry group name")
    current_rank: int = Field(..., description="Current rank")
    current_avg_rs: float = Field(..., description="Current average RS rating")
    current_median_rs: Optional[float] = Field(None, description="Current median RS rating")
    current_weighted_avg_rs: Optional[float] = Field(None, description="Current market-cap weighted average RS rating")
    current_rs_std_dev: Optional[float] = Field(None, description="Current RS dispersion (std dev)")
    num_stocks: int = Field(..., description="Number of stocks in group")
    pct_rs_above_80: Optional[float] = Field(None, description="Percent of stocks with RS > 80")
    top_symbol: Optional[str] = Field(None, description="Best performing stock")
    top_rs_rating: Optional[float] = Field(None, description="RS of top stock")

    # Rank changes
    rank_change_1w: Optional[int] = Field(None, description="Rank change vs 1 week ago")
    rank_change_1m: Optional[int] = Field(None, description="Rank change vs 1 month ago")
    rank_change_3m: Optional[int] = Field(None, description="Rank change vs 3 months ago")
    rank_change_6m: Optional[int] = Field(None, description="Rank change vs 6 months ago")

    # Historical data
    history: List[HistoricalDataPoint] = Field(..., description="Historical rank data")

    # Constituent stocks with metrics
    stocks: List[ConstituentStock] = Field(default=[], description="Stocks in this group with metrics")


class MoversResponse(ScopedResponseMixin):
    """Response for rank movers (gainers and losers)"""

    period: str = Field(..., description="Time period (1w, 1m, 3m, 6m)")
    gainers: List[GroupRankResponse] = Field(..., description="Groups with biggest rank improvements")
    losers: List[GroupRankResponse] = Field(..., description="Groups with biggest rank declines")


class CalculationRequest(BaseModel):
    """Request model for manual ranking calculation"""

    calculation_date: Optional[str] = Field(
        None,
        description="Date to calculate for (YYYY-MM-DD), defaults to today"
    )


class CalculationResponse(BaseModel):
    """Response model for completed calculation"""

    status: str = Field(..., description="Status of the calculation request")
    message: str = Field(..., description="Human-readable message")
    groups_ranked: Optional[int] = Field(None, description="Number of groups ranked")
    date: Optional[str] = Field(None, description="Date calculated")


class CalculationStatusResponse(BaseModel):
    """Response model for calculation task status polling"""

    task_id: str = Field(..., description="Celery task ID")
    status: str = Field(..., description="Task status: queued, running, completed, failed")
    result: Optional[CalculationResponse] = Field(None, description="Result when completed")
    error: Optional[str] = Field(None, description="Error message when failed")


class BackfillRequest(BaseModel):
    """Request model for historical backfill"""

    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")


class BackfillResponse(BaseModel):
    """Response model for backfill task"""

    status: str = Field(..., description="Status of the backfill request")
    message: str = Field(..., description="Human-readable message")
    start_date: str = Field(..., description="Start date of backfill")
    end_date: str = Field(..., description="End date of backfill")
    total_dates: int = Field(..., description="Total trading days to process")
    processed: int = Field(..., description="Days processed")
    skipped: int = Field(..., description="Days skipped (already calculated)")
    errors: int = Field(..., description="Days with errors")
