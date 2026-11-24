"""Pydantic schemas for market breadth API endpoints"""
from pydantic import BaseModel, ConfigDict, Field
from datetime import date as Date
from typing import Optional, List, Dict, Any, Literal


class BreadthContributorItem(BaseModel):
    symbol: str
    company_name: str | None = None
    ibd_industry_group: str
    daily_change_pct: float | None = None
    signals: Dict[str, float]


class BreadthContributorIndexResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_id: Literal["breadth-contributors-v1"] = Field(alias="schema")
    market: str
    calculation_revision: Literal[3]
    dates: List[Date]


class BreadthContributorDocumentResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_id: Literal["breadth-contributors-v1"] = Field(alias="schema")
    market: str
    date: Date
    calculation_revision: Literal[3]
    contributors: List[BreadthContributorItem]


class BreadthResponse(BaseModel):
    """Response model for market breadth data"""

    market: str = Field("US", description="Market code for this breadth snapshot")
    date: Date = Field(..., description="Trading date for this breadth snapshot")

    # Daily movers (4%+ threshold)
    stocks_up_4pct: int = Field(..., description="Number of stocks up 4%+ today")
    stocks_down_4pct: int = Field(..., description="Number of stocks down 4%+ today")

    # Multi-day ratios
    ratio_5day: Optional[float] = Field(None, description="5-day up/down ratio")
    ratio_10day: Optional[float] = Field(None, description="10-day up/down ratio")

    # Quarterly movers (65 trading sessions)
    stocks_up_25pct_quarter: int = Field(..., description="Stocks up 25%+ from a 65-session low")
    stocks_down_25pct_quarter: int = Field(..., description="Stocks down 25%+ from a 65-session high")

    # Monthly movers (exactly 20 trading sessions, 25% threshold)
    stocks_up_25pct_month: int = Field(..., description="Stocks up 25%+ in a month")
    stocks_down_25pct_month: int = Field(..., description="Stocks down 25%+ in a month")

    # Monthly extreme movers (exactly 20 trading sessions, 50% threshold)
    stocks_up_50pct_month: int = Field(..., description="Stocks up 50%+ in a month")
    stocks_down_50pct_month: int = Field(..., description="Stocks down 50%+ in a month")

    # 34-day movers (13% threshold)
    stocks_up_13pct_34days: int = Field(..., description="Stocks up 13%+ in 34 days")
    stocks_down_13pct_34days: int = Field(..., description="Stocks down 13%+ in 34 days")

    # Broad-universe context metrics (optional for legacy rows)
    advancing_count: Optional[int] = Field(None, description="Stocks advancing from the prior adjusted close")
    declining_count: Optional[int] = Field(None, description="Stocks declining from the prior adjusted close")
    unchanged_count: Optional[int] = Field(None, description="Stocks unchanged from the prior adjusted close")
    new_high_52week_count: Optional[int] = Field(None, description="Strict new adjusted 52-week highs (StockBee)")
    new_low_52week_count: Optional[int] = Field(None, description="Strict new adjusted 52-week lows (StockBee)")
    t2108_count: Optional[int] = Field(None, description="Stocks above their adjusted 40-day moving average")
    t2108_pct: Optional[float] = Field(None, description="T2108 percentage of its eligible universe")
    atr_10x_extension_count: Optional[int] = Field(None, description="Stocks extended at least 10 ATR from SMA50 (screenshot-derived)")
    above_sma200_count: Optional[int] = Field(None, description="Stocks above their adjusted 200-day moving average")
    above_sma200_pct: Optional[float] = Field(None, description="Percent of the 200-day eligible universe above its 200-day average")
    down_20pct_from_high_count: Optional[int] = Field(None, description="Stocks at least 20% below their adjusted 252-session high")
    down_20pct_from_high_pct: Optional[float] = Field(None, description="Percent of the off-high eligible universe at least 20% below its high")
    up_4pct_near_high_count: Optional[int] = Field(None, description="Stocks up 4%+ that close within 15% of their 252-session high")
    above_sma20_count: Optional[int] = Field(None, description="Stocks above their adjusted 20-day moving average")
    above_sma20_pct: Optional[float] = Field(None, description="Percent of the 20-day eligible universe above its 20-day average")
    above_sma50_count: Optional[int] = Field(None, description="Stocks above their adjusted 50-day moving average")
    above_sma50_pct: Optional[float] = Field(None, description="Percent of the 50-day eligible universe above its 50-day average")
    trend_template_entry_count: Optional[int] = Field(None, description="Stocks that met the price trend template today but not the session before")

    # Metric-specific eligible denominators
    broad_universe_count: Optional[int] = Field(None, description="Active point-in-time common-stock universe")
    advance_decline_eligible_count: Optional[int] = None
    stockbee_daily_eligible_count: Optional[int] = None
    stockbee_month_eligible_count: Optional[int] = None
    stockbee_34day_eligible_count: Optional[int] = None
    stockbee_quarter_eligible_count: Optional[int] = None
    t2108_eligible_count: Optional[int] = None
    high_low_52week_eligible_count: Optional[int] = None
    atr_extension_eligible_count: Optional[int] = None
    sma200_eligible_count: Optional[int] = None
    off_high_eligible_count: Optional[int] = None
    sma20_eligible_count: Optional[int] = None
    sma50_eligible_count: Optional[int] = None
    trend_template_entry_eligible_count: Optional[int] = None

    # Metadata
    total_stocks_scanned: int = Field(
        ...,
        description="Deprecated compatibility alias for broad_universe_count",
    )
    eligibility_signature: Optional[str] = Field(None, description="Broad-universe signature")
    stockbee_eligibility_signature: Optional[str] = Field(None, description="StockBee liquidity-universe signature")
    calculation_revision: Optional[int] = Field(
        None,
        description="Internal stale-data guard; current value is 3",
    )
    calculation_duration_seconds: Optional[float] = Field(None, description="Time taken to calculate")

    class Config:
        from_attributes = True  # Pydantic v2 (replaces orm_mode)


class TrendDataPoint(BaseModel):
    """Single data point for trend visualization"""

    date: str = Field(..., description="Date in YYYY-MM-DD format")
    value: Optional[float] = Field(None, description="Indicator value for this date")


class TrendResponse(BaseModel):
    """Response model for indicator trend data"""

    indicator: str = Field(..., description="Indicator name")
    market: str = Field("US", description="Market code for this trend")
    data: List[TrendDataPoint] = Field(..., description="Time series data points")
    total_points: int = Field(..., description="Number of data points returned")


class CalculationRequest(BaseModel):
    """Request model for manual breadth calculation"""

    market: str = Field("US", description="Market code: US, HK, IN, JP, KR, TW, CN, SG, CA, or DE")
    calculation_date: Optional[str] = Field(
        None,
        description="Date to calculate for (YYYY-MM-DD), defaults to today"
    )


class CalculationResponse(BaseModel):
    """Response model for triggered breadth calculation"""

    status: str = Field(..., description="Status of the calculation request")
    message: str = Field(..., description="Human-readable message")
    task_id: Optional[str] = Field(None, description="Celery task ID for tracking")


class BackfillRequest(BaseModel):
    """Request model for historical backfill"""

    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    market: str = Field("US", description="Market code: US, HK, IN, JP, KR, TW, CN, SG, CA, or DE")


class BackfillResponse(BaseModel):
    """Response model for triggered backfill task"""

    status: str = Field(..., description="Status of the backfill request")
    message: str = Field(..., description="Human-readable message")
    task_id: str = Field(..., description="Celery task ID for tracking progress")
    dates_to_process: int = Field(..., description="Estimated number of trading days to process")


class BreadthSummary(BaseModel):
    """Summary statistics for breadth data"""

    market: str = Field("US", description="Market code for this summary")
    latest_date: Optional[Date] = Field(None, description="Most recent breadth date")
    total_records: int = Field(..., description="Total breadth records in database")
    date_range_start: Optional[Date] = Field(None, description="Earliest date with data")
    date_range_end: Optional[Date] = Field(None, description="Latest date with data")
