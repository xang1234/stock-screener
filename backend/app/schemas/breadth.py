"""Pydantic schemas for market breadth API endpoints"""
from pydantic import BaseModel, Field
from datetime import date as Date
from typing import Optional, List, Dict, Any


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

    # Quarterly movers (63 trading days)
    stocks_up_25pct_quarter: int = Field(..., description="Stocks up 25%+ in a quarter")
    stocks_down_25pct_quarter: int = Field(..., description="Stocks down 25%+ in a quarter")

    # Monthly movers (21 trading days, 25% threshold)
    stocks_up_25pct_month: int = Field(..., description="Stocks up 25%+ in a month")
    stocks_down_25pct_month: int = Field(..., description="Stocks down 25%+ in a month")

    # Monthly extreme movers (21 trading days, 50% threshold)
    stocks_up_50pct_month: int = Field(..., description="Stocks up 50%+ in a month")
    stocks_down_50pct_month: int = Field(..., description="Stocks down 50%+ in a month")

    # 34-day movers (13% threshold)
    stocks_up_13pct_34days: int = Field(..., description="Stocks up 13%+ in 34 days")
    stocks_down_13pct_34days: int = Field(..., description="Stocks down 13%+ in 34 days")

    # Metadata
    total_stocks_scanned: int = Field(..., description="Total stocks scanned")
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

    market: str = Field("US", description="Market code: US, HK, IN, JP, or TW")
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
    market: str = Field("US", description="Market code: US, HK, IN, JP, or TW")


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
