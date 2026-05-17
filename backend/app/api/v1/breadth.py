"""
API endpoints for market breadth indicators.

Provides access to current and historical breadth data,
trend analysis, and manual calculation triggers.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import date, datetime, timedelta
from typing import List, Optional

from ...config import settings
from ...database import get_db
from ...domain.markets.catalog import get_market_catalog
from ...models.market_breadth import MarketBreadth
from ...schemas.breadth import (
    BreadthResponse,
    TrendResponse,
    TrendDataPoint,
    CalculationRequest,
    CalculationResponse,
    BackfillRequest,
    BackfillResponse,
    BreadthSummary
)
from ...schemas.ui_view_snapshot import UISnapshotEnvelope
from ...tasks.market_queues import SUPPORTED_MARKETS
from ...wiring.bootstrap import get_ui_snapshot_service

logger = logging.getLogger(__name__)

router = APIRouter()
# Restrict breadth requests to markets that declare ``breadth=True`` in the
# market catalog. Markets like SG ship without the breadth dataset and would
# otherwise return 404s for any request that reached the task layer.
_market_catalog = get_market_catalog()
SUPPORTED_BREADTH_MARKETS = {
    code
    for code in _market_catalog.supported_market_codes()
    if _market_catalog.get(code).capabilities.breadth
} & set(SUPPORTED_MARKETS)
_BREADTH_MARKET_QUERY_CODES = [
    code for code in SUPPORTED_MARKETS if code in SUPPORTED_BREADTH_MARKETS
]
MARKET_QUERY_DESCRIPTION = (
    "Market code: " + ", ".join(_BREADTH_MARKET_QUERY_CODES)
)
OPTIONAL_MARKET_QUERY_DESCRIPTION = (
    "Optional market override. " + MARKET_QUERY_DESCRIPTION
)


def _normalize_market_param(market: str | None) -> str:
    normalized = str(market or "US").strip().upper()
    if normalized not in SUPPORTED_BREADTH_MARKETS:
        supported = ", ".join(sorted(SUPPORTED_BREADTH_MARKETS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported market '{market}'. Expected one of: {supported}.",
        )
    return normalized


def _require_task_controls() -> None:
    if not settings.feature_tasks:
        raise HTTPException(
            status_code=403,
            detail="Manual task controls are disabled in desktop mode.",
        )


@router.get("/current", response_model=BreadthResponse)
async def get_current_breadth(
    market: str = Query("US", description=MARKET_QUERY_DESCRIPTION),
    db: Session = Depends(get_db),
):
    """
    Get the most recent market breadth data.

    Returns the latest breadth snapshot from the database.
    """
    normalized_market = _normalize_market_param(market)
    breadth = db.query(MarketBreadth).filter(
        MarketBreadth.market == normalized_market,
    ).order_by(
        MarketBreadth.date.desc()
    ).first()

    if not breadth:
        raise HTTPException(
            status_code=404,
            detail=f"No breadth data available for market {normalized_market}. Run a calculation first."
        )

    return breadth


@router.get("/bootstrap", response_model=UISnapshotEnvelope)
async def get_breadth_bootstrap(
    market: str = Query("US", description=MARKET_QUERY_DESCRIPTION),
    snapshot_service=Depends(get_ui_snapshot_service),
):
    """Return the published breadth bootstrap snapshot if available."""
    normalized_market = _normalize_market_param(market)
    snapshot = snapshot_service.get_breadth_bootstrap(market=normalized_market)
    if snapshot is None:
        raise HTTPException(
            status_code=404,
            detail=f"No published breadth bootstrap snapshot is available for market {normalized_market}",
        )
    return UISnapshotEnvelope(**snapshot.to_dict())


@router.get("/historical", response_model=List[BreadthResponse])
async def get_historical_breadth(
    start_date: date = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: date = Query(..., description="End date (YYYY-MM-DD)"),
    limit: int = Query(365, ge=1, le=730, description="Maximum number of records"),
    market: str = Query("US", description=MARKET_QUERY_DESCRIPTION),
    db: Session = Depends(get_db)
):
    """
    Get historical breadth data for a date range.

    Args:
        start_date: Start of date range
        end_date: End of date range
        limit: Maximum number of records (default: 365, max: 365)

    Returns:
        List of breadth records for the specified date range
    """
    # Validate date range
    if start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail=f"Start date ({start_date}) must be before end date ({end_date})"
        )

    # Limit range to 730 days (2 years)
    max_range = timedelta(days=730)
    if (end_date - start_date) > max_range:
        raise HTTPException(
            status_code=400,
            detail=f"Date range cannot exceed 730 days (2 years)"
        )

    normalized_market = _normalize_market_param(market)

    # Query breadth data
    breadth_records = db.query(MarketBreadth).filter(
        MarketBreadth.date >= start_date,
        MarketBreadth.date <= end_date,
        MarketBreadth.market == normalized_market,
    ).order_by(
        MarketBreadth.date.desc()
    ).limit(limit).all()

    if not breadth_records:
        raise HTTPException(
            status_code=404,
            detail=f"No breadth data found for market {normalized_market} date range {start_date} to {end_date}"
        )

    return breadth_records


@router.get("/trend/{indicator}", response_model=TrendResponse)
async def get_indicator_trend(
    indicator: str,
    days: int = Query(30, ge=1, le=730, description="Number of days to retrieve"),
    market: str = Query("US", description=MARKET_QUERY_DESCRIPTION),
    db: Session = Depends(get_db)
):
    """
    Get time series data for a specific breadth indicator.

    Args:
        indicator: Indicator name (e.g., 'stocks_up_4pct', 'ratio_5day')
        days: Number of days to retrieve (default: 30, max: 365)

    Returns:
        Time series data for the specified indicator
    """
    # Validate indicator name
    valid_indicators = [
        'stocks_up_4pct', 'stocks_down_4pct',
        'ratio_5day', 'ratio_10day',
        'stocks_up_25pct_quarter', 'stocks_down_25pct_quarter',
        'stocks_up_25pct_month', 'stocks_down_25pct_month',
        'stocks_up_50pct_month', 'stocks_down_50pct_month',
        'stocks_up_13pct_34days', 'stocks_down_13pct_34days',
        'total_stocks_scanned'
    ]

    if indicator not in valid_indicators:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid indicator '{indicator}'. Must be one of: {', '.join(valid_indicators)}"
        )

    # Get recent breadth data
    normalized_market = _normalize_market_param(market)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)

    breadth_records = db.query(MarketBreadth).filter(
        MarketBreadth.date >= start_date,
        MarketBreadth.date <= end_date,
        MarketBreadth.market == normalized_market,
    ).order_by(
        MarketBreadth.date.asc()
    ).all()

    if not breadth_records:
        raise HTTPException(
            status_code=404,
            detail=f"No breadth data found for market {normalized_market} in the last {days} days"
        )

    # Extract indicator values
    data_points = []
    for record in breadth_records:
        value = getattr(record, indicator, None)
        data_points.append(TrendDataPoint(
            date=record.date.strftime('%Y-%m-%d'),
            value=value
        ))

    return TrendResponse(
        indicator=indicator,
        market=normalized_market,
        data=data_points,
        total_points=len(data_points)
    )


@router.post("/calculate", response_model=CalculationResponse)
async def trigger_calculation(
    request: CalculationRequest,
    background_tasks: BackgroundTasks,
    market: str | None = Query(None, description=OPTIONAL_MARKET_QUERY_DESCRIPTION),
    db: Session = Depends(get_db)
):
    """
    Manually trigger a breadth calculation.

    Can be used to calculate breadth for a specific date or today.
    Runs as a background task.

    Args:
        request: Calculation request with optional date

    Returns:
        Status and task information
    """
    _require_task_controls()
    normalized_market = _normalize_market_param(market or request.market)
    calculation_date = request.calculation_date

    # Validate date format if provided
    if calculation_date:
        try:
            datetime.strptime(calculation_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format '{calculation_date}'. Use YYYY-MM-DD"
            )

    # Trigger background task
    from ...tasks.breadth_tasks import calculate_daily_breadth

    background_tasks.add_task(calculate_daily_breadth, calculation_date, market=normalized_market)

    date_str = calculation_date or "today"
    return CalculationResponse(
        status="started",
        message=f"Breadth calculation triggered for {normalized_market} {date_str}",
        task_id=None  # Background tasks don't return task IDs
    )


@router.post("/backfill", response_model=BackfillResponse)
async def trigger_backfill(
    request: BackfillRequest,
    market: str | None = Query(None, description=OPTIONAL_MARKET_QUERY_DESCRIPTION),
    db: Session = Depends(get_db)
):
    """
    Trigger a historical backfill for a date range.

    Calculates breadth for all trading days in the specified range.
    Runs as a Celery task with progress tracking.

    Args:
        request: Backfill request with start and end dates

    Returns:
        Task information for tracking progress
    """
    _require_task_controls()
    normalized_market = _normalize_market_param(market or request.market)
    # Validate date format
    try:
        start = datetime.strptime(request.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(request.end_date, "%Y-%m-%d").date()
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format. Use YYYY-MM-DD: {str(e)}"
        )

    # Validate date range
    if start > end:
        raise HTTPException(
            status_code=400,
            detail=f"Start date ({start}) must be before end date ({end})"
        )

    # Limit backfill to 2 years
    max_range = timedelta(days=730)
    if (end - start) > max_range:
        raise HTTPException(
            status_code=400,
            detail=f"Backfill range cannot exceed 2 years (730 days)"
        )

    # Estimate number of trading days (roughly 5/7 of calendar days)
    total_days = (end - start).days + 1
    estimated_trading_days = int(total_days * 5 / 7)

    # Trigger Celery task
    from ...tasks.breadth_tasks import backfill_breadth_data

    task = backfill_breadth_data.delay(request.start_date, request.end_date, market=normalized_market)

    logger.info(
        "Backfill task triggered: %s for %s to %s market=%s",
        task.id,
        request.start_date,
        request.end_date,
        normalized_market,
    )

    return BackfillResponse(
        status="started",
        message=f"Backfill task started for {normalized_market} {request.start_date} to {request.end_date}",
        task_id=task.id,
        dates_to_process=estimated_trading_days
    )


@router.get("/summary", response_model=BreadthSummary)
async def get_breadth_summary(
    market: str = Query("US", description=MARKET_QUERY_DESCRIPTION),
    db: Session = Depends(get_db),
):
    """
    Get summary statistics for available breadth data.

    Returns information about the breadth data coverage,
    including date ranges and record counts.
    """
    normalized_market = _normalize_market_param(market)

    # Get total record count for the requested market partition.
    total_records = db.query(func.count(MarketBreadth.id)).filter(
        MarketBreadth.market == normalized_market,
    ).scalar() or 0

    if total_records == 0:
        return BreadthSummary(
            market=normalized_market,
            latest_date=None,
            total_records=0,
            date_range_start=None,
            date_range_end=None
        )

    # Get date range
    min_date = db.query(func.min(MarketBreadth.date)).filter(
        MarketBreadth.market == normalized_market,
    ).scalar()
    max_date = db.query(func.max(MarketBreadth.date)).filter(
        MarketBreadth.market == normalized_market,
    ).scalar()

    return BreadthSummary(
        market=normalized_market,
        latest_date=max_date,
        total_records=total_records,
        date_range_start=min_date,
        date_range_end=max_date
    )
