"""
API endpoints for IBD Industry Group Rankings.

Provides access to current and historical group rankings,
rank movers, and manual calculation triggers.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List

from ...database import get_db
from ...models.industry import IBDGroupRank
from celery.result import AsyncResult
from ...schemas.groups import (
    GroupRankResponse,
    GroupRankingsResponse,
    GroupDetailResponse,
    MoversResponse,
    CalculationRequest,
    CalculationResponse,
    CalculationTaskResponse,
    CalculationStatusResponse,
    BackfillRequest,
    BackfillResponse,
)
from ...services.ibd_group_rank_service import IBDGroupRankService
from ...tasks.group_rank_tasks import calculate_daily_group_rankings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/rankings/current", response_model=GroupRankingsResponse)
async def get_current_rankings(
    limit: int = Query(50, ge=1, le=197, description="Number of groups to return"),
    db: Session = Depends(get_db)
):
    """
    Get current IBD group rankings sorted by rank (best first).

    Returns the most recent ranking snapshot with rank changes
    for 1 week, 1 month, 3 months, and 6 months.
    """
    service = IBDGroupRankService.get_instance()
    rankings = service.get_current_rankings(db, limit=limit)

    if not rankings:
        raise HTTPException(
            status_code=404,
            detail="No ranking data available. Run a calculation first."
        )

    # Get the date from the first ranking
    ranking_date = rankings[0]['date'] if rankings else None

    return GroupRankingsResponse(
        date=ranking_date,
        total_groups=len(rankings),
        rankings=[GroupRankResponse(**r) for r in rankings]
    )


@router.get("/rankings/movers", response_model=MoversResponse)
async def get_rank_movers(
    period: str = Query("1w", regex="^(1w|1m|3m|6m)$", description="Time period"),
    limit: int = Query(20, ge=1, le=50, description="Number of movers per direction"),
    db: Session = Depends(get_db)
):
    """
    Get groups with biggest rank changes (up or down) over a period.

    Args:
        period: Time period - '1w' (1 week), '1m' (1 month), '3m' (3 months), '6m' (6 months)
        limit: Number of top gainers/losers to return

    Returns:
        Lists of rank gainers and losers
    """
    service = IBDGroupRankService.get_instance()
    movers = service.get_rank_movers(db, period=period, limit=limit)

    if not movers.get('gainers') and not movers.get('losers'):
        raise HTTPException(
            status_code=404,
            detail=f"No mover data available for period '{period}'"
        )

    return MoversResponse(
        period=movers['period'],
        gainers=[GroupRankResponse(**g) for g in movers.get('gainers', [])],
        losers=[GroupRankResponse(**l) for l in movers.get('losers', [])]
    )


@router.get("/rankings/detail", response_model=GroupDetailResponse)
async def get_group_detail(
    group: str = Query(..., description="IBD industry group name"),
    days: int = Query(180, ge=1, le=365, description="Days of history to retrieve"),
    db: Session = Depends(get_db)
):
    """
    Get detailed ranking history for a specific industry group.

    Args:
        group: IBD industry group name (as query parameter to handle special characters like slashes)
        days: Number of days of historical data

    Returns:
        Current rank, rank changes, and historical data points
    """
    service = IBDGroupRankService.get_instance()
    detail = service.get_group_history(db, group, days=days)

    if not detail.get('history'):
        raise HTTPException(
            status_code=404,
            detail=f"No data found for industry group '{group}'"
        )

    return GroupDetailResponse(**detail)


@router.post("/rankings/calculate", response_model=CalculationTaskResponse)
async def trigger_calculation(request: CalculationRequest):
    """
    Manually trigger a group ranking calculation.

    Dispatches the calculation to a Celery background task and returns
    a task_id for status polling. The calculation runs asynchronously.
    """
    # Validate date format if provided
    date_str = None
    if request.calculation_date:
        try:
            datetime.strptime(request.calculation_date, "%Y-%m-%d")
            date_str = request.calculation_date
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format '{request.calculation_date}'. Use YYYY-MM-DD"
            )

    # Dispatch to Celery
    try:
        task = calculate_daily_group_rankings.delay(date_str)
        logger.info(f"Group ranking calculation task dispatched: {task.id}")

        return CalculationTaskResponse(
            task_id=task.id,
            status="queued",
            message=f"Calculation task queued for {date_str or 'today'}"
        )

    except Exception as e:
        logger.error(f"Failed to dispatch calculation task: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue calculation task: {str(e)}"
        )


@router.get("/rankings/calculate/status/{task_id}", response_model=CalculationStatusResponse)
async def get_calculation_status(task_id: str):
    """
    Get the status of a group ranking calculation task.

    Poll this endpoint to check if the calculation is complete.
    """
    try:
        task_result = AsyncResult(task_id)

        if task_result.state == 'PENDING':
            return CalculationStatusResponse(
                task_id=task_id,
                status="queued"
            )
        elif task_result.state == 'STARTED':
            return CalculationStatusResponse(
                task_id=task_id,
                status="running"
            )
        elif task_result.state == 'SUCCESS':
            result = task_result.result
            # Map Celery task result to CalculationResponse
            if result and not result.get('error'):
                return CalculationStatusResponse(
                    task_id=task_id,
                    status="completed",
                    result=CalculationResponse(
                        status="completed",
                        message=f"Successfully ranked {result.get('groups_ranked', 0)} groups",
                        groups_ranked=result.get('groups_ranked'),
                        date=result.get('date')
                    )
                )
            else:
                # Task returned an error dict
                return CalculationStatusResponse(
                    task_id=task_id,
                    status="failed",
                    error=result.get('error', 'Unknown error')
                )
        elif task_result.state == 'FAILURE':
            return CalculationStatusResponse(
                task_id=task_id,
                status="failed",
                error=str(task_result.result) if task_result.result else "Task failed"
            )
        else:
            # Handle other states (RETRY, REVOKED, etc.)
            return CalculationStatusResponse(
                task_id=task_id,
                status="running"
            )

    except Exception as e:
        logger.error(f"Error checking calculation status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check task status: {str(e)}"
        )


@router.post("/rankings/backfill")
async def trigger_backfill(
    request: BackfillRequest,
):
    """
    Trigger a historical backfill for a date range.

    Dispatches a Celery task that calculates rankings for all trading days
    in the specified range. The task runs asynchronously in the background.

    Args:
        request: Backfill request with start and end dates

    Returns:
        Task information for tracking progress
    """
    from ...tasks.group_rank_tasks import backfill_group_rankings

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

    # Limit backfill to 1 year
    max_range = timedelta(days=365)
    if (end - start) > max_range:
        raise HTTPException(
            status_code=400,
            detail=f"Backfill range cannot exceed 1 year (365 days)"
        )

    # Dispatch to Celery (non-blocking)
    try:
        task = backfill_group_rankings.delay(request.start_date, request.end_date)
        logger.info(f"Backfill task dispatched: {task.id}")

        return {
            "status": "started",
            "task_id": task.id,
            "message": f"Backfill task started for {request.start_date} to {request.end_date}",
            "start_date": request.start_date,
            "end_date": request.end_date,
        }

    except Exception as e:
        logger.error(f"Failed to dispatch backfill task: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start backfill: {str(e)}"
        )


@router.get("/summary")
async def get_rankings_summary(db: Session = Depends(get_db)):
    """
    Get summary statistics for available ranking data.

    Returns information about data coverage and date ranges.
    """
    from sqlalchemy import func

    # Get total record count
    total_records = db.query(func.count(IBDGroupRank.id)).scalar() or 0

    if total_records == 0:
        return {
            "status": "no_data",
            "total_records": 0,
            "total_groups": 0,
            "latest_date": None,
            "earliest_date": None,
        }

    # Get date range
    min_date = db.query(func.min(IBDGroupRank.date)).scalar()
    max_date = db.query(func.max(IBDGroupRank.date)).scalar()

    # Get unique group count
    unique_groups = db.query(func.count(func.distinct(IBDGroupRank.industry_group))).scalar()

    # Get count for latest date
    latest_count = db.query(func.count(IBDGroupRank.id)).filter(
        IBDGroupRank.date == max_date
    ).scalar()

    return {
        "status": "ok",
        "total_records": total_records,
        "total_groups": unique_groups,
        "groups_in_latest": latest_count,
        "latest_date": max_date.isoformat() if max_date else None,
        "earliest_date": min_date.isoformat() if min_date else None,
    }


@router.get("/rankings/gaps")
async def get_ranking_gaps(
    max_days: int = Query(365, ge=30, le=365, description="Days to look back for gaps"),
    db: Session = Depends(get_db)
):
    """
    Check for gaps in ranking data without triggering a fill.

    Returns a list of missing trading dates in the specified lookback period.
    Useful for diagnosing data completeness before running a backfill.

    Args:
        max_days: Number of days to look back (default: 365, max: 365)

    Returns:
        Gap statistics and list of missing dates (limited to first 50)
    """
    service = IBDGroupRankService.get_instance()

    missing_dates = service.find_missing_dates(db, lookback_days=max_days)

    return {
        "lookback_days": max_days,
        "gaps_found": len(missing_dates),
        "missing_dates": [d.isoformat() for d in missing_dates[:50]],
        "truncated": len(missing_dates) > 50,
        "oldest_gap": missing_dates[0].isoformat() if missing_dates else None,
        "newest_gap": missing_dates[-1].isoformat() if missing_dates else None,
    }


@router.post("/rankings/gapfill")
async def trigger_gapfill(
    max_days: int = Query(365, ge=30, le=365, description="Days to look back for gaps"),
):
    """
    Trigger gap-fill to detect and fill missing ranking dates.

    Dispatches a Celery task that scans the specified lookback period
    for missing trading days and calculates rankings for them.

    Args:
        max_days: Number of days to look back (default: 365, max: 365)

    Returns:
        Task information for tracking progress
    """
    from ...tasks.group_rank_tasks import gapfill_group_rankings

    try:
        # Dispatch as Celery task
        task = gapfill_group_rankings.delay(max_days=max_days)
        logger.info(f"Gap-fill task dispatched: {task.id}")

        return {
            "status": "started",
            "task_id": task.id,
            "message": f"Gap-fill task started, looking back {max_days} days",
        }

    except Exception as e:
        logger.error(f"Failed to dispatch gap-fill task: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start gap-fill: {str(e)}"
        )
