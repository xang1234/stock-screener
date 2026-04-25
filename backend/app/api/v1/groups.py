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

from ...config import settings
from ...database import get_db
from ...models.industry import IBDGroupRank
from ...schemas.common import TaskResponse
from ...schemas.groups import (
    GroupRankResponse,
    GroupRankingsResponse,
    GroupDetailResponse,
    MoversResponse,
    CalculationRequest,
    CalculationResponse,
    CalculationStatusResponse,
    BackfillRequest,
    BackfillResponse,
)
from ...schemas.ui_view_snapshot import UISnapshotEnvelope
from ...domain.analytics.scope import market_scope_tag
from ...services.market_group_ranking_service import get_market_group_ranking_service
from ...services.ui_snapshot_service import GroupsBootstrapUnavailableError
from ...wiring.bootstrap import get_group_rank_service, get_ui_snapshot_service

logger = logging.getLogger(__name__)

router = APIRouter()
SUPPORTED_GROUP_MARKETS = {"US", "HK", "IN", "JP", "TW"}
DEFAULT_GROUP_PERIOD = "1w"


def _get_group_rank_service():
    return get_group_rank_service()


def _get_market_group_service():
    return get_market_group_ranking_service()


def _normalize_market_param(market: str | None) -> str:
    normalized = str(market or "US").strip().upper()
    if normalized not in SUPPORTED_GROUP_MARKETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported market '{market}'. Expected one of: US, HK, IN, JP, TW.",
        )
    return normalized


def _build_groups_payload(db: Session, *, market: str) -> dict:
    if market == "US":
        service = _get_group_rank_service()
        rankings = service.get_current_rankings(db, limit=197)
        movers = service.get_rank_movers(db, period=DEFAULT_GROUP_PERIOD, limit=10)
    else:
        service = _get_market_group_service()
        rankings = service.get_current_rankings(db, market=market, limit=197)
        movers = service.get_rank_movers(
            db,
            market=market,
            period=DEFAULT_GROUP_PERIOD,
            limit=10,
        )

    if not rankings:
        raise GroupsBootstrapUnavailableError(
            f"No group rankings are available for market {market}."
        )

    ranking_date = rankings[0]["date"]
    scope = market_scope_tag(market)
    return {
        "rankings": GroupRankingsResponse(
            date=ranking_date,
            total_groups=len(rankings),
            rankings=[GroupRankResponse(**row) for row in rankings],
            **scope,
        ).model_dump(mode="json"),
        "movers_period": DEFAULT_GROUP_PERIOD,
        "movers": MoversResponse(
            period=movers["period"],
            gainers=[GroupRankResponse(**row) for row in movers.get("gainers", [])],
            losers=[GroupRankResponse(**row) for row in movers.get("losers", [])],
            **scope,
        ).model_dump(mode="json"),
        "task_controls_enabled": settings.feature_tasks and market == "US",
    }


def _require_task_controls() -> None:
    if not settings.feature_tasks:
        raise HTTPException(
            status_code=403,
            detail="Manual task controls are disabled in desktop mode.",
        )


@router.get("/rankings/current", response_model=GroupRankingsResponse)
async def get_current_rankings(
    limit: int = Query(50, ge=1, le=197, description="Number of groups to return"),
    market: str = Query("US", description="Market code: US, HK, IN, JP, or TW"),
    db: Session = Depends(get_db)
):
    """
    Get current IBD group rankings sorted by rank (best first).

    Returns the most recent ranking snapshot with rank changes
    for 1 week, 1 month, 3 months, and 6 months.
    """
    normalized_market = _normalize_market_param(market)
    if normalized_market == "US":
        service = _get_group_rank_service()
        rankings = service.get_current_rankings(db, limit=limit)
    else:
        service = _get_market_group_service()
        rankings = service.get_current_rankings(
            db,
            market=normalized_market,
            limit=limit,
        )

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
        rankings=[GroupRankResponse(**r) for r in rankings],
        **market_scope_tag(normalized_market),
    )


@router.get("/bootstrap", response_model=UISnapshotEnvelope)
async def get_groups_bootstrap(
    market: str = Query("US", description="Market code: US, HK, IN, JP, or TW"),
    db: Session = Depends(get_db),
    snapshot_service=Depends(get_ui_snapshot_service),
):
    """Return the published group rankings bootstrap snapshot if available."""
    normalized_market = _normalize_market_param(market)
    if normalized_market == "US":
        snapshot = snapshot_service.get_groups_bootstrap()
        if snapshot is not None and not snapshot.is_stale:
            return UISnapshotEnvelope(**snapshot.to_dict())

    try:
        payload = _build_groups_payload(db, market=normalized_market)
    except GroupsBootstrapUnavailableError as exc:
        if normalized_market == "US":
            raise HTTPException(
                status_code=404,
                detail="No published groups bootstrap snapshot is available",
            ) from exc
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    source_revision = str(((payload.get("rankings") or {}).get("date")) or "none")
    return UISnapshotEnvelope(
        snapshot_revision=f"groups:{normalized_market}:{source_revision}",
        source_revision=f"{normalized_market}:{source_revision}",
        published_at=datetime.utcnow(),
        is_stale=False,
        payload=payload,
    )


@router.get("/rankings/movers", response_model=MoversResponse)
async def get_rank_movers(
    period: str = Query("1w", pattern="^(1w|1m|3m|6m)$", description="Time period"),
    limit: int = Query(20, ge=1, le=50, description="Number of movers per direction"),
    market: str = Query("US", description="Market code: US, HK, IN, JP, or TW"),
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
    normalized_market = _normalize_market_param(market)
    if normalized_market == "US":
        service = _get_group_rank_service()
        movers = service.get_rank_movers(db, period=period, limit=limit)
    else:
        service = _get_market_group_service()
        movers = service.get_rank_movers(
            db,
            market=normalized_market,
            period=period,
            limit=limit,
        )

    if not movers.get('gainers') and not movers.get('losers'):
        raise HTTPException(
            status_code=404,
            detail=f"No mover data available for period '{period}'"
        )

    return MoversResponse(
        period=movers['period'],
        gainers=[GroupRankResponse(**g) for g in movers.get('gainers', [])],
        losers=[GroupRankResponse(**loser) for loser in movers.get('losers', [])],
        **market_scope_tag(normalized_market),
    )


@router.get("/rankings/detail", response_model=GroupDetailResponse)
async def get_group_detail(
    group: str = Query(..., description="IBD industry group name"),
    days: int = Query(180, ge=1, le=365, description="Days of history to retrieve"),
    market: str = Query("US", description="Market code: US, HK, IN, JP, or TW"),
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
    normalized_market = _normalize_market_param(market)
    if normalized_market == "US":
        service = _get_group_rank_service()
        detail = service.get_group_history(db, group, days=days)
    else:
        service = _get_market_group_service()
        detail = service.get_group_history(
            db,
            market=normalized_market,
            industry_group=group,
            days=days,
        )

    if not detail.get('history'):
        raise HTTPException(
            status_code=404,
            detail=f"No data found for industry group '{group}'"
        )

    return GroupDetailResponse(**detail, **market_scope_tag(normalized_market))


@router.post("/rankings/calculate", response_model=TaskResponse)
async def trigger_calculation(request: CalculationRequest):
    """
    Manually trigger a group ranking calculation.

    Dispatches the calculation to a Celery background task and returns
    a task_id for status polling. The calculation runs asynchronously.
    """
    _require_task_controls()
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
        from ...tasks.group_rank_tasks import calculate_daily_group_rankings

        task = calculate_daily_group_rankings.delay(date_str, market="US")
        logger.info(f"Group ranking calculation task dispatched: {task.id}")

        return TaskResponse(
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
    _require_task_controls()
    try:
        from celery.result import AsyncResult

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
                    error=(result or {}).get('error', 'Unknown error'),
                    reason_code=(result or {}).get('reason_code'),
                )
        elif task_result.state == 'FAILURE':
            from ...tasks.group_rank_tasks import GroupRankReasonCode

            return CalculationStatusResponse(
                task_id=task_id,
                status="failed",
                error=str(task_result.result) if task_result.result else "Task failed",
                reason_code=GroupRankReasonCode.UNKNOWN,
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
    _require_task_controls()
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
        task = backfill_group_rankings.delay(request.start_date, request.end_date, market="US")
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

    # This summary endpoint is US-scoped today; non-US partitions get their
    # own surfaces via _get_market_group_service.
    us_only = IBDGroupRank.market == "US"

    total_records = db.query(func.count(IBDGroupRank.id)).filter(us_only).scalar() or 0

    if total_records == 0:
        return {
            "status": "no_data",
            "total_records": 0,
            "total_groups": 0,
            "latest_date": None,
            "earliest_date": None,
        }

    min_date = db.query(func.min(IBDGroupRank.date)).filter(us_only).scalar()
    max_date = db.query(func.max(IBDGroupRank.date)).filter(us_only).scalar()

    unique_groups = db.query(
        func.count(func.distinct(IBDGroupRank.industry_group))
    ).filter(us_only).scalar()

    latest_count = db.query(func.count(IBDGroupRank.id)).filter(
        IBDGroupRank.date == max_date,
        us_only,
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
    service = get_group_rank_service()

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
        task = gapfill_group_rankings.delay(max_days=max_days, market="US")
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
