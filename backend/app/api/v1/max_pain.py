"""
Max Pain Dashboard API endpoints with database integration.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, and_, desc
from sqlalchemy.orm import Session

from ...database import get_db
from ...models.max_pain import MaxPainSnapshot, MaxPainBatch
from ...models.stock_universe import StockUniverse
from ...services.options_history_service import Timeframe, fetch_max_pain_history
from ...services.symbol_format import normalize_symbol
from ...tasks.market_queues import data_fetch_queue_for_market
from ...tasks.max_pain_tasks import update_max_pain

logger = logging.getLogger(__name__)
router = APIRouter()


class MaxPainRowResponse(BaseModel):
    """Single row in dashboard"""
    symbol: str
    company_name: str | None
    status: str
    expiration: str | None
    call_oi: int | None
    put_oi: int | None
    put_call_ratio: float | None
    max_pain: float | None
    distance_pct: float | None
    fetched_at: str


class MaxPainDashboardResponse(BaseModel):
    """Response model for max pain dashboard"""
    tickers_ok: int
    tickers_failed: int
    avg_put_call_ratio: float | None
    closest_to_max_pain: str | None
    generated_at: str
    strike_range_pct: float
    max_strikes: int
    total_rows: int
    total_pages: int
    current_page: int
    page_size: int
    rows: list[MaxPainRowResponse]


class BatchStatusResponse(BaseModel):
    """Status of a batch job"""
    batch_id: str
    status: str
    started_at: str
    completed_at: str | None
    tickers_ok: int | None
    tickers_failed: int | None
    error_message: str | None


@router.get("/dashboard")
async def get_max_pain_dashboard(
    db: Session = Depends(get_db),
    hours_back: int = Query(default=24, ge=1, le=720),
    exchange: str | None = Query(default=None, description="Optional exchange filter (e.g. NASDAQ, NYSE)"),
    symbol: str | None = Query(default=None, description="Optional symbol filter"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=10, le=500),
) -> dict:
    """
    Get the latest max pain dashboard data from database.
    
    Falls back to mock data if no recent data available.
    
    Args:
        hours_back: Only return data fetched within this many hours (default: 24)
    """
    try:
        normalized_exchange = (exchange or "").strip().upper() or None
        normalized_symbol = None
        if symbol is not None:
            normalized_symbol = normalize_symbol(symbol)
            if normalized_symbol is None:
                raise HTTPException(status_code=422, detail=f"Invalid symbol format: {symbol!r}")

        if normalized_symbol and normalized_exchange:
            universe_row = (
                db.query(StockUniverse.exchange)
                .filter(func.upper(StockUniverse.symbol) == normalized_symbol)
                .first()
            )
            if universe_row and universe_row.exchange:
                row_exchange = universe_row.exchange.strip().upper()
                if row_exchange != normalized_exchange:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Symbol {normalized_symbol} belongs to exchange {row_exchange}, "
                            f"not {normalized_exchange}."
                        ),
                    )

        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)

        latest_per_ticker = (
            db.query(
                MaxPainSnapshot.ticker.label("ticker"),
                func.max(MaxPainSnapshot.fetched_at).label("latest_fetched_at"),
            )
            .filter(MaxPainSnapshot.fetched_at >= cutoff_time)
            .group_by(MaxPainSnapshot.ticker)
            .subquery()
        )

        snapshot_query = (
            db.query(MaxPainSnapshot)
            .join(
                latest_per_ticker,
                and_(
                    MaxPainSnapshot.ticker == latest_per_ticker.c.ticker,
                    MaxPainSnapshot.fetched_at == latest_per_ticker.c.latest_fetched_at,
                ),
            )
        )

        if normalized_exchange:
            snapshot_query = snapshot_query.join(
                StockUniverse,
                StockUniverse.symbol == MaxPainSnapshot.ticker,
            ).filter(func.upper(StockUniverse.exchange) == normalized_exchange)

        if normalized_symbol:
            snapshot_query = snapshot_query.filter(MaxPainSnapshot.ticker == normalized_symbol)

        total_rows = snapshot_query.count()
        total_pages = max((total_rows + page_size - 1) // page_size, 1)
        safe_page = min(page, total_pages)
        offset = (safe_page - 1) * page_size

        rows = (
            snapshot_query
            .order_by(MaxPainSnapshot.ticker.asc())
            .offset(offset)
            .limit(page_size)
            .all()
        )

        # If the recent-time window returned very few rows, try a wider window
        # (helps when snapshot persistence is sparse). Only do this when the
        # caller requested a small hours_back so we don't silently override an
        # explicit long-range request.
        if len(rows) < min(10, page_size) and hours_back < 168:
            wider_cutoff = datetime.utcnow() - timedelta(hours=168)
            latest_per_ticker_wide = (
                db.query(
                    MaxPainSnapshot.ticker.label("ticker"),
                    func.max(MaxPainSnapshot.fetched_at).label("latest_fetched_at"),
                )
                .filter(MaxPainSnapshot.fetched_at >= wider_cutoff)
                .group_by(MaxPainSnapshot.ticker)
                .subquery()
            )

            snapshot_query_wide = (
                db.query(MaxPainSnapshot)
                .join(
                    latest_per_ticker_wide,
                    and_(
                        MaxPainSnapshot.ticker == latest_per_ticker_wide.c.ticker,
                        MaxPainSnapshot.fetched_at == latest_per_ticker_wide.c.latest_fetched_at,
                    ),
                )
            )

            if normalized_exchange:
                snapshot_query_wide = snapshot_query_wide.join(
                    StockUniverse,
                    StockUniverse.symbol == MaxPainSnapshot.ticker,
                ).filter(func.upper(StockUniverse.exchange) == normalized_exchange)

            if normalized_symbol:
                snapshot_query_wide = snapshot_query_wide.filter(MaxPainSnapshot.ticker == normalized_symbol)

            total_rows = snapshot_query_wide.count()
            total_pages = max((total_rows + page_size - 1) // page_size, 1)
            safe_page = min(page, total_pages)
            offset = (safe_page - 1) * page_size
            rows = (
                snapshot_query_wide
                .order_by(MaxPainSnapshot.ticker.asc())
                .offset(offset)
                .limit(page_size)
                .all()
            )
        
        if not rows:
            if normalized_symbol:
                raise HTTPException(
                    status_code=404,
                    detail=f"No max pain data available for symbol {normalized_symbol}",
                )
            logger.warning(f"No max pain data in database (within {hours_back}h), returning mock data")
            return get_mock_dashboard()
        
        # Calculate stats
        ok_count = len([r for r in rows if r.status == "OK"])
        failed_count = len([r for r in rows if r.status == "FAILED"])
        ratios = [r.put_call_ratio for r in rows if r.put_call_ratio]
        avg_ratio = sum(ratios) / len(ratios) if ratios else None
        
        # Find closest to max pain
        closest_ticker = None
        closest_dist = None
        for r in rows:
            if r.distance_pct is not None:
                if closest_dist is None or abs(r.distance_pct) < abs(closest_dist):
                    closest_dist = r.distance_pct
                    closest_ticker = r.ticker
        
        # Get latest batch info for generated_at
        latest_batch = db.query(MaxPainBatch).order_by(
            desc(MaxPainBatch.completed_at)
        ).first()
        generated_at = (
            latest_batch.completed_at.isoformat()
            if latest_batch and latest_batch.completed_at
            else rows[0].fetched_at.isoformat()
        )
        
        return {
            "tickers_ok": ok_count,
            "tickers_failed": failed_count,
            "avg_put_call_ratio": avg_ratio,
            "closest_to_max_pain": closest_ticker,
            "generated_at": generated_at,
            "strike_range_pct": latest_batch.strike_range_pct if latest_batch else 20,
            "max_strikes": latest_batch.max_strikes if latest_batch else 30,
            "total_rows": total_rows,
            "total_pages": total_pages,
            "current_page": safe_page,
            "page_size": page_size,
            "returned_rows": len(rows),
            "applied_filters": {
                "exchange": normalized_exchange,
                "symbol": normalized_symbol,
                "page": safe_page,
                "page_size": page_size,
                "hours_back": hours_back,
            },
            "rows": [
                {
                    "symbol": r.ticker,
                    "company_name": r.company_name,
                    "status": r.status,
                    "expiration": r.expiration.isoformat() if r.expiration else None,
                    "call_oi": r.call_oi,
                    "put_oi": r.put_oi,
                    "put_call_ratio": r.put_call_ratio,
                    "max_pain": r.max_pain_strike,
                    "distance_pct": r.distance_pct,
                    "fetched_at": r.fetched_at.isoformat(),
                }
                for r in rows
            ],
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error fetching max pain dashboard")
        if normalized_symbol:
            raise HTTPException(
                status_code=500,
                detail="Failed to fetch max pain dashboard for the requested symbol.",
            )
        # Return mock data on generic error for the full dashboard endpoint
        return get_mock_dashboard()


@router.get("/history")
async def get_max_pain_history(
    symbol: str = Query(...),
    timeframe: Timeframe = Query(default=Timeframe.MONTHLY),
    start: date | None = Query(default=None),
    end: date | None = Query(default=None),
    expiration: date | None = Query(
        default=None,
        description=(
            "Pin history to a specific expiration's series (from /v1/options/expirations/{symbol}). "
            "Omit for the default nearest-expiration daily-batch series."
        ),
    ),
    db: Session = Depends(get_db),
) -> dict:
    """Max Pain time series for one ticker over a timeframe window.

    Every point is a real point-in-time EOD snapshot -- never an average.
    Daily/Weekly/Monthly plot every trading day in the window; Quarterly/
    Yearly downsample to the last snapshot of each week/month respectively
    (still a real dated value, not a computed mean) to keep point counts
    chart-friendly. See `sampling`/`period` in the response.

    Without `expiration`, this is the daily batch's nearest-expiration
    series (unchanged behavior). With `expiration`, it's that specific
    expiration's series, built from /v1/options/term-structure calls --
    history only exists from whenever that expiration was first viewed.
    """
    normalized_symbol = normalize_symbol(symbol)
    if normalized_symbol is None:
        raise HTTPException(status_code=422, detail=f"Invalid symbol format: {symbol!r}")

    if start and end and start > end:
        raise HTTPException(status_code=422, detail="start must be on or before end")

    try:
        return fetch_max_pain_history(
            db, symbol=normalized_symbol, timeframe=timeframe, start=start, end=end,
            expiration=expiration,
        )
    except Exception:
        logger.exception("Error fetching max pain history for %s", normalized_symbol)
        raise HTTPException(status_code=500, detail="Failed to fetch max pain history")


@router.post("/trigger-update")
async def trigger_max_pain_update(
    config_path: str | None = Query(default=None),
    wait_until: str | None = Query(default=None),
) -> dict:
    """
    Trigger a background max pain batch update via Celery.
    
    Args:
        config_path: Optional path to config JSON. If omitted, uses all active
            symbols from stock_universe.
        wait_until: Optional HH:MM to wait until (e.g., "22:30")
    
    Returns:
        Task ID for tracking progress
    """
    try:
        logger.info("Max pain update triggered via API")

        # Explicit queue routing (not just .delay()) so this lands on the
        # single-concurrency data_fetch worker instead of the general compute
        # queue -- otherwise a manual trigger could run concurrently with a
        # scheduled data_fetch job and double up load against Yahoo.
        # chain_next=False: this is a standalone manual run, it should not
        # also cascade into GEX and Options.
        task = update_max_pain.apply_async(
            kwargs={"config_path": config_path, "wait_until": wait_until, "chain_next": False},
            queue=data_fetch_queue_for_market("US"),
        )
        
        return {
            "status": "success",
            "task_id": task.id,
            "message": "Update queued successfully. Check status with /max-pain/batch/{task_id}",
        }
    
    except Exception as e:
        logger.error(f"Failed to trigger max pain update: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger update: {str(e)}"
        )


@router.get("/batch/{batch_id}", response_model=BatchStatusResponse)
async def get_batch_status(
    batch_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """
    Get status of a max pain batch job.
    
    Args:
        batch_id: Batch ID from trigger-update response
    """
    batch = db.query(MaxPainBatch).filter(MaxPainBatch.id == batch_id).first()
    
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
    
    return {
        "batch_id": batch.id,
        "status": batch.status,
        "started_at": batch.started_at.isoformat(),
        "completed_at": batch.completed_at.isoformat() if batch.completed_at else None,
        "tickers_ok": batch.tickers_ok,
        "tickers_failed": batch.tickers_failed,
        "error_message": batch.error_message,
    }


@router.get("/health")
async def max_pain_health(db: Session = Depends(get_db)) -> dict:
    """Check if max pain service is available and has recent data"""
    try:
        # Check if we have recent data (within 48 hours)
        cutoff = datetime.utcnow() - timedelta(hours=48)
        recent_count = db.query(func.count(MaxPainSnapshot.id)).filter(
            MaxPainSnapshot.fetched_at >= cutoff
        ).scalar()
        
        # Get latest batch
        latest_batch = db.query(MaxPainBatch).order_by(
            desc(MaxPainBatch.completed_at)
        ).first()
        
        return {
            "status": "ok",
            "service": "max-pain",
            "has_recent_data": recent_count > 0,
            "recent_snapshot_count": recent_count,
            "latest_batch_id": latest_batch.id if latest_batch else None,
            "latest_batch_status": latest_batch.status if latest_batch else None,
            "latest_batch_completed": latest_batch.completed_at.isoformat() if latest_batch and latest_batch.completed_at else None,
        }
    
    except Exception as e:
        logger.exception("Error checking max pain health")
        return {
            "status": "error",
            "service": "max-pain",
            "error": str(e),
        }


def get_mock_dashboard() -> dict:
    """Return mock dashboard data for fallback/testing"""
    return {
        "tickers_ok": 100,
        "tickers_failed": 0,
        "avg_put_call_ratio": 0.74,
        "closest_to_max_pain": "COST",
        "generated_at": datetime.utcnow().isoformat(),
        "strike_range_pct": 20,
        "max_strikes": 30,
        "rows": [
            {
                "symbol": "AAPL",
                "company_name": "Apple Inc.",
                "status": "OK",
                "expiration": "2026-07-31",
                "call_oi": 167127,
                "put_oi": 83797,
                "put_call_ratio": 0.5,
                "max_pain": 320,
                "distance_pct": 4.1,
                "fetched_at": datetime.utcnow().isoformat(),
            },
            {
                "symbol": "MSFT",
                "company_name": "Microsoft Corporation",
                "status": "OK",
                "expiration": "2026-07-31",
                "call_oi": 66170,
                "put_oi": 53302,
                "put_call_ratio": 0.81,
                "max_pain": 385,
                "distance_pct": -0.9,
                "fetched_at": datetime.utcnow().isoformat(),
            },
        ],
    }
