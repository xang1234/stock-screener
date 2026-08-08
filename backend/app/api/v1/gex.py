"""Gamma Exposure (GEX) dashboard API endpoints with DB-backed persistence."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, and_, desc
from sqlalchemy.orm import Session

from ...database import get_db
from ...models.gex import GexBatch, GexSnapshot
from ...models.stock_universe import StockUniverse
from ...services.options_history_service import Timeframe, fetch_gex_history
from ...services.symbol_format import normalize_symbol
from ...tasks.gex_tasks import update_gex
from ...tasks.market_queues import data_fetch_queue_for_market

logger = logging.getLogger(__name__)
router = APIRouter()


class GexRowResponse(BaseModel):
    """Single ticker row in GEX dashboard."""
    symbol: str
    company_name: str | None
    status: str
    expiration: str | None
    spot_price: float | None
    call_gex: float | None
    put_gex: float | None
    total_gex: float | None
    flip_level: float | None
    distance_to_flip_pct: float | None
    fetched_at: str


class GexDashboardResponse(BaseModel):
    """Response model for GEX dashboard."""
    tickers_ok: int
    tickers_failed: int
    avg_total_gex: float | None
    closest_to_flip: str | None
    generated_at: str
    strike_range_pct: float
    max_strikes: int | None
    total_rows: int
    total_pages: int
    current_page: int
    page_size: int
    returned_rows: int
    applied_filters: dict[str, str | int | None]
    rows: list[GexRowResponse]


@router.get("/dashboard", response_model=GexDashboardResponse)
async def get_gex_dashboard(
    db: Session = Depends(get_db),
    hours_back: int = Query(default=48, ge=1, le=720),
    exchange: str | None = Query(default=None, description="Optional exchange filter (e.g. NASDAQ, NYSE)"),
    symbol: str | None = Query(default=None, description="Optional symbol filter"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=10, le=500),
) -> dict:
    """Return current DB-backed GEX dashboard payload."""
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
                GexSnapshot.ticker.label("ticker"),
                func.max(GexSnapshot.fetched_at).label("latest_fetched_at"),
            )
            .filter(GexSnapshot.fetched_at >= cutoff_time)
            .group_by(GexSnapshot.ticker)
            .subquery()
        )

        snapshot_query = (
            db.query(GexSnapshot)
            .join(
                latest_per_ticker,
                and_(
                    GexSnapshot.ticker == latest_per_ticker.c.ticker,
                    GexSnapshot.fetched_at == latest_per_ticker.c.latest_fetched_at,
                ),
            )
        )

        if normalized_exchange:
            snapshot_query = snapshot_query.join(
                StockUniverse,
                StockUniverse.symbol == GexSnapshot.ticker,
            ).filter(func.upper(StockUniverse.exchange) == normalized_exchange)

        if normalized_symbol:
            snapshot_query = snapshot_query.filter(GexSnapshot.ticker == normalized_symbol)

        total_rows = snapshot_query.count()
        total_pages = max((total_rows + page_size - 1) // page_size, 1)
        safe_page = min(page, total_pages)
        offset = (safe_page - 1) * page_size

        rows = (
            snapshot_query
            .order_by(GexSnapshot.ticker.asc())
            .offset(offset)
            .limit(page_size)
            .all()
        )

        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"No GEX data available in last {hours_back} hours. Trigger /v1/gex/trigger-update first.",
            )

        ok_rows = [row for row in rows if row.status == "OK"]
        failed_rows = [row for row in rows if row.status != "OK"]
        gex_values = [row.total_gex for row in ok_rows if row.total_gex is not None]
        avg_total_gex = (sum(gex_values) / len(gex_values)) if gex_values else None

        closest_to_flip = None
        closest_distance = None
        for row in ok_rows:
            distance = row.distance_to_flip_pct
            if distance is None:
                continue
            if closest_distance is None or abs(distance) < abs(closest_distance):
                closest_distance = distance
                closest_to_flip = row.ticker

        latest_batch = db.query(GexBatch).order_by(desc(GexBatch.completed_at)).first()
        generated_at = (
            latest_batch.completed_at.isoformat()
            if latest_batch and latest_batch.completed_at
            else rows[0].fetched_at.isoformat()
        )

        return {
            "tickers_ok": len(ok_rows),
            "tickers_failed": len(failed_rows),
            "avg_total_gex": avg_total_gex,
            "closest_to_flip": closest_to_flip,
            "generated_at": generated_at,
            "strike_range_pct": latest_batch.strike_range_pct if latest_batch else 20.0,
            "max_strikes": latest_batch.max_strikes if latest_batch else None,
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
                    "symbol": row.ticker,
                    "company_name": row.company_name,
                    "status": row.status,
                    "expiration": row.expiration.isoformat() if row.expiration else None,
                    "spot_price": row.spot_price,
                    "call_gex": row.call_gex,
                    "put_gex": row.put_gex,
                    "total_gex": row.total_gex,
                    "flip_level": row.flip_level,
                    "distance_to_flip_pct": row.distance_to_flip_pct,
                    "fetched_at": row.fetched_at.isoformat(),
                }
                for row in rows
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error fetching GEX dashboard")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/history")
async def get_gex_history(
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
    """GEX time series for one ticker over a timeframe window.

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
        return fetch_gex_history(
            db, symbol=normalized_symbol, timeframe=timeframe, start=start, end=end,
            expiration=expiration,
        )
    except Exception:
        logger.exception("Error fetching GEX history for %s", normalized_symbol)
        raise HTTPException(status_code=500, detail="Failed to fetch GEX history")


@router.post("/trigger-update")
async def trigger_gex_update(
    strike_range_pct: float = Query(default=20.0, ge=5, le=80),
    max_strikes: int | None = Query(default=None, ge=5, le=300),
) -> dict:
    """Trigger background GEX update task."""
    try:
        # Explicit queue routing (not just .delay()) so this lands on the
        # single-concurrency data_fetch worker instead of the general compute
        # queue -- otherwise a manual trigger could run concurrently with a
        # scheduled data_fetch job and double up load against Yahoo.
        # chain_next=False: this is a standalone manual run, it should not
        # also cascade into the options update.
        task = update_gex.apply_async(
            kwargs={"strike_range_pct": strike_range_pct, "max_strikes": max_strikes, "chain_next": False},
            queue=data_fetch_queue_for_market("US"),
        )
        return {
            "status": "success",
            "task_id": task.id,
            "message": "Full-universe GEX update queued successfully.",
        }
    except Exception as exc:
        logger.exception("Failed to trigger GEX update")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class GexBatchStatusResponse(BaseModel):
    """Status payload for GEX batch runs."""

    batch_id: str
    status: str
    started_at: str
    completed_at: str | None
    tickers_total: int | None
    tickers_ok: int | None
    tickers_failed: int | None
    error_message: str | None


@router.get("/batch/{batch_id}", response_model=GexBatchStatusResponse)
async def get_batch_status(batch_id: str, db: Session = Depends(get_db)) -> dict:
    """Get status of a GEX batch run."""
    batch = db.query(GexBatch).filter(GexBatch.id == batch_id).first()
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

    return {
        "batch_id": batch.id,
        "status": batch.status,
        "started_at": batch.started_at.isoformat(),
        "completed_at": batch.completed_at.isoformat() if batch.completed_at else None,
        "tickers_total": batch.tickers_total,
        "tickers_ok": batch.tickers_ok,
        "tickers_failed": batch.tickers_failed,
        "error_message": batch.error_message,
    }


@router.get("/health")
async def gex_health(db: Session = Depends(get_db)) -> dict:
    """Check if GEX service has recent persisted data."""
    try:
        cutoff = datetime.utcnow() - timedelta(hours=48)
        recent_count = db.query(func.count(GexSnapshot.id)).filter(GexSnapshot.fetched_at >= cutoff).scalar()

        latest_batch = db.query(GexBatch).order_by(desc(GexBatch.completed_at)).first()

        return {
            "status": "ok",
            "service": "gex",
            "has_recent_data": recent_count > 0,
            "recent_snapshot_count": recent_count,
            "latest_batch_id": latest_batch.id if latest_batch else None,
            "latest_batch_status": latest_batch.status if latest_batch else None,
            "latest_batch_completed": (
                latest_batch.completed_at.isoformat() if latest_batch and latest_batch.completed_at else None
            ),
        }
    except Exception as exc:
        logger.exception("Error checking GEX health")
        return {
            "status": "error",
            "service": "gex",
            "error": str(exc),
        }
