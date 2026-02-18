"""
Fundamentals data management API endpoints.

Provides endpoints for:
- Triggering fundamental data refresh for all stocks
- Triggering fundamental data refresh for specific symbols
- Initial cache population
- Viewing fundamentals cache statistics
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from ...schemas.common import TaskResponse
from ...schemas.fundamentals import FundamentalsCacheStats
from ...tasks.fundamentals_tasks import (
    refresh_all_fundamentals,
    refresh_symbol_fundamentals,
    populate_initial_cache,
    get_cache_stats as get_fundamentals_cache_stats
)

router = APIRouter(prefix="/fundamentals", tags=["fundamentals"])


# Endpoints
@router.post("/refresh/all")
async def trigger_full_fundamentals_refresh():
    """
    Trigger full refresh of fundamental data for all active stocks.

    This queues a Celery task that will refresh fundamentals for all ~7,000
    active stocks in the universe using finviz.

    Expected duration: ~1 hour at 0.5s rate limit.

    Returns:
        Task information including task_id for tracking progress
    """
    try:
        task = refresh_all_fundamentals.delay()
        message = "Fundamentals refresh queued. Expected duration: ~1 hour for ~7,000 stocks."

        return TaskResponse(
            task_id=task.id,
            message=message,
            status="queued"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error queuing fundamentals refresh: {str(e)}"
        )


@router.post("/refresh/symbol/{symbol}")
async def trigger_symbol_fundamentals_refresh(symbol: str):
    """
    Trigger fundamental data refresh for a specific stock symbol.

    This is useful for on-demand updates when you need fresh data for
    a particular stock without waiting for the weekly refresh.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "TSLA")

    Returns:
        Task information including task_id for tracking progress
    """
    try:
        # Validate symbol format (basic check)
        symbol = symbol.upper().strip()
        if not symbol or len(symbol) > 10:
            raise HTTPException(
                status_code=400,
                detail="Invalid symbol format"
            )

        # Queue the task
        task = refresh_symbol_fundamentals.delay(symbol)

        return TaskResponse(
            task_id=task.id,
            message=f"Fundamentals refresh task queued for {symbol}",
            status="queued"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error queuing fundamentals refresh for {symbol}: {str(e)}"
        )


@router.post("/populate")
async def trigger_initial_cache_population(
    limit: Optional[int] = Query(None, description="Limit number of stocks to populate (for testing)"),
    use_hybrid: bool = Query(True, description="Use optimized hybrid fetching")
):
    """
    Trigger initial cache population for fundamental data.

    This is typically run once immediately after deployment to populate
    the cache with fundamental data for all active stocks.

    **Hybrid mode (default)**: ~1 hour for full universe
    **Legacy mode**: ~1 hour for full universe (at 0.5s rate limit)

    Args:
        limit: Optional limit on number of stocks to populate (useful for testing)
        use_hybrid: Use optimized hybrid fetching (default True)

    Returns:
        Task information including task_id and estimated duration
    """
    try:
        estimated_stocks = limit if limit else 7000

        if use_hybrid:
            # Use hybrid task for initial population
            if limit:
                task = refresh_symbols_hybrid.delay(
                    symbols=None,  # Will need to fetch symbols in task
                    include_finviz=True
                )
            else:
                task = refresh_all_fundamentals_hybrid.delay(include_finviz=True)
            estimated_minutes = round(estimated_stocks * 0.5 / 60 + 35, 0)  # ~35 min base + finviz time
        else:
            # Use legacy task
            task = populate_initial_cache.delay(limit=limit)
            estimated_minutes = round(estimated_stocks * 0.5 / 60, 0)  # 0.5 seconds per stock

        message = f"Initial cache population task queued"
        if limit:
            message += f" (testing mode: {limit} stocks, ~{estimated_minutes} minutes)"
        else:
            message += f" (~{estimated_stocks} stocks, ~{estimated_minutes} minutes)"

        return TaskResponse(
            task_id=task.id,
            message=message,
            status="queued"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error queuing initial cache population: {str(e)}"
        )


@router.get("/stats", response_model=FundamentalsCacheStats)
async def get_fundamentals_cache_statistics():
    """
    Get fundamental data cache statistics.

    This endpoint checks a sample of stocks (first 100 active stocks) to
    determine cache health, hit rates, and freshness.

    Returns:
        Cache statistics including:
        - Total stocks checked
        - Redis/DB cache hit counts
        - Fresh vs stale data counts
        - Cache hit rates
        - Last refresh date
    """
    try:
        from ...database import SessionLocal
        from ...models.stock_universe import StockUniverse
        from ...services.fundamentals_cache_service import FundamentalsCacheService
        from datetime import datetime

        db = SessionLocal()

        try:
            # Get sample of active stocks (first 100)
            universe_stocks = db.query(StockUniverse).filter(
                StockUniverse.is_active == True
            ).limit(100).all()

            symbols = [s.symbol for s in universe_stocks]
            cache = FundamentalsCacheService.get_instance()

            # Collect stats
            total = len(symbols)
            redis_cached = 0
            db_cached = 0
            fresh = 0
            stale = 0

            for symbol in symbols:
                stats = cache.get_cache_stats(symbol)

                if stats['redis_cached']:
                    redis_cached += 1
                if stats['db_cached']:
                    db_cached += 1
                if stats['age_days'] is not None:
                    if stats['age_days'] <= 7:
                        fresh += 1
                    else:
                        stale += 1

            # Get most recent update
            from ...models.stock import StockFundamental
            latest = db.query(StockFundamental).order_by(
                StockFundamental.updated_at.desc()
            ).first()

            return FundamentalsCacheStats(
                total_checked=total,
                redis_cached=redis_cached,
                db_cached=db_cached,
                fresh=fresh,
                stale=stale,
                redis_hit_rate=round(redis_cached / total * 100, 1) if total > 0 else 0,
                db_hit_rate=round(db_cached / total * 100, 1) if total > 0 else 0,
                last_refresh_date=latest.updated_at.isoformat() if latest else None,
                timestamp=datetime.now().isoformat()
            )

        finally:
            db.close()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching fundamentals cache stats: {str(e)}"
        )
