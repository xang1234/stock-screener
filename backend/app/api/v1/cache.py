"""
Cache management API endpoints.

Provides endpoints for:
- Viewing cache statistics
- Triggering cache warming
- Invalidating caches
- Monitoring cache performance
- Cache health status (new unified endpoint)
- Smart refresh (new unified endpoint)
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks

from ...schemas.cache import (
    CacheHealthResponse,
    CacheInvalidateRequest,
    CacheStatsResponse,
    CacheWarmRequest,
    DashboardStatsResponse,
    ForceRefreshRequest,
    SmartRefreshRequest,
    SmartRefreshResponse,
    StalenessStatusResponse,
)
from ...schemas.common import TaskResponse
from ...services.cache_manager import CacheManager
from ...tasks.cache_tasks import (
    warm_spy_cache,
    warm_top_symbols,
    daily_cache_warmup,
    invalidate_cache as invalidate_cache_task,
    get_cache_stats as get_cache_stats_task,
    force_refresh_stale_intraday,
    smart_refresh_cache
)
from ...tasks.data_fetch_lock import DataFetchLock
from ...services.price_cache_service import PriceCacheService
from ...utils.market_hours import format_market_status, is_market_open
from ...database import SessionLocal
from ...models.stock import StockFundamental

router = APIRouter(prefix="/cache", tags=["cache"])


# Endpoints
@router.get("/stats", response_model=CacheStatsResponse)
async def get_cache_statistics():
    """
    Get comprehensive cache statistics.

    Returns:
        Cache statistics including Redis status, SPY cache, price cache, and memory usage
    """
    try:
        cache_manager = CacheManager()
        stats = cache_manager.get_cache_stats()
        return stats

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching cache stats: {str(e)}"
        )


@router.get("/market-status")
async def get_market_status():
    """
    Get current market status and hours.

    Returns:
        Market status information
    """
    return {
        "status": format_market_status(),
        "is_open": is_market_open(),
        "timestamp": "now"
    }


@router.post("/warm/spy")
async def warm_spy_benchmark_cache(background_tasks: BackgroundTasks):
    """
    Warm SPY benchmark cache.

    This endpoint triggers background warming of the SPY benchmark data.

    Returns:
        Task information
    """
    try:
        # Run task in background
        task = warm_spy_cache.delay()

        return TaskResponse(
            task_id=task.id,
            message="SPY cache warming task queued",
            status="queued"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error queuing SPY cache warming: {str(e)}"
        )


@router.post("/warm/symbols")
async def warm_symbol_cache(
    request: CacheWarmRequest,
    background_tasks: BackgroundTasks
):
    """
    Warm cache for specific symbols.

    Args:
        request: Cache warming request with symbols or count

    Returns:
        Task information
    """
    try:
        # Run task in background
        task = warm_top_symbols.delay(
            symbols=request.symbols,
            count=request.count
        )

        # Determine message based on parameters
        if request.symbols:
            symbol_count = len(request.symbols)
            message = f"Symbol cache warming task queued for {symbol_count} symbols"
        elif request.count:
            message = f"Symbol cache warming task queued for top {request.count} symbols"
        else:
            message = "Symbol cache warming task queued for ALL active stocks in universe"

        return TaskResponse(
            task_id=task.id,
            message=message,
            status="queued"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error queuing symbol cache warming: {str(e)}"
        )


@router.post("/warm/all")
async def warm_all_caches(background_tasks: BackgroundTasks):
    """
    Warm all caches (SPY + ALL active stocks).

    This triggers the daily cache warmup task which warms:
    - SPY benchmark cache
    - ALL active symbols in the stock universe

    Returns:
        Task information
    """
    try:
        # Run daily warmup task in background
        task = daily_cache_warmup.delay()

        return TaskResponse(
            task_id=task.id,
            message="Full cache warming task queued for ALL active stocks in universe",
            status="queued"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error queuing full cache warming: {str(e)}"
        )


@router.delete("/invalidate")
async def invalidate_caches(request: CacheInvalidateRequest):
    """
    Invalidate cache for a symbol or all caches.

    Args:
        request: Invalidation request with optional symbol

    Returns:
        Invalidation result
    """
    try:
        # Run invalidation task
        task = invalidate_cache_task.delay(symbol=request.symbol)

        message = f"Cache invalidation queued for {request.symbol}" if request.symbol else "All caches invalidation queued"

        return TaskResponse(
            task_id=task.id,
            message=message,
            status="queued"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error invalidating cache: {str(e)}"
        )


@router.get("/hit-rate")
async def get_cache_hit_rate():
    """
    Get cache hit rate statistics.

    Returns:
        Cache hit rate percentage
    """
    try:
        cache_manager = CacheManager()
        hit_rate = cache_manager.get_cache_hit_rate()

        return {
            "hit_rate": hit_rate,
            "hit_rate_str": f"{hit_rate:.1f}%" if hit_rate is not None else "N/A",
            "available": hit_rate is not None
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating hit rate: {str(e)}"
        )


@router.get("/staleness-status", response_model=StalenessStatusResponse)
async def get_staleness_status():
    """
    Get intraday data staleness status.

    Returns information about symbols with stale intraday data:
    - Data fetched during market hours that is now stale (after market close)
    - Count of affected symbols
    - Current market status

    Use this to determine if force-refresh is needed.
    """
    try:
        price_cache = PriceCacheService.get_instance()
        status = price_cache.get_staleness_status()
        return StalenessStatusResponse(**status)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error checking staleness status: {str(e)}"
        )


@router.post("/force-refresh")
async def force_refresh_stale_data(
    request: ForceRefreshRequest,
    background_tasks: BackgroundTasks
):
    """
    Force refresh stale intraday price data.

    This endpoint triggers a background task to refresh symbols that have
    stale intraday data (data fetched during market hours that is now
    outdated after market close).

    Args:
        request:
            - symbols: Optional list of symbols to refresh
            - refresh_all: If True, refresh ALL cached symbols (not just stale ones)

    Returns:
        Task information with task_id for tracking
    """
    try:
        price_cache = PriceCacheService.get_instance()

        # Get symbols to refresh
        if request.symbols:
            symbols = request.symbols
            message = f"Force refresh queued for {len(symbols)} symbols"
        elif request.refresh_all:
            # Get ALL cached symbols from Redis
            symbols = price_cache.get_all_cached_symbols()
            if not symbols:
                return {
                    "task_id": None,
                    "message": "No cached symbols found",
                    "status": "skipped",
                    "symbols_count": 0
                }
            message = f"Force refresh queued for ALL {len(symbols)} cached symbols"
        else:
            # Get only stale symbols
            symbols = price_cache.get_stale_intraday_symbols()

            if not symbols:
                return {
                    "task_id": None,
                    "message": "No stale intraday data detected",
                    "status": "skipped",
                    "symbols_count": 0
                }

            message = f"Force refresh queued for {len(symbols)} stale symbols"

        # Queue the force refresh task
        task = force_refresh_stale_intraday.delay(symbols=symbols)

        return TaskResponse(
            task_id=task.id,
            message=message,
            status="queued"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error queuing force refresh: {str(e)}"
        )


@router.get("/symbol/{symbol}")
async def get_symbol_cache_info(symbol: str):
    """
    Get cache information for a specific symbol.

    Args:
        symbol: Stock symbol

    Returns:
        Cache information for the symbol
    """
    try:
        cache_manager = CacheManager()
        stats = cache_manager.price_cache.get_cache_stats(symbol)

        return stats

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching cache info for {symbol}: {str(e)}"
        )


@router.get("/health", response_model=CacheHealthResponse)
async def get_cache_health():
    """
    Get cache health status with unified state indicator.

    This is the NEW primary endpoint for cache status. It uses SPY as a proxy
    for overall cache health (O(1) check) and includes warmup metadata for
    detecting partial failures.

    Returns one of 6 states:
    - fresh: Cache is up to date (SPY has expected date + last warmup complete)
    - updating: Refresh task is currently running
    - stuck: Task running but no progress for >30 minutes
    - partial: Last warmup incomplete (some symbols failed)
    - stale: SPY missing expected trading date
    - error: Redis unavailable or other error

    Returns:
        CacheHealthResponse with status, dates, message, and task info
    """
    try:
        price_cache = PriceCacheService.get_instance()
        health = price_cache.get_cache_health_status()
        return CacheHealthResponse(**health)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error checking cache health: {str(e)}"
        )


@router.post("/refresh", response_model=SmartRefreshResponse)
async def smart_refresh(request: SmartRefreshRequest):
    """
    Trigger smart cache refresh with prioritized fetching.

    This is the NEW unified refresh endpoint that replaces the confusing
    split between /warm/all and /force-refresh.

    Modes:
    - auto (default): Refresh all currently cached symbols
    - full: Refresh entire universe (~5000 symbols, takes ~2 hours)

    Key features:
    - Always warms SPY first (required for RS calculations)
    - Fetches symbols in market cap order (high cap first)
    - Prevents double-refresh (returns existing task info if running)

    Returns:
        SmartRefreshResponse with status and task_id
    """
    try:
        # Check if task is already running
        lock = DataFetchLock.get_instance()
        running = lock.get_current_task()

        if running:
            return SmartRefreshResponse(
                status="already_running",
                task_id=running.get("task_id"),
                message=f"Refresh already in progress ({running.get('task_name')})"
            )

        # Queue smart refresh task
        task = smart_refresh_cache.delay(mode=request.mode)

        mode_desc = "cached symbols" if request.mode == "auto" else "entire universe (~2 hours)"
        return SmartRefreshResponse(
            status="queued",
            task_id=task.id,
            message=f"Smart refresh started for {mode_desc}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error queuing smart refresh: {str(e)}"
        )


@router.post("/force-cancel")
async def force_cancel_refresh():
    """
    Force-cancel a stuck refresh task.

    Use this when a task appears stuck (no progress for >30 minutes).
    Releases the lock so a new refresh can be started.

    Safety checks:
    - Requires task to have no heartbeat update for >30 minutes
    - Won't cancel actively progressing tasks

    Returns:
        Status and message
    """
    try:
        lock = DataFetchLock.get_instance()
        running = lock.get_current_task()

        if not running:
            return {
                "status": "no_task",
                "message": "No task is currently running"
            }

        # Check if task is actually stuck
        price_cache = PriceCacheService.get_instance()
        minutes = price_cache._get_minutes_since_heartbeat()

        if minutes is not None and minutes < 30:
            return {
                "status": "active",
                "message": f"Task is active (last progress {int(minutes)} min ago). Cannot cancel.",
                "task_id": running.get("task_id"),
                "task_name": running.get("task_name")
            }

        # Force release the lock
        lock.force_release()

        # Clear heartbeat
        price_cache.clear_warmup_heartbeat()

        return {
            "status": "cancelled",
            "message": f"Task {running.get('task_name')} force-cancelled",
            "task_id": running.get("task_id")
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error force-cancelling task: {str(e)}"
        )


@router.get("/dashboard-stats", response_model=DashboardStatsResponse)
async def get_dashboard_cache_statistics():
    """
    Get comprehensive cache statistics for UI dashboard.

    This endpoint aggregates cache health metrics from all cache services:
    - Fundamentals cache (Redis + DB)
    - Price/technical data cache (SPY + symbols)
    - Market status

    Returns:
        Comprehensive dashboard statistics including freshness, hit rates,
        and last update times for all cache types
    """
    try:
        from datetime import datetime
        from ...models.stock_universe import StockUniverse

        # Get fundamentals cache stats using efficient SQL aggregate queries
        db = SessionLocal()
        try:
            from sqlalchemy import func
            from datetime import timedelta

            # Get total active stocks count
            total_stocks = db.query(func.count(StockUniverse.id)).filter(
                StockUniverse.is_active == True
            ).scalar()

            # Get fundamentals counts using efficient SQL aggregates
            seven_days_ago = datetime.now() - timedelta(days=7)

            # Total cached (all rows in stock_fundamentals)
            cached_count = db.query(func.count(StockFundamental.id)).scalar()

            # Fresh count (updated within 7 days)
            fresh_count = db.query(func.count(StockFundamental.id)).filter(
                StockFundamental.updated_at >= seven_days_ago
            ).scalar()

            # Stale count
            stale_count = cached_count - fresh_count

            fundamentals_stats = {
                'total_stocks': total_stocks,
                'cached_count': cached_count,
                'fresh_count': fresh_count,
                'stale_count': stale_count,
                'hit_rate': round(cached_count / total_stocks * 100, 1) if total_stocks > 0 else 0
            }

            # Get most recent fundamental update from database
            latest_fundamental = db.query(StockFundamental).order_by(
                StockFundamental.updated_at.desc()
            ).first()
            last_fundamental_update = latest_fundamental.updated_at.isoformat() if latest_fundamental else None

        finally:
            db.close()

        # Get price cache stats
        cache_manager = CacheManager()
        price_stats = cache_manager.get_cache_stats()

        # Extract SPY cache info (corrected key names)
        spy_cache = price_stats.get('spy_cache', {})
        spy_cached = spy_cache.get('2y_cached', False)
        spy_ttl_seconds = spy_cache.get('2y_ttl', 0)
        spy_ttl = (spy_ttl_seconds / 3600) if spy_ttl_seconds else 0  # Convert seconds to hours

        # Get market status
        market_status = {
            "is_open": is_market_open(),
            "status": format_market_status(),
            "next_open": "N/A"  # Could be enhanced with next market open time
        }

        # Build dashboard response
        dashboard_stats = {
            "fundamentals": {
                "total_stocks": fundamentals_stats.get('total_stocks', 0),
                "cached_count": fundamentals_stats.get('cached_count', 0),
                "fresh_count": fundamentals_stats.get('fresh_count', 0),
                "stale_count": fundamentals_stats.get('stale_count', 0),
                "last_update": last_fundamental_update,
                "hit_rate": fundamentals_stats.get('hit_rate', 0)
            },
            "prices": {
                "spy_cached": spy_cached,
                "spy_last_update": "N/A",  # Could be enhanced to extract last update date
                "spy_ttl_hours": round(spy_ttl, 1),
                "total_symbols_cached": price_stats.get('price_cache', {}).get('symbols_cached', 0),
                "last_warmup": "N/A"  # Could track last warmup timestamp
            },
            "market_status": market_status,
            "timestamp": datetime.now().isoformat()
        }

        return DashboardStatsResponse(**dashboard_stats)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching dashboard stats: {str(e)}"
        )
