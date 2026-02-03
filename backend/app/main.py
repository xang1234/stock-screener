"""
Main FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .config import settings
from .database import init_db


def cleanup_invalid_universe_scans():
    """
    One-time cleanup of scans with invalid universes.

    Removes scans with universes that are no longer supported
    (nyse, nasdaq, sp500). Only "test", "all", and "custom" universes are allowed.
    """
    from .database import SessionLocal
    from .models.scan_result import Scan, ScanResult

    db = SessionLocal()
    try:
        invalid_universes = ["nyse", "nasdaq", "sp500"]
        total_deleted = 0

        for universe in invalid_universes:
            scans = db.query(Scan).filter(Scan.universe == universe).all()
            for scan in scans:
                # Delete results first (foreign key constraint)
                deleted_results = db.query(ScanResult).filter(
                    ScanResult.scan_id == scan.scan_id
                ).delete()
                db.delete(scan)
                print(f"Deleted scan {scan.scan_id} with invalid universe '{universe}' ({deleted_results} results)")
                total_deleted += 1

        if total_deleted > 0:
            db.commit()
            print(f"Cleaned up {total_deleted} scans with invalid universes")
        else:
            print("No scans with invalid universes found")

    except Exception as e:
        print(f"Error cleaning up invalid universe scans: {e}")
        db.rollback()
    finally:
        db.close()


async def trigger_gapfill_on_startup():
    """
    Trigger gap-fill as a background Celery task.

    This is non-blocking - the server starts immediately while
    the gap-fill runs in the background via Celery.

    Uses a small delay to let any scheduled tasks get priority,
    and explicitly routes to the data_fetch queue for serialization.
    """
    import asyncio

    try:
        from .tasks.group_rank_tasks import gapfill_group_rankings

        max_days = getattr(settings, 'group_rank_gapfill_max_days', 365)
        startup_delay = getattr(settings, 'data_fetch_startup_delay', 5)

        # Small delay to let scheduled tasks get priority
        await asyncio.sleep(startup_delay)

        # Dispatch to data_fetch queue (serialized with other data-fetching tasks)
        task = gapfill_group_rankings.apply_async(
            kwargs={'max_days': max_days},
            queue='data_fetch'
        )
        print(f"IBD Group Rankings gap-fill task dispatched: {task.id}")
        print(f"  Looking back {max_days} days for gaps")
        print(f"  Routed to 'data_fetch' queue for serialization")

    except Exception as e:
        # Don't block startup if gap-fill dispatch fails
        print(f"Warning: Failed to dispatch gap-fill task: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    print("Starting Stock Scanner API...")
    print(f"Database: {settings.database_url}")
    print(f"CORS origins: {settings.cors_origins_list}")

    # Initialize database
    init_db()
    print("Database initialized")

    # Cleanup scans with invalid universes (optional one-time migration)
    if getattr(settings, "invalid_universe_cleanup_enabled", False):
        cleanup_invalid_universe_scans()

    # Trigger non-blocking gap-fill for IBD group rankings
    if getattr(settings, 'group_rank_gapfill_enabled', True):
        await trigger_gapfill_on_startup()

    yield

    # Shutdown
    print("Shutting down Stock Scanner API...")


# Create FastAPI application
app = FastAPI(
    title="Stock Scanner API",
    description="CANSLIM + Minervini stock scanner with industry group analysis",
    version="0.1.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Stock Scanner API",
        "version": "0.1.0",
        "description": "CANSLIM + Minervini stock scanner",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with yfinance connectivity test."""
    import yfinance as yf

    # Test yfinance connectivity with a known stock
    yfinance_status = "unknown"
    try:
        test_ticker = yf.Ticker("SPY")
        test_info = test_ticker.info
        if test_info and len(test_info) > 3:
            yfinance_status = "ok"
        else:
            yfinance_status = "degraded"
    except Exception as e:
        yfinance_status = f"error: {type(e).__name__}"

    return {
        "status": "healthy",
        "database": "connected",
        "version": "0.1.0",
        "yfinance": yfinance_status
    }


# Include API routers
from .api.v1.router import router as api_router
app.include_router(api_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
