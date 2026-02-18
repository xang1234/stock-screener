"""
Main FastAPI application entry point.
"""
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from sqlalchemy import text

from .config import settings
from .database import init_db, engine
from .services.redis_pool import get_redis_client


def run_universe_migration():
    """
    Run idempotent universe schema migration on startup.

    Adds structured universe columns (universe_key, universe_type, etc.)
    to the scans table and backfills existing rows. Replaces the old
    destructive cleanup that deleted scans with 'nyse', 'nasdaq', 'sp500'.
    """
    from .db_migrations.universe_migration import migrate_scan_universe_schema_and_backfill

    try:
        migrate_scan_universe_schema_and_backfill(engine)
    except Exception as e:
        print(f"Warning: Universe migration failed (non-fatal): {e}")


def run_feature_store_migration():
    """
    Run idempotent Feature Store schema migration on startup.

    Creates the 4 Feature Store tables (feature_runs, feature_run_universe_symbols,
    stock_feature_daily, feature_run_pointers) if they don't already exist.
    """
    from .db_migrations.feature_store_migration import migrate_feature_store_tables

    try:
        migrate_feature_store_tables(engine)
    except Exception as e:
        print(f"Warning: Feature Store migration failed (non-fatal): {e}")


def run_scan_feature_run_migration():
    """Add feature_run_id FK column to scans table."""
    from .db_migrations.scan_feature_run_migration import migrate_scan_feature_run_id

    try:
        migrate_scan_feature_run_id(engine)
    except Exception as e:
        print(f"Warning: Scan feature_run_id migration failed (non-fatal): {e}")


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

    # Run schema migrations (idempotent — safe on every startup)
    run_universe_migration()
    run_feature_store_migration()
    run_scan_feature_run_migration()

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


@app.get("/livez")
async def liveness():
    """Liveness probe - zero dependencies, confirms process is responsive."""
    return {"status": "ok"}


@app.get("/readyz")
async def readiness():
    """Readiness probe - checks database and Redis connectivity."""
    checks = {}
    healthy = True

    # DB: verify connectivity with a lightweight query
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {type(e).__name__}"
        healthy = False

    # Redis: verify connectivity via ping
    try:
        client = get_redis_client()
        if client and client.ping():
            checks["redis"] = "ok"
        else:
            checks["redis"] = "error: unavailable"
            healthy = False
    except Exception as e:
        checks["redis"] = f"error: {type(e).__name__}"
        healthy = False

    status_code = 200 if healthy else 503
    return JSONResponse(
        content={"status": "ok" if healthy else "degraded", "checks": checks},
        status_code=status_code,
    )


@app.get("/health")
async def health_check():
    """Deprecated health check - use /readyz instead."""
    result = await readiness()
    body = json.loads(result.body)
    body["deprecated"] = True
    body["use_instead"] = "/readyz"
    return JSONResponse(content=body, status_code=result.status_code)


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
