"""
Main FastAPI application entry point.
"""
import asyncio
import json
import os

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


def run_theme_pipeline_state_migration():
    """
    Run idempotent Theme pipeline-state schema migration on startup.

    Creates content_item_pipeline_state and supporting indexes used by
    pipeline-scoped extraction/reprocessing orchestration.
    """
    from .db_migrations.theme_pipeline_state_migration import migrate_theme_pipeline_state

    try:
        migrate_theme_pipeline_state(engine)
    except Exception as e:
        print(f"Warning: Theme pipeline-state migration failed (non-fatal): {e}")


def run_theme_cluster_identity_migration():
    """
    Run idempotent Theme cluster identity migration on startup.

    Adds canonical_key/display_name and enforces pipeline-scoped uniqueness
    for theme cluster identity.
    """
    from .db_migrations.theme_cluster_identity_migration import migrate_theme_cluster_identity

    migrate_theme_cluster_identity(engine)


def run_theme_aliases_migration():
    """
    Run idempotent Theme alias schema migration on startup.

    Creates theme_aliases table and indexes used for extraction-time
    exact alias matching and alias quality analytics.
    """
    from .db_migrations.theme_aliases_migration import migrate_theme_aliases

    migrate_theme_aliases(engine)


def run_theme_match_decision_migration():
    """
    Run idempotent theme mention match-decision migration on startup.

    Adds decision telemetry fields used by extraction-time matching
    observability and threshold governance.
    """
    from .db_migrations.theme_match_decision_migration import migrate_theme_match_decision

    migrate_theme_match_decision(engine)


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

    # Verify WAL mode is active (critical for concurrent writers in Docker)
    if "sqlite" in settings.database_url:
        with engine.connect() as conn:
            journal_mode = conn.execute(text("PRAGMA journal_mode")).scalar()
            print(f"SQLite journal mode: {journal_mode}")
            if journal_mode != "wal":
                print(f"WARNING: Expected WAL journal mode but got '{journal_mode}'. "
                      "Concurrent writes may cause corruption.")

        # Check WAL file size — large WAL indicates checkpoint failure or concurrent access
        db_path = settings.database_url.replace("sqlite:///", "")
        wal_path = db_path + "-wal"
        if os.path.exists(wal_path):
            wal_mb = os.path.getsize(wal_path) / (1024 * 1024)
            if wal_mb > 100:
                print(f"WARNING: WAL file is {wal_mb:.0f}MB — may indicate "
                      "concurrent access or checkpoint failure")

    # Run schema migrations (idempotent — safe on every startup)
    run_universe_migration()
    run_feature_store_migration()
    run_scan_feature_run_migration()
    run_theme_pipeline_state_migration()
    run_theme_cluster_identity_migration()
    run_theme_aliases_migration()
    run_theme_match_decision_migration()

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
    """Readiness probe - checks database and Redis connectivity.

    Uses sqlite_master count (sub-ms) instead of PRAGMA quick_check (1-3s)
    to avoid blocking the event loop. Redis is a soft dependency — its
    absence degrades the service but doesn't make it unhealthy.
    """
    checks = {}
    healthy = True

    # DB check — offloaded to thread pool (non-blocking)
    try:
        def _check_db():
            with engine.connect() as conn:
                count = conn.execute(text("SELECT count(*) FROM sqlite_master")).scalar()
                return count > 0

        if await asyncio.to_thread(_check_db):
            checks["database"] = "ok"
        else:
            checks["database"] = "error: no tables found"
            healthy = False
    except Exception as e:
        checks["database"] = f"error: {type(e).__name__}"
        healthy = False

    # Redis check — soft dependency (degraded, not unhealthy)
    try:
        def _check_redis():
            client = get_redis_client()
            return client and client.ping()

        if await asyncio.to_thread(_check_redis):
            checks["redis"] = "ok"
        else:
            checks["redis"] = "warning: unavailable"
    except Exception as e:
        checks["redis"] = f"warning: {type(e).__name__}"

    status_code = 200 if healthy else 503
    status_label = "ok" if healthy else "unhealthy"
    if healthy and checks.get("redis", "").startswith("warning"):
        status_label = "degraded"

    return JSONResponse(
        content={"status": status_label, "checks": checks},
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
