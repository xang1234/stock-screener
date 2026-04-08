"""
Main FastAPI application entry point.
"""
import asyncio
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from .config import settings
from .database import init_db, engine
from .infra.db.portability import table_exists
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


def run_theme_lifecycle_migration():
    """
    Run idempotent theme lifecycle migration on startup.

    Adds lifecycle state/timestamp fields and transition audit table.
    """
    from .db_migrations.theme_lifecycle_migration import migrate_theme_lifecycle

    migrate_theme_lifecycle(engine)


def run_theme_relationships_migration():
    """
    Run idempotent theme relationships migration on startup.

    Creates non-destructive semantic relationship edges between theme clusters.
    """
    from .db_migrations.theme_relationships_migration import migrate_theme_relationships

    migrate_theme_relationships(engine)


def run_theme_merge_suggestion_safety_migration():
    """
    Run idempotent merge-suggestion safety migration on startup.

    Adds canonical pair identity columns and strict approval idempotency fields.
    """
    from .db_migrations.theme_merge_suggestion_safety_migration import migrate_theme_merge_suggestion_safety

    migrate_theme_merge_suggestion_safety(engine)


def run_universe_lifecycle_migration():
    """Run idempotent stock universe lifecycle migration on startup."""
    from .db_migrations.universe_lifecycle_migration import migrate_universe_lifecycle

    migrate_universe_lifecycle(engine)


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


async def trigger_ui_snapshot_rebuild_on_startup():
    """Publish default UI bootstrap snapshots in the background on startup."""
    try:
        from .services.ui_snapshot_service import safe_publish_all_bootstraps

        await asyncio.sleep(0)
        await asyncio.to_thread(safe_publish_all_bootstraps)
        print("UI snapshot bootstrap rebuild dispatched")
    except Exception as e:
        print(f"Warning: Failed to rebuild UI snapshots on startup: {e}")


def initialize_runtime() -> None:
    """Run the synchronous runtime initialization."""
    print("Starting Stock Scanner API...")
    print(f"Database: {settings.database_url}")
    print(f"CORS origins: {settings.cors_origins_list}")

    # Initialize database
    init_db()
    print("Database initialized")

    # Legacy data migrations — safe to remove once all deployments have run them
    run_universe_migration()
    run_theme_pipeline_state_migration()
    run_theme_cluster_identity_migration()
    run_theme_lifecycle_migration()
    run_theme_relationships_migration()
    run_theme_merge_suggestion_safety_migration()
    run_universe_lifecycle_migration()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    initialize_runtime()

    # Trigger non-blocking gap-fill for IBD group rankings
    if getattr(settings, 'group_rank_gapfill_enabled', True):
        await trigger_gapfill_on_startup()
    asyncio.create_task(trigger_ui_snapshot_rebuild_on_startup())

    yield

    # Shutdown
    print("Shutting down Stock Scanner API...")
    engine.dispose()
    print("Database connections closed")


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

_READINESS_TABLES = ("scans", "scan_results", "stock_universe")


@app.get("/")
async def root():
    """Return API information."""
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

    Uses a cheap schema existence check instead of a deep integrity scan
    so the endpoint stays non-blocking. Redis is a soft dependency — its
    absence degrades the service but doesn't make it unhealthy.
    """
    checks = {}
    healthy = True

    # DB check — offloaded to thread pool (non-blocking)
    try:
        def _check_db():
            with engine.connect() as conn:
                return all(
                    table_exists(conn, name)
                    for name in _READINESS_TABLES
                )

        if await asyncio.to_thread(_check_db):
            checks["database"] = "ok"
        else:
            checks["database"] = "error: required tables missing"
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

# Mount MCP Streamable HTTP transport (JSON-RPC over HTTP)
if settings.mcp_http_enabled:
    from fastapi import Depends

    from .interfaces.mcp import mcp_router
    from .services.server_auth import require_server_session

    app.include_router(
        mcp_router,
        prefix="/mcp",
        dependencies=[Depends(require_server_session)],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
