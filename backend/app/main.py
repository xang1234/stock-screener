"""
Main FastAPI application entry point.
"""
import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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


def run_theme_pipeline_run_migration():
    """
    Run idempotent Theme pipeline-run schema migration on startup.

    Adds pipeline metadata to theme_pipeline_runs so run tracking and
    snapshot invalidation can stay pipeline-scoped.
    """
    from .db_migrations.theme_pipeline_run_migration import migrate_theme_pipeline_run_schema

    migrate_theme_pipeline_run_schema(engine)


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


def run_theme_embedding_freshness_migration():
    """
    Run idempotent theme embedding freshness migration on startup.

    Adds embedding freshness metadata used to avoid unnecessary recompute.
    """
    from .db_migrations.theme_embedding_freshness_migration import migrate_theme_embedding_freshness

    migrate_theme_embedding_freshness(engine)


def run_theme_merge_suggestion_safety_migration():
    """
    Run idempotent merge-suggestion safety migration on startup.

    Adds canonical pair identity columns and strict approval idempotency fields.
    """
    from .db_migrations.theme_merge_suggestion_safety_migration import migrate_theme_merge_suggestion_safety

    migrate_theme_merge_suggestion_safety(engine)


def run_theme_taxonomy_migration():
    """
    Run idempotent L1/L2 taxonomy migration on startup.

    Adds parent_cluster_id, is_l1, taxonomy_level, and assignment tracking
    columns to theme_clusters for hierarchical theme grouping.
    """
    from .db_migrations.theme_taxonomy_migration import migrate_theme_taxonomy

    migrate_theme_taxonomy(engine)


def run_ui_view_snapshot_migration():
    """
    Run idempotent UI snapshot schema migration on startup.

    Creates published bootstrap snapshot tables used by the read-optimized
    dashboard bootstrap endpoints.
    """
    from .db_migrations.ui_view_snapshot_migration import migrate_ui_view_snapshot_tables

    migrate_ui_view_snapshot_tables(engine)


def run_universe_lifecycle_migration():
    """Run idempotent stock universe lifecycle migration on startup."""
    from .db_migrations.universe_lifecycle_migration import migrate_universe_lifecycle

    migrate_universe_lifecycle(engine)


def run_provider_snapshot_migration():
    """Run idempotent provider snapshot migration on startup."""
    from .db_migrations.provider_snapshot_migration import migrate_provider_snapshot_tables

    migrate_provider_snapshot_tables(engine)


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

            # Quick integrity check on startup (checks page structure, fast)
            try:
                rows = conn.execute(text("PRAGMA quick_check")).fetchall()
                errors = [r[0] for r in rows
                          if r[0] != "ok"
                          and not r[0].startswith("Fragmentation")
                          and not r[0].startswith("***")]
                if errors:
                    print(f"CRITICAL: Database integrity check failed:")
                    for err in errors[:5]:
                        print(f"  {err}")
                    print("The database may be corrupted. Check data/backups/ for recent copies.")
                else:
                    print("Database integrity: ok")
            except Exception as e:
                print(f"CRITICAL: Database integrity check error: {e}")

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
    run_theme_pipeline_run_migration()
    run_theme_cluster_identity_migration()
    run_theme_aliases_migration()
    run_theme_match_decision_migration()
    run_theme_lifecycle_migration()
    run_theme_relationships_migration()
    run_theme_embedding_freshness_migration()
    run_theme_merge_suggestion_safety_migration()
    run_theme_taxonomy_migration()
    run_ui_view_snapshot_migration()
    run_universe_lifecycle_migration()
    run_provider_snapshot_migration()

    # Trigger non-blocking gap-fill for IBD group rankings
    if not settings.desktop_mode and getattr(settings, 'group_rank_gapfill_enabled', True):
        await trigger_gapfill_on_startup()
    asyncio.create_task(trigger_ui_snapshot_rebuild_on_startup())

    yield

    # Shutdown
    print("Shutting down Stock Scanner API...")
    if "sqlite" in settings.database_url:
        try:
            with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                conn.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
            print("WAL checkpoint complete")
        except Exception as e:
            print(f"WAL checkpoint failed (non-fatal): {e}")
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


def _desktop_index_path() -> Path:
    return settings.frontend_dist_path / "index.html"


def _desktop_static_available() -> bool:
    return settings.desktop_mode and _desktop_index_path().exists()


def _is_reserved_frontend_path(path: str) -> bool:
    reserved_prefixes = (
        "api/",
        "docs",
        "redoc",
        "openapi.json",
        "livez",
        "readyz",
        "health",
    )
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in reserved_prefixes)


@app.get("/")
async def root():
    """Serve the desktop SPA or return API information."""
    if _desktop_static_available():
        return FileResponse(_desktop_index_path())
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


@app.get("/{full_path:path}", include_in_schema=False)
async def desktop_spa_fallback(full_path: str):
    """Serve built frontend assets in desktop mode."""
    if not _desktop_static_available() or not full_path or _is_reserved_frontend_path(full_path):
        raise HTTPException(status_code=404, detail="Not found")

    candidate = settings.frontend_dist_path / full_path
    if candidate.is_file():
        return FileResponse(candidate)

    return FileResponse(_desktop_index_path())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=not settings.desktop_mode,
    )
