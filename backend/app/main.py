"""Main FastAPI application entry point."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

from .config import settings
from .database import SessionLocal, engine
from .infra.db.migrations import migrate_database_to_head
from .infra.db.portability import table_exists
from .services.redis_pool import get_redis_client
from .wiring.bootstrap import clear_runtime_services, initialize_process_runtime_services

logger = logging.getLogger(__name__)


def _log_critical_error(
    *,
    message: str,
    exc: Exception,
    event: str,
    path: str,
    error_code: str,
    level: int = logging.ERROR,
    pipeline: str | None = None,
    run_id: str | None = None,
    symbol: str | None = None,
) -> None:
    """Emit structured critical-path logs for startup/auth/cache/theme operations."""
    logger.log(
        level,
        message,
        extra={
            "event": event,
            "path": path,
            "pipeline": pipeline,
            "run_id": run_id,
            "symbol": symbol,
            "error_code": error_code,
        },
        exc_info=exc,
    )


def _redacted_database_url(database_url: str) -> str:
    try:
        return make_url(database_url).render_as_string(hide_password=True)
    except (TypeError, ValueError, AttributeError, SQLAlchemyError):
        return "<invalid>"


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
        logger.info(
            "IBD Group Rankings gap-fill task dispatched",
            extra={"task_id": task.id, "max_days": max_days, "queue": "data_fetch"},
        )

    except (ImportError, AttributeError, RuntimeError, OSError) as exc:
        # Don't block startup if gap-fill dispatch fails
        _log_critical_error(
            message="Failed to dispatch IBD group rankings gap-fill task during startup",
            exc=exc,
            event="startup_gapfill_dispatch_failed",
            path="main.trigger_gapfill_on_startup",
            error_code="gapfill_dispatch_failed",
            level=logging.WARNING,
        )
    except Exception as exc:
        # Keep startup resilient even when Celery broker/client raises unexpected errors.
        _log_critical_error(
            message="Unexpected error dispatching IBD group rankings gap-fill task during startup",
            exc=exc,
            event="startup_gapfill_dispatch_failed",
            path="main.trigger_gapfill_on_startup",
            error_code="gapfill_dispatch_unexpected",
            level=logging.WARNING,
        )


async def trigger_ui_snapshot_rebuild_on_startup():
    """Publish default UI bootstrap snapshots in the background on startup."""
    try:
        from .services.ui_snapshot_service import safe_publish_all_bootstraps

        await asyncio.sleep(0)
        await asyncio.to_thread(safe_publish_all_bootstraps)
        logger.info("UI snapshot bootstrap rebuild dispatched")
    except (ImportError, RuntimeError, OSError) as exc:
        _log_critical_error(
            message="Failed to rebuild UI snapshots during startup",
            exc=exc,
            event="startup_ui_snapshot_rebuild_failed",
            path="main.trigger_ui_snapshot_rebuild_on_startup",
            error_code="ui_snapshot_rebuild_failed",
            level=logging.WARNING,
        )
    except Exception as exc:
        _log_critical_error(
            message="Unexpected error rebuilding UI snapshots during startup",
            exc=exc,
            event="startup_ui_snapshot_rebuild_failed",
            path="main.trigger_ui_snapshot_rebuild_on_startup",
            error_code="ui_snapshot_rebuild_unexpected",
            level=logging.WARNING,
        )


def initialize_runtime() -> None:
    """Run the synchronous runtime initialization."""
    logger.info(
        "Starting Stock Scanner API",
        extra={
            "database_url": _redacted_database_url(settings.database_url),
            "cors_origins": settings.cors_origins_list,
        },
    )
    action = migrate_database_to_head(engine)
    logger.info("Database schema ready", extra={"migration_action": action})


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    initialize_runtime()
    runtime_services = initialize_process_runtime_services(session_factory=SessionLocal)
    app.state.runtime_services = runtime_services
    if settings.mcp_http_enabled:
        from .interfaces.mcp.http_transport import create_mcp_http_server

        app.state.mcp_server = create_mcp_http_server(session_factory=SessionLocal)

    # Trigger non-blocking gap-fill for IBD group rankings
    if getattr(settings, 'group_rank_gapfill_enabled', True):
        await trigger_gapfill_on_startup()
    asyncio.create_task(trigger_ui_snapshot_rebuild_on_startup())

    try:
        yield
    finally:
        # Shutdown
        logger.info("Shutting down Stock Scanner API")
        clear_runtime_services()
        if hasattr(app.state, "runtime_services"):
            delattr(app.state, "runtime_services")
        if hasattr(app.state, "mcp_server"):
            delattr(app.state, "mcp_server")
        engine.dispose()
        logger.info("Database connections closed")


def _docs_enabled() -> bool:
    return settings.server_expose_api_docs


_docs_url = "/docs" if _docs_enabled() else None
_redoc_url = "/redoc" if _docs_enabled() else None
_openapi_url = "/openapi.json" if _docs_enabled() else None


app = FastAPI(
    title="Stock Scanner API",
    description="CANSLIM + Minervini stock scanner with industry group analysis",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=_docs_url,
    redoc_url=_redoc_url,
    openapi_url=_openapi_url,
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
    payload = {
        "name": "Stock Scanner API",
        "version": "0.1.0",
        "description": "CANSLIM + Minervini stock scanner",
        "status": "running",
    }
    if _docs_url is not None:
        payload["docs"] = _docs_url
    return payload


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
    except (SQLAlchemyError, OSError, RuntimeError, ValueError) as exc:
        _log_critical_error(
            message="Database readiness probe failed",
            exc=exc,
            event="readiness_check_failed",
            path="/readyz",
            error_code="readiness_database_check_failed",
            level=logging.WARNING,
        )
        checks["database"] = f"error: {type(exc).__name__}"
        healthy = False
    except Exception as exc:
        _log_critical_error(
            message="Unexpected database readiness probe failure",
            exc=exc,
            event="readiness_check_failed",
            path="/readyz",
            error_code="readiness_database_check_unexpected",
            level=logging.WARNING,
        )
        checks["database"] = f"error: {type(exc).__name__}"
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
    except (OSError, RuntimeError, ValueError) as exc:
        _log_critical_error(
            message="Redis readiness probe failed",
            exc=exc,
            event="readiness_check_failed",
            path="/readyz",
            error_code="readiness_redis_check_failed",
            level=logging.WARNING,
        )
        checks["redis"] = f"warning: {type(exc).__name__}"
    except Exception as exc:
        _log_critical_error(
            message="Unexpected Redis readiness probe failure",
            exc=exc,
            event="readiness_check_failed",
            path="/readyz",
            error_code="readiness_redis_check_unexpected",
            level=logging.WARNING,
        )
        checks["redis"] = f"warning: {type(exc).__name__}"

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
