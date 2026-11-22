"""Main FastAPI application entry point."""

import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from .config import settings
from .database import SessionLocal, engine
from .infra.db.migrations import migrate_database_to_head
from .infra.db.portability import table_exists
from .infra.server_timing import install_server_timing
from .services.redis_pool import get_redis_client
from .wiring.bootstrap import (
    clear_runtime_services,
    initialize_process_runtime_services,
    reset_runtime_services,
    set_runtime_services,
)

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


from .utils.db_url import redacted_database_url as _redacted_database_url  # noqa: E402


class RuntimeServicesContextMiddleware:
    """Bind the app's runtime services to the request's context.

    Plain ASGI rather than ``@app.middleware("http")``: BaseHTTPMiddleware
    runs the endpoint in a separate task and pipes the response body through
    a memory stream on every request. Here the whole request, including
    response streaming and Starlette background tasks, runs inside this
    call, so the binding covers them without re-wrapping.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        runtime_services = getattr(scope["app"].state, "runtime_services", None)
        if runtime_services is None:
            await self.app(scope, receive, send)
            return

        token = set_runtime_services(runtime_services)
        try:
            await self.app(scope, receive, send)
        finally:
            reset_runtime_services(token)


def initialize_runtime() -> None:
    """Run the synchronous runtime initialization."""
    logger.info(
        "Starting Stock Scanner API",
        extra={
            "database_url": _redacted_database_url(settings.database_url),
            "cors_origins": settings.cors_origins_list,
        },
    )
    migrate_database_to_head(engine)
    logger.info("Database schema ready")
    from .services.social_source_admin_service import SocialSourceAdminService

    with SessionLocal() as db:
        service = SocialSourceAdminService(db)
        sources = service.ensure_seed_sources()
        runtime = service.apply_deployment_settings(settings)
    logger.info("Social source inventory ready", extra={"source_count": len(sources)})
    logger.info(
        "Social runtime ready",
        extra={"mode": runtime.mode, "provider": runtime.provider},
    )


async def trigger_ui_snapshot_rebuild_on_startup() -> None:
    """Compatibility shim for tests that still monkeypatch the old startup hook."""
    return None


def _publish_group_history_reconciliation() -> None:
    """Publish from a daemon thread that cannot delay process shutdown."""
    from .tasks.group_history_tasks import discover_group_history_reconciliation

    try:
        result = discover_group_history_reconciliation.delay()
        logger.info(
            "Group history startup reconciliation queued",
            extra={"task_id": result.id},
        )
    except Exception:
        logger.warning(
            "Group history startup reconciliation failed",
            exc_info=True,
        )


# Long enough to cover the API workers and restarts of one deploy; a restart
# after it expires retries a feature that is still unpublished.
STARTUP_REFRESH_GUARD_SECONDS = 6 * 60 * 60


def _publish_missing_feature_refreshes() -> None:
    """Queue the first COT and social imports instead of waiting for Beat.

    COT otherwise waits for the weekday 17:00 ET slot and social for its next
    six-hourly slot. A Redis NX key per feature lets every API worker call
    this while only one enqueues; both tasks are idempotent regardless.
    """
    from .infra.db.models.social_signals import (
        SocialSignalRunPointer,
        SocialSourceRegistry,
    )
    from .services.bootstrap_readiness_service import has_compatible_cot_publication
    from .services.redis_pool import get_redis_client

    try:
        with SessionLocal() as db:
            registry = db.get(SocialSourceRegistry, 1)
            missing = {
                "cot": not has_compatible_cot_publication(db),
                "social": (
                    registry is not None
                    and registry.mode == "live"
                    and registry.provider != "disabled"
                    and db.get(SocialSignalRunPointer, "latest_published") is None
                ),
            }
        redis = get_redis_client()
        for feature, unpublished in missing.items():
            if not unpublished or redis is None or not redis.set(
                f"startup_refresh:{feature}",
                "1",
                nx=True,
                ex=STARTUP_REFRESH_GUARD_SECONDS,
            ):
                continue
            if feature == "cot":
                from .interfaces.tasks.cot_tasks import refresh_cot as task
            else:
                from .interfaces.tasks.social_signal_tasks import (
                    refresh_social_signals as task,
                )
            try:
                result = task.delay(origin="startup")
            except Exception:
                # Release the guard so another worker or restart retries, and
                # keep going: one feature's broker failure must not skip the next.
                redis.delete(f"startup_refresh:{feature}")
                logger.warning(
                    "Initial %s refresh publication failed", feature, exc_info=True
                )
                continue
            logger.info(
                "Initial %s refresh queued", feature, extra={"task_id": result.id}
            )
    except Exception:
        logger.warning("Initial feature refresh publication failed", exc_info=True)


def _publish_startup_work() -> None:
    _publish_group_history_reconciliation()
    _publish_missing_feature_refreshes()


def trigger_group_history_reconciliation_on_startup() -> dict[str, str]:
    """Launch best-effort broker publication outside loop-owned executors."""
    publisher = threading.Thread(
        target=_publish_startup_work,
        name="startup-publisher",
        daemon=True,
    )
    publisher.start()
    return {"status": "dispatching"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    initialize_runtime()
    from .services.server_auth import server_auth_configured, server_auth_required

    if server_auth_required():
        # Resolves the cookie-signing secret, generating it on first start, so an
        # unwritable data dir stops startup instead of failing the first login.
        server_auth_configured()
    runtime_services = initialize_process_runtime_services(session_factory=SessionLocal)
    app.state.runtime_services = runtime_services
    trigger_group_history_reconciliation_on_startup()
    if settings.mcp_http_enabled:
        from .interfaces.mcp.http_transport import create_mcp_http_server

        app.state.mcp_server = create_mcp_http_server(session_factory=SessionLocal)

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

# NOTE: response compression is intentionally NOT done here. Every supported
# deployment compresses API JSON at the edge (nginx gzip_types includes
# application/json; the Caddy HTTPS overlay sets `encode zstd gzip`), which
# also correctly skips SSE (text/event-stream). Backend-level GZipMiddleware
# would duplicate that work on this container's CPU budget and gzip SSE.


app.add_middleware(RuntimeServicesContextMiddleware)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Outside every layer, Starlette's error handling included, so 500s are timed.
install_server_timing(app)

_READINESS_TABLES = ("scans", "scan_results", "stock_universe")


@app.get("/")
def root():
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


from .services.legacy_theme_write_guard import (
    ECONOMIC_ENDPOINT_REQUIRED,
    LegacyThemeWritesBlocked,
)


@app.exception_handler(LegacyThemeWritesBlocked)
async def legacy_theme_writes_blocked_handler(request, exc):
    # A cutover between a guarded route's entry check and its first write
    # (#472): same 409 as the entry check, not a 500.
    return JSONResponse(status_code=409, content={"detail": ECONOMIC_ENDPOINT_REQUIRED})


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
