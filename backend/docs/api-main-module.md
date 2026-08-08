# Backend Module Documentation: FastAPI Application Entry Point

## 1. Module Overview

### Purpose
High-level summary of what this module/file does.

This module defines the primary FastAPI application entry point for the stock screener backend. It is responsible for application bootstrapping, startup and shutdown lifecycle management, middleware configuration, health and readiness probes, and mounting the main API router.

### Core Responsibilities
- Initialize the FastAPI application and configure its metadata.
- Manage startup and shutdown events through the lifespan context manager.
- Bind runtime services into request and background execution contexts.
- Expose operational health endpoints for liveness and readiness monitoring.
- Mount the versioned API surface under `/api/v1`.
- Optionally enable the MCP HTTP transport layer when configured.

### Architecture / Scope
This module sits at the API boundary and application bootstrap layer. It depends on configuration, database access, Redis connectivity, and runtime wiring services, but it does not implement the screening business logic itself. Instead, it provides the runtime environment required for the rest of the backend to operate.

## 2. Dependencies & Configuration

### External Packages/Libraries
- FastAPI: Web framework for routing, middleware, and request handling.
- Starlette: Underlying ASGI framework used for middleware and response handling.
- SQLAlchemy: Database engine management and connection lifecycle control.
- Pydantic / Pydantic Settings: Environment-driven configuration loading.
- Uvicorn: ASGI server used when running the application directly.
- Redis client abstractions: Used for readiness checks and runtime-service integration.

### Environment Variables & Secrets
- `DATABASE_URL`: Database connection string for PostgreSQL.
- `CORS_ORIGINS`: Comma-separated allowed origins for browser access.
- `API_HOST`: Host interface used when running the app directly.
- `API_PORT`: Port used by the local ASGI server.
- `SERVER_EXPOSE_API_DOCS`: Enables or disables `/docs`, `/redoc`, and `/openapi.json`.
- `MCP_HTTP_ENABLED`: Enables the MCP HTTP transport integration.
- `SERVER_AUTH_ENABLED`, `SERVER_AUTH_PASSWORD`, `SERVER_AUTH_SESSION_SECRET`: Authentication-related settings.
- `ADMIN_API_KEY`: Privileged administrative API key configuration.
- Additional provider-specific API keys are defined in the application settings model.

## 3. Data Structures & Types

### `Settings`
Type: Pydantic settings model

Purpose: Central configuration object loaded from environment variables.

Key fields:
- `database_url: str`
- `api_host: str`
- `api_port: int`
- `cors_origins: str`
- `server_expose_api_docs: bool`
- `mcp_http_enabled: bool`
- `server_auth_enabled: bool`

### FastAPI Application Instance
Type: `FastAPI`

Purpose: Main application object that exposes routes and middleware.

### Runtime Service Context
Type: runtime object attached to `app.state.runtime_services`

Purpose: Carries process-scoped or request-scoped service references for downstream components.

### Readiness Response Payload
Type: JSON object

Fields:
- `status: str` (`ok`, `degraded`, or `unhealthy`)
- `checks: dict[str, str]`

## 4. API Endpoints / Exposed Interfaces

### `GET /`
Purpose: Returns basic API metadata.

Success response:
- Status: `200 OK`
- Example payload:
```json
{
  "name": "Stock Scanner API",
  "version": "0.1.0",
  "description": "CANSLIM + Minervini stock scanner",
  "status": "running"
}
```

### `GET /livez`
Purpose: Lightweight liveness probe.

Success response:
- Status: `200 OK`
- Example payload:
```json
{
  "status": "ok"
}
```

### `GET /readyz`
Purpose: Readiness probe for database and Redis availability.

Request parameters: None.

Success response:
- Status: `200 OK` when required database tables exist and Redis is reachable.
- Example payload:
```json
{
  "status": "ok",
  "checks": {
    "database": "ok",
    "redis": "ok"
  }
}
```

Error responses:
- Status: `503 Service Unavailable` if required tables are missing or health checks fail.

### `GET /health`
Purpose: Deprecated compatibility wrapper around `/readyz`.

Behavior: Returns the readiness payload plus deprecation metadata.

## 5. Functions & Methods Reference

| Function / Component | Signature | Description |
|---|---|---|
| `_log_critical_error` | `def _log_critical_error(...): None` | Emits structured logging for startup, authentication, cache, and theme-related failures. |
| `_bind_runtime_to_response_background` | `def _bind_runtime_to_response_background(response: Response, runtime_services: Any) -> None` | Preserves runtime service context when background tasks are executed after a request completes. |
| `initialize_runtime` | `def initialize_runtime() -> None` | Runs database migrations and logs startup configuration. |
| `trigger_ui_snapshot_rebuild_on_startup` | `async def trigger_ui_snapshot_rebuild_on_startup() -> None` | Compatibility shim for legacy startup hooks and tests. |
| `lifespan` | `async def lifespan(app: FastAPI)` | Manages startup and shutdown lifecycle, initializes services, and disposes resources. |
| `_docs_enabled` | `def _docs_enabled() -> bool` | Returns whether API docs should be exposed based on configuration. |
| `root` | `async def root()` | Returns the basic API descriptor payload. |
| `liveness` | `async def liveness()` | Returns the lightweight liveness response. |
| `readiness` | `async def readiness()` | Verifies the health of the database and Redis dependencies. |
| `health_check` | `async def health_check()` | Deprecated wrapper around readiness for compatibility. |

### Parameters Table Example: `readiness`

| Parameter | Type | Required | Description |
|---|---|---|---|
| None | N/A | N/A | The endpoint is parameterless. |

### Return Value
- `root`: JSON-serializable metadata payload.
- `liveness`: A simple `{ "status": "ok" }` response.
- `readiness`: A JSON response containing readiness status and dependency checks.
- `health_check`: A compatibility response based on readiness output.

### Side Effects / I/O
- Performs database connectivity checks through SQLAlchemy.
- Performs Redis ping checks through the Redis client abstraction.
- Runs database migrations during startup.
- Initializes and disposes runtime services.
- Optionally creates an MCP HTTP server when enabled.

## 6. Error Handling & Edge Cases

- Database readiness failures are caught and converted into structured warning logs.
- Redis readiness failures are treated as degraded conditions rather than fatal startup failures.
- Missing runtime services are handled gracefully by skipping context binding when unavailable.
- Startup exceptions can still propagate through the lifespan manager, allowing the process to fail fast if critical initialization fails.
- The readiness endpoint uses lightweight table existence checks rather than deep schema validation to keep monitoring efficient.

## 7. Business Logic & Implementation Highlights

- Startup initialization is split into database migration readiness and runtime service initialization.
- The middleware `bind_runtime_services_context` ensures that downstream handlers can access runtime-scoped services even when work continues asynchronously in background tasks.
- The readiness endpoint is designed for operational monitoring rather than data correctness validation.
- Documentation exposure is controlled through configuration, allowing production deployments to disable OpenAPI surfaces if needed.
- The module acts as a thin but critical orchestration layer, providing the runtime plumbing required by the rest of the backend.
