"""Streamable HTTP transport for the MCP server (spec 2025-03-26).

Mounts as a FastAPI router at ``/mcp`` and delegates JSON-RPC dispatch to the
same ``MarketCopilotMcpServer`` that powers the stdio transport.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from fastapi import APIRouter, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import SessionLocal

from .market_copilot import MarketCopilotService
from .server import MarketCopilotMcpServer

logger = logging.getLogger(__name__)

mcp_router = APIRouter(tags=["mcp"])

SessionFactory = Callable[..., Any]


def create_mcp_http_server(
    *,
    session_factory: SessionFactory = SessionLocal,
    runtime_settings: Any = settings,
) -> MarketCopilotMcpServer:
    """Build the process-scoped MCP HTTP server."""
    service = MarketCopilotService(session_factory, runtime_settings)
    return MarketCopilotMcpServer(service, transport=None)


def _get_server_from_app_state(request: Request) -> MarketCopilotMcpServer:
    server = getattr(request.app.state, "mcp_server", None)
    if server is None:
        raise RuntimeError("MCP HTTP server is not initialized on app.state.mcp_server")
    return server


@mcp_router.post("/")
async def mcp_post(request: Request) -> Response:
    """Handle a JSON-RPC request over HTTP (Streamable HTTP transport).

    Accepts a single JSON-RPC message, dispatches via the shared server
    instance, and returns the JSON-RPC response.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error: invalid JSON"},
            },
            status_code=200,
            media_type="application/json",
        )

    if not isinstance(body, dict):
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid Request: expected a JSON object, not array or scalar"},
            },
            status_code=200,
            media_type="application/json",
        )

    server = _get_server_from_app_state(request)
    response = await run_in_threadpool(server.dispatch, body)

    if response is None:
        # Notification — no response required per JSON-RPC spec.
        return Response(status_code=202)

    return JSONResponse(content=response, media_type="application/json")


@mcp_router.get("/")
async def mcp_sse() -> Response:
    """SSE keepalive endpoint (placeholder).

    The Streamable HTTP spec allows GET for server-initiated events.
    We don't push events, so this returns 405 for now.
    """
    return Response(status_code=405)


@mcp_router.delete("/")
async def mcp_session_teardown() -> Response:
    """Session teardown (no-op for stateless server)."""
    return Response(status_code=200)
