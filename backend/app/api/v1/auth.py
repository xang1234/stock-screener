"""Single-user server authentication endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ...schemas.auth import AuthLoginRequest, ServerAuthStatusResponse
from ...services.server_auth import (
    attach_auth_cookie,
    clear_auth_cookie,
    get_server_auth_status,
    issue_session_token,
    password_matches,
)

router = APIRouter()


@router.get("/status", response_model=ServerAuthStatusResponse)
def get_auth_status(request: Request) -> ServerAuthStatusResponse:
    """Return whether the current client is authenticated."""
    status = get_server_auth_status(request)
    return ServerAuthStatusResponse(**status.__dict__)


@router.post("/login", response_model=ServerAuthStatusResponse)
def login_server(request: Request, payload: AuthLoginRequest) -> JSONResponse:
    """Authenticate the current browser session with the shared server password."""
    status = get_server_auth_status()
    if not status.required:
        response = JSONResponse(ServerAuthStatusResponse(**status.__dict__).model_dump())
        return response
    if not status.configured:
        raise HTTPException(
            status_code=503,
            detail="Server authentication is required but not configured.",
        )
    if not password_matches(payload.password):
        raise HTTPException(status_code=401, detail="Invalid password")

    token = issue_session_token()
    body = ServerAuthStatusResponse(
        required=True,
        configured=True,
        authenticated=True,
        message="Authenticated.",
    )
    response = JSONResponse(body.model_dump())
    attach_auth_cookie(response, token, request=request)
    return response


@router.post("/logout", response_model=ServerAuthStatusResponse)
def logout_server() -> JSONResponse:
    """Clear the current browser auth session."""
    status = get_server_auth_status()
    body = ServerAuthStatusResponse(
        required=status.required,
        configured=status.configured,
        authenticated=False if status.required else status.authenticated,
        message="Signed out." if status.required else status.message,
    )
    response = JSONResponse(body.model_dump())
    clear_auth_cookie(response)
    return response
