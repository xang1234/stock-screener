"""Schemas for server authentication endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AuthLoginRequest(BaseModel):
    """Password login request for single-user server deployments."""

    password: str = Field(..., min_length=1, max_length=512)


class ServerAuthStatusResponse(BaseModel):
    """Current authentication status for the requesting client."""

    required: bool
    configured: bool
    authenticated: bool
    mode: str = "session_cookie"
    message: str | None = None
