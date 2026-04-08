"""Schemas for runtime capabilities."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ScanDefaultsResponse(BaseModel):
    """Backend-owned default scan profile exposed to the frontend."""

    universe: str
    screeners: list[str] = Field(default_factory=list)
    composite_method: str
    criteria: dict[str, Any] = Field(default_factory=dict)


class AppAuthStatusResponse(BaseModel):
    """Authentication state exposed to the frontend shell."""

    required: bool = False
    configured: bool = True
    authenticated: bool = True
    mode: str = "session_cookie"
    message: str | None = None


class AppCapabilitiesResponse(BaseModel):
    """Feature/capability flags exposed to the frontend."""

    features: dict[str, bool]
    ui_snapshots: dict[str, bool] = Field(default_factory=dict)
    scan_defaults: ScanDefaultsResponse
    api_base_path: str = "/api"
    auth: AppAuthStatusResponse = Field(default_factory=AppAuthStatusResponse)
