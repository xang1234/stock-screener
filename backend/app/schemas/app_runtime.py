"""Schemas for runtime capabilities and local bootstrap controls."""

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
    bootstrap_required: bool = False
    primary_market: str = "US"
    enabled_markets: list[str] = Field(default_factory=lambda: ["US"])
    bootstrap_state: str = "not_started"
    supported_markets: list[str] = Field(default_factory=lambda: ["US", "HK", "JP", "TW"])
    api_base_path: str = "/api"
    auth: AppAuthStatusResponse = Field(default_factory=AppAuthStatusResponse)


class RuntimeBootstrapStatusResponse(BaseModel):
    """Current persisted local-runtime bootstrap state."""

    bootstrap_required: bool
    empty_system: bool
    primary_market: str
    enabled_markets: list[str]
    bootstrap_state: str
    supported_markets: list[str] = Field(default_factory=lambda: ["US", "HK", "JP", "TW"])


class RuntimeBootstrapRequest(BaseModel):
    """Bootstrap request payload for first-run local setup."""

    primary_market: str = "US"
    enabled_markets: list[str] = Field(default_factory=lambda: ["US"])


class RuntimeBootstrapStartResponse(RuntimeBootstrapStatusResponse):
    """Bootstrap response including the queued orchestration task."""

    task_id: str | None = None


class RuntimeMarketsUpdateRequest(BaseModel):
    """Patch request for persisted local market preferences."""

    primary_market: str
    enabled_markets: list[str]
