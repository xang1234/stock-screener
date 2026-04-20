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
    supported_markets: list[str] = Field(default_factory=lambda: ["US", "HK", "IN", "JP", "TW"])
    api_base_path: str = "/api"
    auth: AppAuthStatusResponse = Field(default_factory=AppAuthStatusResponse)


class RuntimeBootstrapStatusResponse(BaseModel):
    """Current persisted local-runtime bootstrap state."""

    bootstrap_required: bool
    empty_system: bool
    primary_market: str
    enabled_markets: list[str]
    bootstrap_state: str
    supported_markets: list[str] = Field(default_factory=lambda: ["US", "HK", "IN", "JP", "TW"])


class RuntimeBootstrapRequest(BaseModel):
    """Bootstrap request payload for first-run local setup."""

    primary_market: str = "US"
    enabled_markets: list[str] = Field(default_factory=lambda: ["US"])


class RuntimeBootstrapStartResponse(RuntimeBootstrapStatusResponse):
    """Bootstrap response including the queued orchestration task."""

    task_id: str | None = None


class RuntimeActivityBootstrapResponse(BaseModel):
    """Bootstrap progress summary for the frontend shell."""

    state: str
    app_ready: bool
    primary_market: str
    enabled_markets: list[str] = Field(default_factory=list)
    current_stage: str | None = None
    progress_mode: str = "indeterminate"
    percent: float | None = None
    message: str | None = None
    background_warning: str | None = None


class RuntimeActivitySummaryResponse(BaseModel):
    """Compact runtime activity summary for the header."""

    active_market_count: int = 0
    active_markets: list[str] = Field(default_factory=list)
    status: str = "idle"


class RuntimeActivityMarketResponse(BaseModel):
    """Per-market runtime activity row."""

    market: str
    lifecycle: str
    stage_key: str | None = None
    stage_label: str | None = None
    status: str
    progress_mode: str = "indeterminate"
    percent: float | None = None
    current: int | None = None
    total: int | None = None
    message: str | None = None
    task_name: str | None = None
    task_id: str | None = None
    updated_at: str | None = None


class RuntimeActivityResponse(BaseModel):
    """Unified runtime activity payload for bootstrap and operations UI."""

    bootstrap: RuntimeActivityBootstrapResponse
    summary: RuntimeActivitySummaryResponse
    markets: list[RuntimeActivityMarketResponse] = Field(default_factory=list)


class RuntimeMarketsUpdateRequest(BaseModel):
    """Patch request for persisted local market preferences."""

    primary_market: str
    enabled_markets: list[str]
