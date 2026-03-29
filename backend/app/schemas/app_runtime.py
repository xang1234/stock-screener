"""Schemas for desktop runtime setup/update capabilities."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RuntimeStepResponse(BaseModel):
    """Status for an individual desktop setup/update step."""

    name: str
    label: str
    status: str
    message: str | None = None
    details: dict[str, Any] | None = None


class BootstrapStepResponse(RuntimeStepResponse):
    """Compatibility alias for legacy bootstrap responses."""


class DataDomainStatusResponse(BaseModel):
    """Freshness and readiness for one local data domain."""

    ready: bool
    last_success_at: str | None = None
    message: str | None = None


class DataStatusResponse(BaseModel):
    """Desktop-local data readiness across the main domains."""

    local_data_present: bool
    starter_baseline_active: bool = False
    setup_completed_at: str | None = None
    prices: DataDomainStatusResponse
    breadth: DataDomainStatusResponse
    groups: DataDomainStatusResponse
    fundamentals: DataDomainStatusResponse
    universe: DataDomainStatusResponse


class BootstrapStatusResponse(BaseModel):
    """Legacy bootstrap wire shape retained for compatibility."""

    status: str = Field(..., description="idle, queued, running, completed, or failed")
    job_id: str | None = None
    message: str | None = None
    current_step: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    current: int | None = None
    total: int | None = None
    percent: float | None = None
    steps: list[BootstrapStepResponse] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class SetupOptionResponse(BaseModel):
    """Available first-run setup choices for desktop installs."""

    id: str
    label: str
    description: str
    recommended: bool = False


class SetupStatusResponse(BootstrapStatusResponse):
    """Current first-run setup status."""

    mode: str | None = None
    starter_baseline_active: bool = False
    app_ready: bool = False
    data_status: DataStatusResponse | None = None


class UpdateStatusResponse(BaseModel):
    """Current desktop update status."""

    status: str = Field(..., description="idle, queued, running, completed, or failed")
    scope: str | None = None
    triggered_by: str | None = None
    job_id: str | None = None
    message: str | None = None
    current_step: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    last_success_at: str | None = None
    current: int | None = None
    total: int | None = None
    percent: float | None = None
    steps: list[RuntimeStepResponse] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    data_status: DataStatusResponse | None = None


class ScanDefaultsResponse(BaseModel):
    """Backend-owned default scan profile exposed to the frontend."""

    universe: str
    screeners: list[str] = Field(default_factory=list)
    composite_method: str
    criteria: dict[str, Any] = Field(default_factory=dict)


class AppCapabilitiesResponse(BaseModel):
    """Feature/capability flags exposed to the frontend."""

    desktop_mode: bool
    features: dict[str, bool]
    ui_snapshots: dict[str, bool] = Field(default_factory=dict)
    scan_defaults: ScanDefaultsResponse
    api_base_path: str = "/api"
    bootstrap_required: bool
    bootstrap: BootstrapStatusResponse
    setup_required: bool = False
    setup: SetupStatusResponse
    setup_options: list[SetupOptionResponse] = Field(default_factory=list)
    update: UpdateStatusResponse
    data_status: DataStatusResponse
