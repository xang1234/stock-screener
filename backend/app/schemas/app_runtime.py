"""Schemas for desktop runtime capabilities and bootstrap state."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BootstrapStepResponse(BaseModel):
    """Status for an individual desktop bootstrap step."""

    name: str
    label: str
    status: str
    message: str | None = None
    details: dict[str, Any] | None = None


class BootstrapStatusResponse(BaseModel):
    """Current desktop bootstrap status."""

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


class AppCapabilitiesResponse(BaseModel):
    """Feature/capability flags exposed to the frontend."""

    desktop_mode: bool
    features: dict[str, bool]
    ui_snapshots: dict[str, bool] = Field(default_factory=dict)
    api_base_path: str = "/api"
    bootstrap_required: bool
    bootstrap: BootstrapStatusResponse
