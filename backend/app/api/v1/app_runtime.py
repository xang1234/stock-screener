"""Runtime capability and bootstrap endpoints for desktop installs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ...config import settings
from ...schemas.app_runtime import AppCapabilitiesResponse, BootstrapStatusResponse
from ...wiring.bootstrap import get_desktop_bootstrap_service

router = APIRouter()


def _default_bootstrap_state() -> dict:
    return {
        "status": "completed" if not settings.desktop_mode else "idle",
        "job_id": None,
        "message": "Desktop bootstrap is not required" if not settings.desktop_mode else "Desktop bootstrap has not started",
        "current_step": None,
        "started_at": None,
        "completed_at": None,
        "current": 0,
        "total": 0,
        "percent": 100.0 if not settings.desktop_mode else 0.0,
        "steps": [],
        "warnings": [],
        "error": None,
    }


def _get_bootstrap_state() -> dict:
    if not settings.desktop_mode:
        return _default_bootstrap_state()
    return get_desktop_bootstrap_service().get_status()


@router.get("/app-capabilities", response_model=AppCapabilitiesResponse)
async def get_app_capabilities() -> AppCapabilitiesResponse:
    """Return frontend capability flags and bootstrap state."""
    bootstrap = _get_bootstrap_state()
    return AppCapabilitiesResponse(
        desktop_mode=settings.desktop_mode,
        features=settings.capability_flags(),
        bootstrap_required=settings.desktop_mode and bootstrap["status"] != "completed",
        bootstrap=BootstrapStatusResponse(**bootstrap),
    )


@router.post("/app/bootstrap", response_model=BootstrapStatusResponse)
async def start_desktop_bootstrap(
    force: bool = Query(False, description="Restart bootstrap even if a prior run already completed"),
) -> BootstrapStatusResponse:
    """Queue desktop bootstrap work for first-run local installs."""
    if not settings.desktop_mode:
        raise HTTPException(status_code=400, detail="Desktop bootstrap is only available in desktop mode")
    state = get_desktop_bootstrap_service().start_bootstrap(force=force)
    return BootstrapStatusResponse(**state)


@router.get("/app/bootstrap/status", response_model=BootstrapStatusResponse)
async def get_desktop_bootstrap_status() -> BootstrapStatusResponse:
    """Return the current desktop bootstrap job status."""
    return BootstrapStatusResponse(**_get_bootstrap_state())
