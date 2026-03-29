"""Runtime capability, setup, and update endpoints for desktop installs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ...config import settings
from ...domain.scanning.defaults import get_default_scan_profile
from ...schemas.app_runtime import (
    AppCapabilitiesResponse,
    BootstrapStatusResponse,
    DataStatusResponse,
    ScanDefaultsResponse,
    SetupOptionResponse,
    SetupStatusResponse,
    UpdateStatusResponse,
)

router = APIRouter()


def _default_data_status() -> dict:
    return {
        "local_data_present": False,
        "starter_baseline_active": False,
        "setup_completed_at": None,
        "prices": {"ready": False, "last_success_at": None, "message": "Desktop data is not installed yet."},
        "breadth": {"ready": False, "last_success_at": None, "message": "Desktop data is not installed yet."},
        "groups": {"ready": False, "last_success_at": None, "message": "Desktop data is not installed yet."},
        "fundamentals": {"ready": False, "last_success_at": None, "message": "Desktop data is not installed yet."},
        "universe": {"ready": False, "last_success_at": None, "message": "Desktop data is not installed yet."},
    }


def _default_setup_state() -> dict:
    return {
        "status": "completed" if not settings.desktop_mode else "idle",
        "mode": None,
        "job_id": None,
        "message": "Desktop setup is not required" if not settings.desktop_mode else "Desktop setup has not started",
        "current_step": None,
        "started_at": None,
        "completed_at": None,
        "current": 0,
        "total": 0,
        "percent": 100.0 if not settings.desktop_mode else 0.0,
        "steps": [],
        "warnings": [],
        "error": None,
        "starter_baseline_active": False,
        "app_ready": not settings.desktop_mode,
        "data_status": _default_data_status(),
    }


def _default_update_state() -> dict:
    return {
        "status": "completed" if not settings.desktop_mode else "idle",
        "scope": None,
        "triggered_by": None,
        "job_id": None,
        "message": "Automatic updates are not required" if not settings.desktop_mode else "Automatic updates are idle",
        "current_step": None,
        "started_at": None,
        "completed_at": None,
        "last_success_at": None,
        "current": 0,
        "total": 0,
        "percent": 100.0 if not settings.desktop_mode else 0.0,
        "steps": [],
        "warnings": [],
        "error": None,
        "data_status": _default_data_status(),
    }


def _default_bootstrap_state() -> dict:
    setup = _default_setup_state()
    return {
        "status": setup["status"],
        "job_id": setup["job_id"],
        "message": setup["message"],
        "current_step": setup["current_step"],
        "started_at": setup["started_at"],
        "completed_at": setup["completed_at"],
        "current": setup["current"],
        "total": setup["total"],
        "percent": setup["percent"],
        "steps": setup["steps"],
        "warnings": setup["warnings"],
        "error": setup["error"],
    }


def _get_setup_state() -> dict:
    if not settings.desktop_mode:
        return _default_setup_state()
    from ...wiring.bootstrap import get_desktop_setup_service

    return get_desktop_setup_service().get_status()


def _get_update_state() -> dict:
    if not settings.desktop_mode:
        return _default_update_state()
    from ...wiring.bootstrap import get_desktop_update_service

    return get_desktop_update_service().get_status()


def _get_bootstrap_state() -> dict:
    if not settings.desktop_mode:
        return _default_bootstrap_state()
    from ...wiring.bootstrap import get_desktop_setup_service

    return get_desktop_setup_service().get_legacy_bootstrap_status()


@router.get("/app-capabilities", response_model=AppCapabilitiesResponse)
async def get_app_capabilities() -> AppCapabilitiesResponse:
    """Return frontend capability flags and desktop setup/update status."""
    setup = _get_setup_state()
    update = _get_update_state()
    data_status = setup.get("data_status") or update.get("data_status") or _default_data_status()
    from ...wiring.bootstrap import get_ui_snapshot_service

    setup_required = settings.desktop_mode and not bool(setup.get("app_ready"))
    return AppCapabilitiesResponse(
        desktop_mode=settings.desktop_mode,
        features=settings.capability_flags(),
        ui_snapshots=get_ui_snapshot_service().ui_snapshot_flags(),
        scan_defaults=ScanDefaultsResponse(**get_default_scan_profile()),
        bootstrap_required=setup_required,
        bootstrap=BootstrapStatusResponse(**_get_bootstrap_state()),
        setup_required=setup_required,
        setup=SetupStatusResponse(**setup),
        setup_options=[SetupOptionResponse(**option) for option in ([] if not settings.desktop_mode else _get_setup_options())],
        update=UpdateStatusResponse(**update),
        data_status=DataStatusResponse(**data_status),
    )


def _get_setup_options() -> list[dict]:
    if not settings.desktop_mode:
        return []
    from ...wiring.bootstrap import get_desktop_setup_service

    return get_desktop_setup_service().get_options()


@router.post("/app/setup", response_model=SetupStatusResponse)
async def start_desktop_setup(
    mode: str = Query("quick_start", description="quick_start or download_latest"),
    force: bool = Query(False, description="Restart setup even if a prior run already completed"),
) -> SetupStatusResponse:
    """Start first-run desktop setup for the local install."""
    if not settings.desktop_mode:
        raise HTTPException(status_code=400, detail="Desktop setup is only available in desktop mode")
    from ...wiring.bootstrap import get_desktop_setup_service

    state = get_desktop_setup_service().start_setup(mode=mode, force=force)
    return SetupStatusResponse(**state)


@router.get("/app/setup/status", response_model=SetupStatusResponse)
async def get_desktop_setup_status() -> SetupStatusResponse:
    """Return current desktop setup status."""
    return SetupStatusResponse(**_get_setup_state())


@router.post("/app/update/run-now", response_model=UpdateStatusResponse)
async def run_desktop_update_now(
    scope: str = Query("manual", description="manual, daily, core, or weekly"),
    force: bool = Query(False, description="Queue a new update even if one is already in progress"),
) -> UpdateStatusResponse:
    """Queue or run a desktop-local data refresh."""
    if not settings.desktop_mode:
        raise HTTPException(status_code=400, detail="Desktop updates are only available in desktop mode")
    from ...wiring.bootstrap import get_desktop_update_service

    state = get_desktop_update_service().start_update(scope=scope, triggered_by="manual", force=force)
    return UpdateStatusResponse(**state)


@router.get("/app/update/status", response_model=UpdateStatusResponse)
async def get_desktop_update_status() -> UpdateStatusResponse:
    """Return current desktop update status."""
    return UpdateStatusResponse(**_get_update_state())


@router.post("/app/bootstrap", response_model=BootstrapStatusResponse)
async def start_desktop_bootstrap(
    force: bool = Query(False, description="Restart setup even if a prior run already completed"),
) -> BootstrapStatusResponse:
    """Compatibility alias for desktop setup."""
    state = await start_desktop_setup(mode="quick_start", force=force)
    return BootstrapStatusResponse(
        status=state.status,
        job_id=state.job_id,
        message=state.message,
        current_step=state.current_step,
        started_at=state.started_at,
        completed_at=state.completed_at,
        current=state.current,
        total=state.total,
        percent=state.percent,
        steps=state.steps,
        warnings=state.warnings,
        error=state.error,
    )


@router.get("/app/bootstrap/status", response_model=BootstrapStatusResponse)
async def get_desktop_bootstrap_status() -> BootstrapStatusResponse:
    """Compatibility alias for desktop setup status."""
    return BootstrapStatusResponse(**_get_bootstrap_state())
