"""Runtime capability endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from ...domain.scanning.defaults import get_default_scan_profile
from ...database import get_db
from ...schemas.app_runtime import (
    AppCapabilitiesResponse,
    AppAuthStatusResponse,
    RuntimeBootstrapRequest,
    RuntimeBootstrapStartResponse,
    RuntimeBootstrapStatusResponse,
    RuntimeMarketsUpdateRequest,
    ScanDefaultsResponse,
)
from ...config import settings
from ...services.server_auth import get_server_auth_status, require_server_session
from ...services.runtime_preferences_service import (
    get_runtime_bootstrap_status,
    save_runtime_preferences,
)
from ...tasks.runtime_bootstrap_tasks import queue_local_runtime_bootstrap

router = APIRouter()


def _bootstrap_status_payload(status: object) -> dict[str, object]:
    """Serialize runtime bootstrap status from dataclasses or test doubles."""
    return {
        "bootstrap_required": bool(getattr(status, "bootstrap_required")),
        "empty_system": bool(getattr(status, "empty_system")),
        "primary_market": str(getattr(status, "primary_market")),
        "enabled_markets": list(getattr(status, "enabled_markets")),
        "bootstrap_state": str(getattr(status, "bootstrap_state")),
        "supported_markets": list(getattr(status, "supported_markets")),
    }


@router.get("/app-capabilities", response_model=AppCapabilitiesResponse)
async def get_app_capabilities(
    request: Request,
    db: Session = Depends(get_db),
) -> AppCapabilitiesResponse:
    """Return frontend capability flags and auth status."""
    from ...wiring.bootstrap import get_ui_snapshot_service

    auth = get_server_auth_status(request)
    bootstrap_status = get_runtime_bootstrap_status(db)
    return AppCapabilitiesResponse(
        features=settings.capability_flags(),
        ui_snapshots=get_ui_snapshot_service().ui_snapshot_flags(),
        scan_defaults=ScanDefaultsResponse(**get_default_scan_profile()),
        bootstrap_required=bootstrap_status.bootstrap_required,
        primary_market=bootstrap_status.primary_market,
        enabled_markets=bootstrap_status.enabled_markets,
        bootstrap_state=bootstrap_status.bootstrap_state,
        supported_markets=bootstrap_status.supported_markets,
        auth=AppAuthStatusResponse(**auth.__dict__),
    )


@router.get("/runtime/bootstrap-status", response_model=RuntimeBootstrapStatusResponse)
async def get_bootstrap_status(
    db: Session = Depends(get_db),
) -> RuntimeBootstrapStatusResponse:
    """Return the persisted local bootstrap state and effective readiness."""
    return RuntimeBootstrapStatusResponse(
        **_bootstrap_status_payload(get_runtime_bootstrap_status(db))
    )


@router.post(
    "/runtime/bootstrap",
    response_model=RuntimeBootstrapStartResponse,
    dependencies=[Depends(require_server_session)],
)
async def start_runtime_bootstrap(
    request: RuntimeBootstrapRequest,
    db: Session = Depends(get_db),
) -> RuntimeBootstrapStartResponse:
    """Persist local bootstrap choices and queue the primary-market sync."""
    current_status = get_runtime_bootstrap_status(db)
    if current_status.bootstrap_state == "running":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "bootstrap_already_running",
                "message": "Local bootstrap is already running.",
                "bootstrap_state": current_status.bootstrap_state,
                "primary_market": current_status.primary_market,
                "enabled_markets": current_status.enabled_markets,
            },
        )
    prefs = save_runtime_preferences(
        db,
        primary_market=request.primary_market,
        enabled_markets=request.enabled_markets,
        bootstrap_state="running",
    )
    try:
        task_id = queue_local_runtime_bootstrap(
            primary_market=prefs.primary_market,
            enabled_markets=prefs.enabled_markets,
        )
    except Exception:
        save_runtime_preferences(
            db,
            primary_market=prefs.primary_market,
            enabled_markets=prefs.enabled_markets,
            bootstrap_state=current_status.bootstrap_state,
        )
        raise
    status = get_runtime_bootstrap_status(db)
    payload = {
        **_bootstrap_status_payload(status),
        "task_id": task_id,
    }
    return RuntimeBootstrapStartResponse(**payload)


@router.patch(
    "/runtime/markets",
    response_model=RuntimeBootstrapStatusResponse,
    dependencies=[Depends(require_server_session)],
)
async def update_runtime_markets(
    request: RuntimeMarketsUpdateRequest,
    db: Session = Depends(get_db),
) -> RuntimeBootstrapStatusResponse:
    """Update persisted local market preferences without re-running bootstrap."""
    current_status = get_runtime_bootstrap_status(db)
    prefs = save_runtime_preferences(
        db,
        primary_market=request.primary_market,
        enabled_markets=request.enabled_markets,
        bootstrap_state=current_status.bootstrap_state,
    )
    status = get_runtime_bootstrap_status(db)
    payload = {
        **_bootstrap_status_payload(status),
        "primary_market": prefs.primary_market,
        "enabled_markets": prefs.enabled_markets,
    }
    return RuntimeBootstrapStatusResponse(**payload)
