"""Runtime capability endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ...domain.scanning.defaults import get_default_scan_profile
from ...schemas.app_runtime import (
    AppCapabilitiesResponse,
    AppAuthStatusResponse,
    ScanDefaultsResponse,
)
from ...config import settings
from ...services.server_auth import get_server_auth_status

router = APIRouter()


@router.get("/app-capabilities", response_model=AppCapabilitiesResponse)
async def get_app_capabilities(request: Request) -> AppCapabilitiesResponse:
    """Return frontend capability flags and auth status."""
    from ...wiring.bootstrap import get_ui_snapshot_service

    auth = get_server_auth_status(request)
    return AppCapabilitiesResponse(
        features=settings.capability_flags(),
        ui_snapshots=get_ui_snapshot_service().ui_snapshot_flags(),
        scan_defaults=ScanDefaultsResponse(**get_default_scan_profile()),
        auth=AppAuthStatusResponse(**auth.__dict__),
    )
