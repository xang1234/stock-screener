"""Runtime capability endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from ...config import settings
from ...database import get_db
from ...deployment.enabled_markets import markets_outside_deployment_message
from ...domain.markets import UnsupportedMarketError
from ...domain.markets.catalog import get_market_catalog
from ...domain.scanning.defaults import get_default_scan_profile
from ...infra.db.models.social_signals import SocialSourceRegistry
from ...schemas.app_runtime import (
    AppAuthStatusResponse,
    AppCapabilitiesResponse,
    RuntimeActivityResponse,
    RuntimeBootstrapRequest,
    RuntimeBootstrapStartResponse,
    RuntimeBootstrapStatusResponse,
    RuntimeMarketsUpdateRequest,
    ScanDefaultsResponse,
)
from ...services.bootstrap_run_manifest import (
    CURRENT_BOOTSTRAP_OWNERSHIP_VERSION,
    BootstrapAlreadyRunning,
    BootstrapRunManifestRepository,
)
from ...services.market_activity_service import get_runtime_activity_status
from ...services.runtime_activity_contract import bootstrap_stage_metadata
from ...services.runtime_preferences_service import (
    DEFAULT_ENABLED_MARKETS,
    RuntimeBootstrapStatus,
    get_bootstrap_readiness_service,
    get_runtime_bootstrap_status,
    save_runtime_preferences,
)
from ...services.runtime_universe_options import build_runtime_universe_options_payload
from ...services.server_auth import get_server_auth_status, require_server_session
from ...tasks.runtime_bootstrap_tasks import (
    BootstrapDispatchError,
    queue_local_runtime_bootstrap,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _bootstrap_status_payload(status: object) -> dict[str, object]:
    """Serialize runtime bootstrap status from dataclasses or test doubles."""
    return {
        "bootstrap_required": bool(getattr(status, "bootstrap_required")),
        "empty_system": bool(getattr(status, "empty_system")),
        "primary_market": str(getattr(status, "primary_market")),
        "enabled_markets": list(getattr(status, "enabled_markets")),
        "bootstrap_state": str(getattr(status, "bootstrap_state")),
        "supported_markets": list(getattr(status, "supported_markets")),
        "deployment_markets": settings.enabled_markets_list,
        "bootstrap_stages": bootstrap_stage_metadata(),
    }


def _reject_markets_outside_deployment(requested: list[str]) -> None:
    """422 unless every market the request will actually use is deployed."""
    try:
        message = markets_outside_deployment_message(
            requested, settings.enabled_markets_list
        )
    except UnsupportedMarketError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if message is not None:
        raise HTTPException(status_code=422, detail=message)


def _refresh_runtime_bootstrap_status(db: Session) -> RuntimeBootstrapStatus:
    """Discard request-session identities changed by bootstrap orchestration."""
    db.expire_all()
    return get_runtime_bootstrap_status(db)


def _has_unfenced_running_bootstrap(
    db: Session,
    status: RuntimeBootstrapStatus,
) -> bool:
    if status.bootstrap_state != "running":
        return False
    manifest = BootstrapRunManifestRepository().load(db)
    return bool(
        manifest is None
        or manifest.ownership_version < CURRENT_BOOTSTRAP_OWNERSHIP_VERSION
    )


def _bootstrap_already_running_error(status: RuntimeBootstrapStatus) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={
            "code": "bootstrap_already_running",
            "message": "Local bootstrap is already running.",
            "bootstrap_state": status.bootstrap_state,
            "primary_market": status.primary_market,
            "enabled_markets": status.enabled_markets,
        },
    )


@router.get("/app-capabilities", response_model=AppCapabilitiesResponse)
def get_app_capabilities(
    request: Request,
    db: Session = Depends(get_db),
) -> AppCapabilitiesResponse:
    """Return frontend capability flags and auth status."""
    from ...wiring.bootstrap import get_ui_snapshot_service

    auth = get_server_auth_status(request)
    bootstrap_status = get_runtime_bootstrap_status(db)
    market_catalog = get_market_catalog()
    features = settings.capability_flags()
    if hasattr(db, "get"):
        social_runtime = db.get(SocialSourceRegistry, 1)
        if social_runtime is not None:
            features["social_signals"] = (
                social_runtime.mode == "live"
                and social_runtime.provider != "disabled"
            )
    return AppCapabilitiesResponse(
        features=features,
        ui_snapshots=get_ui_snapshot_service().ui_snapshot_flags(),
        scan_defaults=ScanDefaultsResponse(
            **get_default_scan_profile(bootstrap_status.primary_market)
        ),
        bootstrap_required=bootstrap_status.bootstrap_required,
        primary_market=bootstrap_status.primary_market,
        enabled_markets=bootstrap_status.enabled_markets,
        bootstrap_state=bootstrap_status.bootstrap_state,
        supported_markets=market_catalog.supported_market_codes(),
        deployment_markets=settings.enabled_markets_list,
        bootstrap_stages=bootstrap_stage_metadata(),
        market_catalog=market_catalog.as_runtime_payload(),
        universe_options=build_runtime_universe_options_payload(
            enabled_markets=bootstrap_status.enabled_markets,
        ),
        auth=AppAuthStatusResponse(**auth.__dict__),
    )


@router.get("/runtime/bootstrap-status", response_model=RuntimeBootstrapStatusResponse)
def get_bootstrap_status(
    db: Session = Depends(get_db),
) -> RuntimeBootstrapStatusResponse:
    """Return the persisted local bootstrap state and effective readiness."""
    status = get_runtime_bootstrap_status(db)
    readiness = get_bootstrap_readiness_service()
    return RuntimeBootstrapStatusResponse(
        **_bootstrap_status_payload(status),
        market_stages={
            market: readiness.stage_status(db, market)
            for market in status.enabled_markets
        },
        features=readiness.feature_status(db, enabled_markets=status.enabled_markets),
    )


@router.get("/runtime/activity", response_model=RuntimeActivityResponse)
def get_runtime_activity(
    db: Session = Depends(get_db),
) -> RuntimeActivityResponse:
    """Return unified bootstrap and per-market background activity status."""
    return RuntimeActivityResponse(**get_runtime_activity_status(db))


@router.post(
    "/runtime/bootstrap",
    response_model=RuntimeBootstrapStartResponse,
    dependencies=[Depends(require_server_session)],
)
def start_runtime_bootstrap(
    request: RuntimeBootstrapRequest,
    db: Session = Depends(get_db),
) -> RuntimeBootstrapStartResponse:
    """Persist local bootstrap choices and queue the primary-market sync."""
    # The bootstrap plan reads an empty list as the primary market alone.
    planned_markets = request.enabled_markets or [request.primary_market]
    _reject_markets_outside_deployment([request.primary_market, *planned_markets])
    previous_status = get_runtime_bootstrap_status(db)
    if _has_unfenced_running_bootstrap(db, previous_status):
        raise _bootstrap_already_running_error(previous_status)
    try:
        task_id = queue_local_runtime_bootstrap(
            primary_market=request.primary_market,
            enabled_markets=request.enabled_markets,
        )
    except BootstrapAlreadyRunning:
        status = _refresh_runtime_bootstrap_status(db)
        raise _bootstrap_already_running_error(status) from None
    except BootstrapDispatchError as exc:
        if not exc.dispatched_any:
            save_runtime_preferences(
                db,
                primary_market=request.primary_market,
                enabled_markets=planned_markets,
                bootstrap_state=previous_status.bootstrap_state,
            )
            raise
        logger.warning(
            "Bootstrap dispatch failed after queueing one or more market workflows",
            extra={
                "primary_market": exc.primary_market,
                "enabled_markets": exc.enabled_markets,
                "primary_task_id": exc.primary_task_id,
                "market_task_ids": exc.market_task_ids,
            },
            exc_info=True,
        )
        status = _refresh_runtime_bootstrap_status(db)
        payload = {
            **_bootstrap_status_payload(status),
            "task_id": exc.primary_task_id,
        }
        return RuntimeBootstrapStartResponse(**payload)
    status = _refresh_runtime_bootstrap_status(db)
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
def update_runtime_markets(
    request: RuntimeMarketsUpdateRequest,
    db: Session = Depends(get_db),
) -> RuntimeBootstrapStatusResponse:
    """Update persisted local market preferences without re-running bootstrap."""
    # The service saves an empty list as its default markets.
    _reject_markets_outside_deployment(
        [request.primary_market, *(request.enabled_markets or DEFAULT_ENABLED_MARKETS)]
    )
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
