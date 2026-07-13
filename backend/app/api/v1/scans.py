"""
Bulk scan API endpoints.

Handles creating scans, checking progress, and retrieving results.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from typing import List, Literal, Any
from pydantic import BaseModel, ValidationError
import logging

from ...schemas.cache import SmartRefreshResponse
from ...schemas.scanning import (
    ExplainResponse,
    ScanCreateRequest,
    ScanCreateResponse,
    ScanListItem,
    ScanListResponse,
    ScanResultItem,
    ScanStatusResponse,
    SetupDetailsResponse,
    normalize_scan_warnings_for_response,
)
from ...schemas.ui_view_snapshot import UISnapshotEnvelope
from ...database import SessionLocal
from ...domain.markets import market_registry
from ...domain.markets.catalog import get_market_catalog
from ...domain.universe.indexes import index_registry
from ...services.market_activity_gate import MarketActivityGate, MarketGateConflict
from ...services.market_activity_service import get_runtime_activity_status
from ...wiring.bootstrap import (
    get_uow,
    get_create_scan_use_case,
    get_get_single_result_use_case,
    get_get_setup_details_use_case,
    get_get_peers_use_case,
    get_explain_stock_use_case,
    get_job_backend,
    get_ui_snapshot_service,
)
from .scan_queries import router as scan_queries_router
from ...use_cases.scanning.create_scan import ActiveScanConflictError, StaleMarketDataError

logger = logging.getLogger(__name__)
router = APIRouter()
router.include_router(scan_queries_router)
_market_catalog = get_market_catalog()
SUPPORTED_SCAN_REFRESH_MARKETS = _market_catalog.market_codes_with_capability(
    "feature_snapshot"
)


class ScanCacheRefreshRequest(BaseModel):
    """Request payload for scan-facing cache refresh recovery."""

    market: str
    mode: Literal["auto", "full"] = "full"


def _normalize_scan_refresh_market(market: str) -> str:
    """Require an explicit market partition for scan recovery refreshes."""
    normalized = str(market or "").strip().upper()
    if normalized not in SUPPORTED_SCAN_REFRESH_MARKETS:
        supported = ", ".join(SUPPORTED_SCAN_REFRESH_MARKETS)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported market '{market}'. Expected one of: {supported}.",
        )
    return normalized


def _get_market_refresh_conflict_detail(market: str | None) -> dict[str, object] | None:
    if not market:
        return None

    result = MarketActivityGate(
        session_factory=SessionLocal,
        runtime_activity_reader=get_runtime_activity_status,
    ).check(market)
    if isinstance(result, MarketGateConflict):
        return result.detail
    return None


def _resolve_scan_guard_market(universe_def: Any) -> str | None:
    if getattr(universe_def, "market", None):
        return universe_def.market.value
    if getattr(universe_def, "exchange", None):
        market = market_registry.market_for_exchange(universe_def.exchange.value)
        return market.code if market is not None else None
    if getattr(universe_def, "index", None):
        return index_registry.market_for(universe_def.index.value)
    return None


@router.get("/bootstrap", response_model=UISnapshotEnvelope)
async def get_scan_bootstrap(
    scan_id: str | None = Query(None, description="Optional explicit scan bootstrap variant"),
    snapshot_service: Any = Depends(get_ui_snapshot_service),
):
    """Return the published scan bootstrap snapshot if available."""
    snapshot = snapshot_service.get_scan_bootstrap(scan_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="No published scan bootstrap snapshot is available")
    return UISnapshotEnvelope(**snapshot.to_dict())


@router.get("", response_model=ScanListResponse)
async def list_scans(
    limit: int = Query(20, ge=1, le=100, description="Number of scans to return"),
    market: str | None = Query(None, description="Restrict to scans of one universe market (e.g. US, HK)"),
    uow: Any = Depends(get_uow),
):
    """Get list of all scans ordered by most recent first."""
    normalized_market: str | None = None
    if market is not None:
        normalized_market = market.strip().upper()
        if normalized_market not in _market_catalog.supported_market_codes():
            supported = ", ".join(_market_catalog.supported_market_codes())
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported market '{market}'. Expected one of: {supported}.",
            )
    try:
        with uow:
            scans = uow.scans.list_recent(limit=limit, market=normalized_market)

            scan_items = []
            for scan in scans:
                scan_items.append(ScanListItem(
                    scan_id=scan.scan_id,
                    status=scan.status,
                    trigger_source=getattr(scan, "trigger_source", "manual") or "manual",
                    universe_def=scan.get_universe_definition(),
                    total_stocks=scan.total_stocks or 0,
                    passed_stocks=scan.passed_stocks or 0,
                    started_at=scan.started_at,
                    completed_at=scan.completed_at,
                    source="feature_store" if scan.feature_run_id else "scan_results",
                    warnings=normalize_scan_warnings_for_response(
                        getattr(scan, "warnings", None)
                    ),
                ))

        return ScanListResponse(scans=scan_items)

    except Exception as e:
        logger.error(f"Error listing scans: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing scans: {str(e)}")


@router.post("", response_model=ScanCreateResponse)
async def create_scan(
    request: ScanCreateRequest,
    response: Response,
    uow: Any = Depends(get_uow),
    use_case: Any = Depends(get_create_scan_use_case),
):
    """Create a new bulk scan via CreateScanUseCase."""
    from ...domain.common.errors import ValidationError as DomainValidationError
    from ...use_cases.scanning.create_scan import CreateScanCommand

    universe_resolution = _build_universe_resolution(request)
    universe_def = universe_resolution.universe_def
    if universe_resolution.used_legacy:
        logger.warning(universe_resolution.deprecation_log_message())
        for key, value in universe_resolution.deprecation_headers().items():
            response.headers[key] = value
        from ...services.universe_compat_metrics import record_legacy_universe_usage

        record_legacy_universe_usage(universe_resolution.legacy_value)
    guard_market = _resolve_scan_guard_market(universe_def)
    market_refresh_conflict = _get_market_refresh_conflict_detail(guard_market)
    if market_refresh_conflict is not None:
        raise HTTPException(status_code=409, detail=market_refresh_conflict)
    universe_projection = universe_def.storage_projection()
    cmd = CreateScanCommand(
        universe_def=universe_def,
        universe_label=universe_projection.label,
        universe_key=universe_projection.key,
        universe_type=universe_projection.type,
        universe_market=universe_projection.market,
        universe_exchange=universe_projection.exchange,
        universe_index=universe_projection.index,
        universe_symbols=universe_projection.symbols,
        screeners=request.screeners,
        composite_method=request.composite_method,
        criteria=request.criteria,
        idempotency_key=request.idempotency_key,
    )
    try:
        result = use_case.execute(uow, cmd)
    except ActiveScanConflictError as e:
        raise HTTPException(status_code=409, detail=e.to_dict()) from e
    except StaleMarketDataError as e:
        raise HTTPException(status_code=409, detail=e.to_dict()) from e
    except DomainValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create scan: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to queue scan task")

    if result.status == "completed":
        from ...services.ui_snapshot_service import safe_publish_scan_bootstrap

        safe_publish_scan_bootstrap(result.scan_id)
        safe_publish_scan_bootstrap()

    return ScanCreateResponse(
        scan_id=result.scan_id,
        status=result.status,
        total_stocks=result.total_stocks,
        message=(
            f"Scan completed instantly for {result.total_stocks} stocks"
            if result.status == "completed"
            else f"Scan queued for {result.total_stocks} stocks"
        ),
        feature_run_id=result.feature_run_id,
        warnings=normalize_scan_warnings_for_response(result.warnings),
        universe_def=universe_def,
    )


@router.post("/refresh-cache", response_model=SmartRefreshResponse)
async def refresh_scan_cache(request: ScanCacheRefreshRequest):
    """Queue a manual market data refresh from the scan workflow."""
    from .cache import _queue_manual_smart_refresh

    try:
        market = _normalize_scan_refresh_market(request.market)
        return _queue_manual_smart_refresh(mode=request.mode, market=market)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _build_universe_def(request: ScanCreateRequest):
    """Convert request fields into a typed UniverseDefinition."""
    return _build_universe_resolution(request).universe_def


def _build_universe_resolution(request: ScanCreateRequest):
    """Resolve request fields into typed universe metadata + compat diagnostics."""
    from ...services.universe_compat_adapter import resolve_scan_universe_request

    try:
        return resolve_scan_universe_request(
            universe_def=request.universe_def,
            legacy_universe=request.universe,
            legacy_symbols=request.symbols,
        )
    except (ValueError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{scan_id}/status", response_model=ScanStatusResponse)
async def get_scan_status(
    scan_id: str,
    uow: Any = Depends(get_uow),
):
    """Get scan progress and status."""
    try:
        with uow:
            scan = uow.scans.get_by_scan_id(scan_id)
            if not scan:
                raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")

            # Calculate progress from Celery task state for real-time updates
            progress = 0.0
            completed_stocks = 0
            eta_seconds = None
            status = scan.status

            if scan.status == "completed":
                progress = 100.0
                completed_stocks = scan.total_stocks or 0
            elif scan.status == "cancelled":
                completed_stocks = uow.scan_results.count_by_scan_id(scan_id)
                progress = (completed_stocks / scan.total_stocks * 100) if scan.total_stocks else 0
            elif scan.status == "failed":
                progress = 0.0
                completed_stocks = 0
            else:
                # Get real-time progress from the configured job backend
                if scan.task_id:
                    snapshot = get_job_backend().get_status(scan.task_id)
                    if snapshot is not None:
                        status = snapshot.status or status
                        progress = snapshot.percent or 0.0
                        completed_stocks = snapshot.current or 0
                        eta_seconds = snapshot.eta_seconds
                        if snapshot.status == "completed":
                            progress = 100.0
                            completed_stocks = scan.total_stocks or completed_stocks
                        elif snapshot.status == "cancelled":
                            progress = (
                                completed_stocks / scan.total_stocks * 100
                                if scan.total_stocks
                                else 0
                            )
                        elif snapshot.status == "failed":
                            progress = 0.0
                            completed_stocks = completed_stocks or 0
                    else:
                        completed_stocks = uow.scan_results.count_by_scan_id(scan_id)
                        progress = (completed_stocks / scan.total_stocks * 100) if scan.total_stocks else 0
                else:
                    completed_stocks = uow.scan_results.count_by_scan_id(scan_id)
                    progress = (completed_stocks / scan.total_stocks * 100) if scan.total_stocks else 0

            # Fallback ETA estimation
            if eta_seconds is None and scan.status == "running" and scan.total_stocks:
                remaining = scan.total_stocks - completed_stocks
                eta_seconds = remaining  # ~1 second per stock

            return ScanStatusResponse(
                scan_id=scan_id,
                status=status,
                progress=round(progress, 2),
                total_stocks=scan.total_stocks or 0,
                completed_stocks=completed_stocks,
                passed_stocks=scan.passed_stocks or 0,
                started_at=scan.started_at,
                eta_seconds=eta_seconds,
                warnings=normalize_scan_warnings_for_response(
                    getattr(scan, "warnings", None)
                ),
                universe_def=scan.get_universe_definition(),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting scan status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting scan status: {str(e)}")


@router.post("/{scan_id}/cancel")
async def cancel_scan(
    scan_id: str,
    uow: Any = Depends(get_uow),
):
    """Cancel a running scan."""
    try:
        with uow:
            scan = uow.scans.get_by_scan_id(scan_id)
            if not scan:
                raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")

            if scan.status not in ["queued", "running"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot cancel scan with status '{scan.status}'. Only queued or running scans can be cancelled."
                )

            uow.scans.update_status(scan_id, "cancelled")
            uow.commit()

        from ...services.ui_snapshot_service import safe_publish_scan_bootstrap

        safe_publish_scan_bootstrap(scan_id)
        safe_publish_scan_bootstrap()
        logger.info(f"Marked scan {scan_id} as cancelled")
        return {
            "message": f"Scan {scan_id} cancelled successfully",
            "scan_id": scan_id,
            "status": "cancelled",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling scan: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error cancelling scan: {str(e)}")


@router.delete("/{scan_id}")
async def delete_scan(
    scan_id: str,
    uow: Any = Depends(get_uow),
):
    """Delete a scan and all its results."""
    try:
        with uow:
            scan = uow.scans.get_by_scan_id(scan_id)
            if not scan:
                raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")

            if scan.status in ("queued", "running"):
                raise HTTPException(
                    status_code=409,
                    detail=f"Cannot delete scan with status '{scan.status}'. Cancel it first.",
                )

            uow.scan_results.delete_by_scan_id(scan_id)
            uow.scans.delete(scan_id)
            uow.commit()

        logger.info(f"Deleted scan {scan_id}")
        return {"message": f"Scan {scan_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting scan: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting scan: {str(e)}")


@router.get("/{scan_id}/result/{symbol}", response_model=ScanResultItem)
async def get_single_result(
    scan_id: str,
    symbol: str,
    detail_level: Literal["core", "full"] = Query(
        "core",
        description="Response detail level. 'core' excludes heavy setup-engine payload fields.",
    ),
    uow: Any = Depends(get_uow),
    use_case: Any = Depends(get_get_single_result_use_case),
):
    """
    Get a single stock result from a scan by symbol.

    This is an optimized endpoint for fetching a single stock's data
    instead of fetching all results and searching client-side.
    """
    try:
        from ...domain.common.errors import EntityNotFoundError
        from ...use_cases.scanning.get_single_result import GetSingleResultQuery

        result = use_case.execute(
            uow,
            GetSingleResultQuery(
                scan_id=scan_id,
                symbol=symbol,
                include_setup_payload=(detail_level == "full"),
            ),
        )
    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting single result: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting single result: {str(e)}")
    return ScanResultItem.from_domain(
        result.item,
        include_setup_payload=(detail_level == "full"),
    )


@router.get("/{scan_id}/setup/{symbol}", response_model=SetupDetailsResponse)
async def get_setup_details(
    scan_id: str,
    symbol: str,
    uow: Any = Depends(get_uow),
    use_case: Any = Depends(get_get_setup_details_use_case),
):
    """Get setup-engine explain payload for a single symbol."""
    try:
        from ...domain.common.errors import EntityNotFoundError
        from ...use_cases.scanning.get_setup_details import GetSetupDetailsQuery

        result = use_case.execute(
            uow,
            GetSetupDetailsQuery(scan_id=scan_id, symbol=symbol),
        )
    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting setup details: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting setup details: {str(e)}")

    return SetupDetailsResponse(
        scan_id=scan_id,
        symbol=result.symbol,
        se_explain=result.se_explain,
        se_candidates=result.se_candidates,
    )


@router.get("/{scan_id}/peers/{symbol}", response_model=List[ScanResultItem])
async def get_industry_peers(
    scan_id: str,
    symbol: str,
    peer_type: str = Query("industry", pattern="^(industry|sector)$"),
    uow: Any = Depends(get_uow),
    use_case: Any = Depends(get_get_peers_use_case),
):
    """Get peer stocks in the same industry group or sector as the given symbol."""
    try:
        from ...domain.common.errors import EntityNotFoundError
        from ...domain.scanning.models import PeerType
        from ...use_cases.scanning.get_peers import GetPeersQuery

        query = GetPeersQuery(
            scan_id=scan_id,
            symbol=symbol,
            peer_type=PeerType(peer_type),
        )
        result = use_case.execute(uow, query)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting peers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting peers: {str(e)}")

    logger.info(
        f"Found {len(result.peers)} peers for {symbol} "
        f"({result.peer_type.value}: '{result.group_name}')"
    )
    return [ScanResultItem.from_domain(item) for item in result.peers]


# ---------------------------------------------------------------------------
# Explain endpoint
# ---------------------------------------------------------------------------


@router.get("/{scan_id}/explain/{symbol}", response_model=ExplainResponse)
async def explain_stock(
    scan_id: str,
    symbol: str,
    uow: Any = Depends(get_uow),
    use_case: Any = Depends(get_explain_stock_use_case),
):
    """Explain why a stock received its composite score and rating."""
    try:
        from ...domain.common.errors import EntityNotFoundError
        from ...use_cases.scanning.explain_stock import ExplainStockQuery

        query = ExplainStockQuery(scan_id=scan_id, symbol=symbol)
        result = use_case.execute(uow, query)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error explaining stock: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error explaining stock: {str(e)}")
    return ExplainResponse.from_domain(result.explanation)
