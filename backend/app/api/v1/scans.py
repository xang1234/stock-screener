"""
Bulk scan API endpoints.

Handles creating scans, checking progress, and retrieving results.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import Optional, List, Literal
from pydantic import ValidationError
import logging

from ...schemas.scanning import (
    ExplainResponse,
    FilterOptionsResponse,
    ScanCreateRequest,
    ScanCreateResponse,
    ScanListItem,
    ScanListResponse,
    ScanResultItem,
    ScanResultsResponse,
    ScanSymbolsResponse,
    ScanStatusResponse,
    SetupDetailsResponse,
)
from ...schemas.universe import UniverseDefinition
from ...wiring.bootstrap import (
    get_uow,
    get_create_scan_use_case,
    get_get_filter_options_use_case,
    get_get_scan_results_use_case,
    get_get_scan_symbols_use_case,
    get_get_single_result_use_case,
    get_get_setup_details_use_case,
    get_get_peers_use_case,
    get_export_scan_results_use_case,
    get_explain_stock_use_case,
)
from ...use_cases.scanning.create_scan import CreateScanCommand, CreateScanUseCase
from ...use_cases.scanning.explain_stock import ExplainStockQuery, ExplainStockUseCase
from ...use_cases.scanning.export_scan_results import ExportScanResultsQuery, ExportScanResultsUseCase
from ...use_cases.scanning.get_filter_options import GetFilterOptionsQuery, GetFilterOptionsUseCase
from ...use_cases.scanning.get_peers import GetPeersQuery, GetPeersUseCase
from ...use_cases.scanning.get_scan_results import GetScanResultsQuery, GetScanResultsUseCase
from ...use_cases.scanning.get_scan_symbols import GetScanSymbolsQuery, GetScanSymbolsUseCase
from ...use_cases.scanning.get_setup_details import GetSetupDetailsQuery, GetSetupDetailsUseCase
from ...use_cases.scanning.get_single_result import GetSingleResultQuery, GetSingleResultUseCase
from ...infra.db.uow import SqlUnitOfWork
from ...domain.common.errors import EntityNotFoundError, ValidationError as DomainValidationError
from ...domain.scanning.filter_spec import FilterSpec, SortSpec, PageSpec, QuerySpec
from ...domain.scanning.models import ExportFormat, PeerType
from .scan_filter_params import parse_scan_filters, parse_scan_sort, parse_page_spec

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("", response_model=ScanListResponse)
async def list_scans(
    limit: int = Query(20, ge=1, le=100, description="Number of scans to return"),
    uow: SqlUnitOfWork = Depends(get_uow),
):
    """Get list of all scans ordered by most recent first."""
    try:
        with uow:
            scans = uow.scans.list_recent(limit=limit)

            scan_items = []
            for scan in scans:
                symbols_count = len(scan.universe_symbols) if scan.universe_symbols else None
                scan_items.append(ScanListItem(
                    scan_id=scan.scan_id,
                    status=scan.status,
                    universe=scan.universe or "unknown",
                    universe_type=scan.universe_type,
                    universe_exchange=scan.universe_exchange,
                    universe_index=scan.universe_index,
                    universe_symbols_count=symbols_count,
                    total_stocks=scan.total_stocks or 0,
                    passed_stocks=scan.passed_stocks or 0,
                    started_at=scan.started_at,
                    completed_at=scan.completed_at,
                    source="feature_store" if scan.feature_run_id else "scan_results",
                ))

        return ScanListResponse(scans=scan_items)

    except Exception as e:
        logger.error(f"Error listing scans: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing scans: {str(e)}")


@router.post("", response_model=ScanCreateResponse)
async def create_scan(
    request: ScanCreateRequest,
    uow: SqlUnitOfWork = Depends(get_uow),
    use_case: CreateScanUseCase = Depends(get_create_scan_use_case),
):
    """Create a new bulk scan via CreateScanUseCase."""
    universe_def = _build_universe_def(request)
    cmd = CreateScanCommand(
        universe_def=universe_def,
        universe_label=universe_def.label(),
        universe_key=universe_def.key(),
        universe_type=universe_def.type.value,
        universe_exchange=universe_def.exchange.value if universe_def.exchange else None,
        universe_index=universe_def.index.value if universe_def.index else None,
        universe_symbols=universe_def.symbols,
        screeners=request.screeners,
        composite_method=request.composite_method,
        criteria=request.criteria,
        idempotency_key=request.idempotency_key,
    )
    try:
        result = use_case.execute(uow, cmd)
    except DomainValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create scan: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to queue scan task")

    return ScanCreateResponse(
        scan_id=result.scan_id,
        status=result.status,
        total_stocks=result.total_stocks,
        message=f"Scan queued for {result.total_stocks} stocks",
        feature_run_id=result.feature_run_id,
    )


def _build_universe_def(request: ScanCreateRequest) -> UniverseDefinition:
    """Convert request fields into a typed UniverseDefinition."""
    if request.universe_def is not None:
        return request.universe_def
    try:
        return UniverseDefinition.from_legacy(request.universe, request.symbols)
    except (ValueError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))


def _build_optional_page_spec(
    page: int | None,
    per_page: int | None,
) -> PageSpec | None:
    """Build optional pagination settings for symbol-list endpoints."""
    if page is None and per_page is None:
        return None
    return PageSpec(
        page=page or 1,
        per_page=per_page or 100,
    )


@router.get("/{scan_id}/status", response_model=ScanStatusResponse)
async def get_scan_status(
    scan_id: str,
    uow: SqlUnitOfWork = Depends(get_uow),
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
                # Get real-time progress from Celery task
                if scan.task_id:
                    from celery.result import AsyncResult
                    task_result = AsyncResult(scan.task_id)

                    if task_result.state == 'PROGRESS' and task_result.info:
                        progress = task_result.info.get('percent', 0.0)
                        completed_stocks = task_result.info.get('current', 0)
                        eta_seconds = task_result.info.get('eta_seconds')
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
                status=scan.status,
                progress=round(progress, 2),
                total_stocks=scan.total_stocks or 0,
                completed_stocks=completed_stocks,
                passed_stocks=scan.passed_stocks or 0,
                started_at=scan.started_at,
                eta_seconds=eta_seconds,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting scan status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting scan status: {str(e)}")


@router.post("/{scan_id}/cancel")
async def cancel_scan(
    scan_id: str,
    uow: SqlUnitOfWork = Depends(get_uow),
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


@router.get("/{scan_id}/results", response_model=ScanResultsResponse)
async def get_scan_results(
    scan_id: str,
    passes_only: bool = Query(False, description="Show only stocks passing template"),
    include_sparklines: bool = Query(True, description="Include sparkline data"),
    detail_level: Literal["table", "full"] = Query(
        "table",
        description="Response detail level. 'table' excludes heavy setup-engine payload fields.",
    ),
    filters: FilterSpec = Depends(parse_scan_filters),
    sort: SortSpec = Depends(parse_scan_sort),
    page: PageSpec = Depends(parse_page_spec),
    uow: SqlUnitOfWork = Depends(get_uow),
    use_case: GetScanResultsUseCase = Depends(get_get_scan_results_use_case),
):
    """Get scan results with pagination, sorting, and filtering."""
    try:
        query = GetScanResultsQuery(
            scan_id=scan_id,
            query_spec=QuerySpec(filters=filters, sort=sort, page=page),
            include_sparklines=include_sparklines,
            include_setup_payload=(detail_level == "full"),
            passes_only=passes_only,
        )
        result = use_case.execute(uow, query)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting scan results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting scan results: {str(e)}")

    return ScanResultsResponse(
        scan_id=scan_id,
        total=result.page.total,
        page=result.page.page,
        per_page=result.page.per_page,
        pages=result.page.total_pages,
        results=[
            ScanResultItem.from_domain(
                item,
                include_setup_payload=(detail_level == "full"),
            )
            for item in result.page.items
        ],
    )


@router.get("/{scan_id}/symbols", response_model=ScanSymbolsResponse)
async def get_scan_symbols(
    scan_id: str,
    passes_only: bool = Query(False, description="Show only stocks passing template"),
    page: int | None = Query(None, ge=1, description="Optional page number"),
    per_page: int | None = Query(None, ge=1, le=100, description="Optional results per page"),
    filters: FilterSpec = Depends(parse_scan_filters),
    sort: SortSpec = Depends(parse_scan_sort),
    uow: SqlUnitOfWork = Depends(get_uow),
    use_case: GetScanSymbolsUseCase = Depends(get_get_scan_symbols_use_case),
):
    """Get a lightweight, filtered symbol list for chart navigation."""
    try:
        page_spec = _build_optional_page_spec(page, per_page)
        result = use_case.execute(
            uow,
            GetScanSymbolsQuery(
                scan_id=scan_id,
                filters=filters,
                sort=sort,
                page=page_spec,
                passes_only=passes_only,
            ),
        )
    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting scan symbols: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting scan symbols: {str(e)}")

    return ScanSymbolsResponse(
        scan_id=scan_id,
        total=result.total,
        symbols=list(result.symbols),
        page=result.page,
        per_page=result.per_page,
        next_cursor=None,
    )


@router.delete("/{scan_id}")
async def delete_scan(
    scan_id: str,
    uow: SqlUnitOfWork = Depends(get_uow),
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


@router.get("/{scan_id}/export")
async def export_scan_results(
    scan_id: str,
    format: str = Query(default="csv", pattern="^(csv)$"),
    passes_only: bool = Query(False, description="Show only stocks passing template"),
    filters: FilterSpec = Depends(parse_scan_filters),
    sort: SortSpec = Depends(parse_scan_sort),
    uow: SqlUnitOfWork = Depends(get_uow),
    use_case: ExportScanResultsUseCase = Depends(get_export_scan_results_use_case),
):
    """Export scan results to CSV with full filter and sort support."""
    try:
        query = ExportScanResultsQuery(
            scan_id=scan_id,
            filters=filters,
            sort=sort,
            export_format=ExportFormat(format),
            passes_only=passes_only,
        )
        result = use_case.execute(uow, query)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error exporting scan results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error exporting scan results: {str(e)}")

    return StreamingResponse(
        iter([result.content]),
        media_type=result.media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{result.filename}"',
            "Content-Length": str(len(result.content)),
        },
    )


@router.get("/{scan_id}/filter-options", response_model=FilterOptionsResponse)
async def get_filter_options(
    scan_id: str,
    uow: SqlUnitOfWork = Depends(get_uow),
    use_case: GetFilterOptionsUseCase = Depends(get_get_filter_options_use_case),
):
    """
    Get unique values for categorical filters from this scan's results.

    Returns lists of unique IBD industries, GICS sectors, and ratings
    that exist in the scan results, for populating filter dropdowns.
    """
    try:
        result = use_case.execute(uow, GetFilterOptionsQuery(scan_id=scan_id))
    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting filter options: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting filter options: {str(e)}")

    return FilterOptionsResponse(
        ibd_industries=list(result.options.ibd_industries),
        gics_sectors=list(result.options.gics_sectors),
        ratings=list(result.options.ratings),
    )


@router.get("/{scan_id}/result/{symbol}", response_model=ScanResultItem)
async def get_single_result(
    scan_id: str,
    symbol: str,
    detail_level: Literal["core", "full"] = Query(
        "core",
        description="Response detail level. 'core' excludes heavy setup-engine payload fields.",
    ),
    uow: SqlUnitOfWork = Depends(get_uow),
    use_case: GetSingleResultUseCase = Depends(get_get_single_result_use_case),
):
    """
    Get a single stock result from a scan by symbol.

    This is an optimized endpoint for fetching a single stock's data
    instead of fetching all results and searching client-side.
    """
    try:
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
    uow: SqlUnitOfWork = Depends(get_uow),
    use_case: GetSetupDetailsUseCase = Depends(get_get_setup_details_use_case),
):
    """Get setup-engine explain payload for a single symbol."""
    try:
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
    uow: SqlUnitOfWork = Depends(get_uow),
    use_case: GetPeersUseCase = Depends(get_get_peers_use_case),
):
    """Get peer stocks in the same industry group or sector as the given symbol."""
    try:
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
    uow: SqlUnitOfWork = Depends(get_uow),
    use_case: ExplainStockUseCase = Depends(get_explain_stock_use_case),
):
    """Explain why a stock received its composite score and rating."""
    try:
        query = ExplainStockQuery(scan_id=scan_id, symbol=symbol)
        result = use_case.execute(uow, query)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error explaining stock: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error explaining stock: {str(e)}")
    return ExplainResponse.from_domain(result.explanation)
