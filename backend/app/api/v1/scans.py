"""
Bulk scan API endpoints.

Handles creating scans, checking progress, and retrieving results.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import Optional, List
from pydantic import BaseModel, Field, ValidationError
from datetime import datetime
import logging

from ...schemas.universe import UniverseDefinition
from ...wiring.bootstrap import (
    get_uow,
    get_create_scan_use_case,
    get_get_filter_options_use_case,
    get_get_scan_results_use_case,
    get_get_single_result_use_case,
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
from ...use_cases.scanning.get_single_result import GetSingleResultQuery, GetSingleResultUseCase
from ...infra.db.uow import SqlUnitOfWork
from ...domain.common.errors import EntityNotFoundError, ValidationError as DomainValidationError
from ...domain.scanning.filter_spec import FilterSpec, SortSpec, PageSpec, QuerySpec
from ...domain.scanning.models import ExportFormat, PeerType, ScanResultItemDomain
from .scan_filter_params import parse_scan_filters, parse_scan_sort, parse_page_spec

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models
class ScanCreateRequest(BaseModel):
    """Request model for creating a new scan"""
    universe: str = Field(
        default="all",
        description="Legacy universe selector. Accepts: all, test, custom, nyse, nasdaq, amex, sp500. "
                    "Prefer universe_def for new integrations."
    )
    symbols: Optional[List[str]] = Field(default=None, description="Custom symbol list (if universe=custom/test)")
    universe_def: Optional[UniverseDefinition] = Field(
        default=None,
        description="Structured universe definition. Takes precedence over legacy universe field."
    )
    criteria: Optional[dict] = Field(default=None, description="Scan criteria")

    # Multi-screener fields
    screeners: List[str] = Field(
        default=["minervini"],
        description="Screeners to run: minervini, canslim, ipo, custom, volume_breakthrough"
    )
    composite_method: str = Field(
        default="weighted_average",
        description="How to combine scores: weighted_average, maximum, minimum"
    )

    # Idempotency
    idempotency_key: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional idempotency key. Repeated POSTs with the same key return the existing scan."
    )


class ScanCreateResponse(BaseModel):
    """Response model for scan creation"""
    scan_id: str
    status: str
    total_stocks: int
    message: str
    feature_run_id: Optional[int] = None


class ScanStatusResponse(BaseModel):
    """Response model for scan status"""
    scan_id: str
    status: str
    progress: float
    total_stocks: int
    completed_stocks: int
    passed_stocks: int
    started_at: datetime
    eta_seconds: Optional[int] = None


class ScanResultItem(BaseModel):
    """Individual scan result item"""
    symbol: str
    company_name: Optional[str] = None  # Company name from stock_universe
    composite_score: float
    rating: str

    # Individual screener scores
    minervini_score: Optional[float] = None
    canslim_score: Optional[float] = None
    ipo_score: Optional[float] = None
    custom_score: Optional[float] = None
    volume_breakthrough_score: Optional[float] = None

    # Minervini fields
    rs_rating: Optional[float] = None
    rs_rating_1m: Optional[float] = None
    rs_rating_3m: Optional[float] = None
    rs_rating_12m: Optional[float] = None
    stage: Optional[int] = None
    stage_name: Optional[str] = None
    current_price: Optional[float] = None
    volume: Optional[float] = None
    market_cap: Optional[float] = None
    ma_alignment: Optional[bool] = None
    vcp_detected: Optional[bool] = None
    vcp_score: Optional[float] = None
    vcp_pivot: Optional[float] = None
    vcp_ready_for_breakout: Optional[bool] = None
    vcp_contraction_ratio: Optional[float] = None
    vcp_atr_score: Optional[float] = None
    passes_template: Optional[bool] = None

    # Growth fields
    adr_percent: Optional[float] = None
    eps_growth_qq: Optional[float] = None
    sales_growth_qq: Optional[float] = None
    eps_growth_yy: Optional[float] = None
    sales_growth_yy: Optional[float] = None

    # Valuation fields
    peg_ratio: Optional[float] = None

    # EPS Rating (IBD-style 0-99 percentile)
    eps_rating: Optional[int] = None

    # Phase 4: Industry classifications
    ibd_industry_group: Optional[str] = None
    ibd_group_rank: Optional[int] = None  # IBD group rank (1=best)
    gics_sector: Optional[str] = None
    gics_industry: Optional[str] = None

    # RS Sparkline data (30-day stock/SPY ratio trend)
    rs_sparkline_data: Optional[List[float]] = None
    rs_trend: Optional[int] = None  # -1=declining, 0=flat, 1=improving

    # Price Sparkline data (30-day normalized price trend)
    price_sparkline_data: Optional[List[float]] = None
    price_change_1d: Optional[float] = None  # 1-day percentage change
    price_trend: Optional[int] = None  # -1=down, 0=flat, 1=up overall

    # IPO date for age filtering
    ipo_date: Optional[str] = None  # Format: YYYY-MM-DD

    # Beta and Beta-Adjusted RS metrics
    beta: Optional[float] = None
    beta_adj_rs: Optional[float] = None
    beta_adj_rs_1m: Optional[float] = None
    beta_adj_rs_3m: Optional[float] = None
    beta_adj_rs_12m: Optional[float] = None

    # Multi-screener metadata
    screeners_run: Optional[List[str]] = None


class ScanResultsResponse(BaseModel):
    """Response model for paginated scan results"""
    scan_id: str
    total: int
    page: int
    per_page: int
    pages: int
    results: List[ScanResultItem]


class ScanListItem(BaseModel):
    """Individual scan in the list"""
    scan_id: str
    status: str
    universe: str  # Legacy label (backward compat)
    universe_type: Optional[str] = None
    universe_exchange: Optional[str] = None
    universe_index: Optional[str] = None
    universe_symbols_count: Optional[int] = None
    total_stocks: int
    passed_stocks: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    source: Optional[str] = None


class ScanListResponse(BaseModel):
    """Response model for list of scans"""
    scans: List[ScanListItem]


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
        results=[_domain_to_response(item) for item in result.page.items],
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


class FilterOptionsResponse(BaseModel):
    """Response model for filter options"""
    ibd_industries: List[str]
    gics_sectors: List[str]
    ratings: List[str]


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


def _domain_to_response(item: ScanResultItemDomain) -> ScanResultItem:
    """Map a domain scan result to the HTTP response model.

    This is the canonical domain-to-HTTP mapper.  All field unpacking
    from ``extended_fields`` happens here so that endpoint handlers
    stay thin and don't duplicate the 40+ field mapping.
    """
    ef = item.extended_fields
    return ScanResultItem(
        symbol=item.symbol,
        company_name=ef.get("company_name"),
        composite_score=item.composite_score,
        rating=item.rating,
        # Individual screener scores
        minervini_score=ef.get("minervini_score"),
        canslim_score=ef.get("canslim_score"),
        ipo_score=ef.get("ipo_score"),
        custom_score=ef.get("custom_score"),
        volume_breakthrough_score=ef.get("volume_breakthrough_score"),
        # Minervini fields
        rs_rating=ef.get("rs_rating"),
        rs_rating_1m=ef.get("rs_rating_1m"),
        rs_rating_3m=ef.get("rs_rating_3m"),
        rs_rating_12m=ef.get("rs_rating_12m"),
        stage=ef.get("stage"),
        stage_name=ef.get("stage_name"),
        current_price=item.current_price,
        volume=ef.get("volume"),
        market_cap=ef.get("market_cap"),
        ma_alignment=ef.get("ma_alignment"),
        vcp_detected=ef.get("vcp_detected"),
        vcp_score=ef.get("vcp_score"),
        vcp_pivot=ef.get("vcp_pivot"),
        vcp_ready_for_breakout=ef.get("vcp_ready_for_breakout"),
        vcp_contraction_ratio=ef.get("vcp_contraction_ratio"),
        vcp_atr_score=ef.get("vcp_atr_score"),
        passes_template=ef.get("passes_template"),
        # Growth fields
        adr_percent=ef.get("adr_percent"),
        eps_growth_qq=ef.get("eps_growth_qq"),
        sales_growth_qq=ef.get("sales_growth_qq"),
        eps_growth_yy=ef.get("eps_growth_yy"),
        sales_growth_yy=ef.get("sales_growth_yy"),
        # Valuation
        peg_ratio=ef.get("peg_ratio"),
        # EPS Rating
        eps_rating=ef.get("eps_rating"),
        # Industry classifications
        ibd_industry_group=ef.get("ibd_industry_group"),
        ibd_group_rank=ef.get("ibd_group_rank"),
        gics_sector=ef.get("gics_sector"),
        gics_industry=ef.get("gics_industry"),
        # Sparklines
        rs_sparkline_data=ef.get("rs_sparkline_data"),
        rs_trend=ef.get("rs_trend"),
        price_sparkline_data=ef.get("price_sparkline_data"),
        price_change_1d=ef.get("price_change_1d"),
        price_trend=ef.get("price_trend"),
        # IPO date
        ipo_date=ef.get("ipo_date"),
        # Beta and Beta-Adjusted RS
        beta=ef.get("beta"),
        beta_adj_rs=ef.get("beta_adj_rs"),
        beta_adj_rs_1m=ef.get("beta_adj_rs_1m"),
        beta_adj_rs_3m=ef.get("beta_adj_rs_3m"),
        beta_adj_rs_12m=ef.get("beta_adj_rs_12m"),
        # Multi-screener metadata
        screeners_run=item.screeners_run,
    )


@router.get("/{scan_id}/result/{symbol}", response_model=ScanResultItem)
async def get_single_result(
    scan_id: str,
    symbol: str,
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
            uow, GetSingleResultQuery(scan_id=scan_id, symbol=symbol)
        )
    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting single result: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting single result: {str(e)}")
    return _domain_to_response(result.item)


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
    return [_domain_to_response(item) for item in result.peers]


# ---------------------------------------------------------------------------
# Explain endpoint
# ---------------------------------------------------------------------------


class CriterionResultResponse(BaseModel):
    """One criterion's contribution within a screener."""
    name: str
    score: float
    max_score: float
    passed: bool


class ScreenerExplanationResponse(BaseModel):
    """Full explanation for one screener's evaluation."""
    screener_name: str
    score: float
    passes: bool
    rating: str
    criteria: List[CriterionResultResponse]


class ExplainResponse(BaseModel):
    """Complete explanation of a stock's composite score."""
    symbol: str
    composite_score: float
    rating: str
    composite_method: str
    screeners_passed: int
    screeners_total: int
    screener_explanations: List[ScreenerExplanationResponse]
    rating_thresholds: dict


def _explanation_to_response(explanation) -> ExplainResponse:
    """Convert domain StockExplanation → Pydantic response."""
    return ExplainResponse(
        symbol=explanation.symbol,
        composite_score=explanation.composite_score,
        rating=explanation.rating,
        composite_method=explanation.composite_method,
        screeners_passed=explanation.screeners_passed,
        screeners_total=explanation.screeners_total,
        screener_explanations=[
            ScreenerExplanationResponse(
                screener_name=se.screener_name,
                score=se.score,
                passes=se.passes,
                rating=se.rating,
                criteria=[
                    CriterionResultResponse(
                        name=c.name,
                        score=c.score,
                        max_score=c.max_score,
                        passed=c.passed,
                    )
                    for c in se.criteria
                ],
            )
            for se in explanation.screener_explanations
        ],
        rating_thresholds=explanation.rating_thresholds,
    )


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
    return _explanation_to_response(result.explanation)
