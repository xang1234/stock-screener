"""
Bulk scan API endpoints.

Handles creating scans, checking progress, and retrieving results.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging
import csv
import io

from ...database import get_db
from ...models.scan_result import Scan, ScanResult
from ...models.stock_universe import StockUniverse
from pydantic import ValidationError
from ...schemas.universe import UniverseDefinition
from ...wiring.bootstrap import get_uow, get_create_scan_use_case, get_get_filter_options_use_case, get_get_single_result_use_case
from ...use_cases.scanning.create_scan import CreateScanCommand, CreateScanUseCase
from ...use_cases.scanning.get_filter_options import GetFilterOptionsQuery, GetFilterOptionsUseCase
from ...use_cases.scanning.get_single_result import GetSingleResultQuery, GetSingleResultUseCase
from ...infra.db.uow import SqlUnitOfWork
from ...domain.common.errors import EntityNotFoundError, ValidationError as DomainValidationError
from ...domain.scanning.models import ScanResultItemDomain

logger = logging.getLogger(__name__)
router = APIRouter()


def parse_ipo_after_preset(preset: str) -> Optional[str]:
    """
    Parse IPO date preset to a date string (YYYY-MM-DD).

    Presets:
    - 6m: 6 months ago
    - 1y: 1 year ago
    - 2y: 2 years ago
    - 3y: 3 years ago
    - 5y: 5 years ago
    - YYYY-MM-DD: explicit date

    Returns:
        Date string in YYYY-MM-DD format, or None if invalid
    """
    if not preset:
        return None

    preset = preset.strip().lower()
    today = datetime.now().date()

    if preset == '6m':
        cutoff = today - relativedelta(months=6)
    elif preset == '1y':
        cutoff = today - relativedelta(years=1)
    elif preset == '2y':
        cutoff = today - relativedelta(years=2)
    elif preset == '3y':
        cutoff = today - relativedelta(years=3)
    elif preset == '5y':
        cutoff = today - relativedelta(years=5)
    else:
        # Try to parse as explicit date
        try:
            cutoff = datetime.strptime(preset, '%Y-%m-%d').date()
        except ValueError:
            return None

    return cutoff.strftime('%Y-%m-%d')


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


class ScanListResponse(BaseModel):
    """Response model for list of scans"""
    scans: List[ScanListItem]


@router.get("", response_model=ScanListResponse)
async def list_scans(
    limit: int = Query(20, ge=1, le=100, description="Number of scans to return"),
    db: Session = Depends(get_db)
):
    """
    Get list of all scans ordered by most recent first.

    Args:
        limit: Maximum number of scans to return (default 20)
        db: Database session

    Returns:
        List of scans with their status
    """
    try:
        scans = db.query(Scan).order_by(desc(Scan.started_at)).limit(limit).all()

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
                completed_at=scan.completed_at
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
    db: Session = Depends(get_db)
):
    """
    Get scan progress and status.

    Args:
        scan_id: Scan UUID
        db: Database session

    Returns:
        Scan status with progress information
    """
    try:
        # Get scan from database
        scan = db.query(Scan).filter(Scan.scan_id == scan_id).first()

        if not scan:
            raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")

        # Calculate progress - read from Celery task state for real-time updates
        progress = 0.0
        completed_stocks = 0
        eta_seconds = None

        if scan.status == "completed":
            progress = 100.0
            completed_stocks = scan.total_stocks or 0
        elif scan.status == "cancelled":
            completed_count = db.query(ScanResult).filter(
                ScanResult.scan_id == scan_id
            ).count()
            completed_stocks = completed_count
            progress = (completed_stocks / scan.total_stocks * 100) if scan.total_stocks else 0
            eta_seconds = None
        elif scan.status == "failed":
            progress = 0.0
            completed_stocks = 0
        else:
            # Get real-time progress from Celery task
            if scan.task_id:
                from celery.result import AsyncResult
                task_result = AsyncResult(scan.task_id)

                # Check if task has state info
                if task_result.state == 'PROGRESS' and task_result.info:
                    # Use real-time progress from task state
                    progress = task_result.info.get('percent', 0.0)
                    completed_stocks = task_result.info.get('current', 0)
                    eta_seconds = task_result.info.get('eta_seconds')
                else:
                    # Fallback: count database results (less accurate, updates in batches)
                    completed_count = db.query(ScanResult).filter(
                        ScanResult.scan_id == scan_id
                    ).count()
                    completed_stocks = completed_count
                    progress = (completed_stocks / scan.total_stocks * 100) if scan.total_stocks else 0
            else:
                # No task_id: fallback to database count
                completed_count = db.query(ScanResult).filter(
                    ScanResult.scan_id == scan_id
                ).count()
                completed_stocks = completed_count
                progress = (completed_stocks / scan.total_stocks * 100) if scan.total_stocks else 0

        # Legacy ETA estimation (only used if Celery doesn't provide eta_seconds)
        if eta_seconds is None and scan.status == "running" and scan.total_stocks:
            remaining = scan.total_stocks - completed_stocks
            eta_seconds = remaining  # 1 second per stock

        return ScanStatusResponse(
            scan_id=scan_id,
            status=scan.status,
            progress=round(progress, 2),
            total_stocks=scan.total_stocks or 0,
            completed_stocks=completed_stocks,
            passed_stocks=scan.passed_stocks or 0,
            started_at=scan.started_at,
            eta_seconds=eta_seconds
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting scan status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting scan status: {str(e)}")


@router.post("/{scan_id}/cancel")
async def cancel_scan(
    scan_id: str,
    db: Session = Depends(get_db)
):
    """
    Cancel a running scan.

    Args:
        scan_id: Scan UUID
        db: Database session

    Returns:
        Cancellation confirmation
    """
    try:
        # Get scan from database
        scan = db.query(Scan).filter(Scan.scan_id == scan_id).first()

        if not scan:
            raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")

        # Check if scan is cancellable
        if scan.status not in ["queued", "running"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel scan with status '{scan.status}'. Only queued or running scans can be cancelled."
            )

        # Update scan status to cancelled
        # Note: The Celery task will continue running but won't save results
        # The task checks scan status before saving each result
        scan.status = "cancelled"
        scan.completed_at = datetime.utcnow()
        db.commit()

        logger.info(f"Marked scan {scan_id} as cancelled")

        return {
            "message": f"Scan {scan_id} cancelled successfully",
            "scan_id": scan_id,
            "status": "cancelled"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling scan: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error cancelling scan: {str(e)}")


@router.get("/{scan_id}/results", response_model=ScanResultsResponse)
async def get_scan_results(
    scan_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=100, description="Results per page"),
    sort_by: str = Query("minervini_score", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order: asc or desc"),
    # Text search
    symbol_search: Optional[str] = Query(None, description="Symbol search pattern"),
    # Categorical filters
    min_score: Optional[float] = Query(None, description="Minimum Minervini score filter"),
    max_score: Optional[float] = Query(None, description="Maximum Minervini score filter"),
    stage: Optional[int] = Query(None, description="Stage filter (1-4)"),
    ratings: Optional[str] = Query(None, description="Rating filter (comma-separated)"),
    ibd_industries: Optional[str] = Query(None, description="IBD industry filter (comma-separated)"),
    ibd_industries_mode: Optional[str] = Query(None, description="IBD industry filter mode: include or exclude"),
    gics_sectors: Optional[str] = Query(None, description="GICS sector filter (comma-separated)"),
    gics_sectors_mode: Optional[str] = Query(None, description="GICS sector filter mode: include or exclude"),
    passes_only: bool = Query(False, description="Show only stocks passing template"),
    # Composite score range
    min_composite: Optional[float] = Query(None, description="Minimum composite score"),
    max_composite: Optional[float] = Query(None, description="Maximum composite score"),
    # Individual screener scores
    min_canslim: Optional[float] = Query(None, description="Minimum CANSLIM score"),
    max_canslim: Optional[float] = Query(None, description="Maximum CANSLIM score"),
    min_ipo: Optional[float] = Query(None, description="Minimum IPO score"),
    max_ipo: Optional[float] = Query(None, description="Maximum IPO score"),
    min_custom: Optional[float] = Query(None, description="Minimum custom score"),
    max_custom: Optional[float] = Query(None, description="Maximum custom score"),
    min_vol_breakthrough: Optional[float] = Query(None, description="Minimum vol breakthrough score"),
    max_vol_breakthrough: Optional[float] = Query(None, description="Maximum vol breakthrough score"),
    # RS ratings
    min_rs: Optional[float] = Query(None, ge=0, le=100, description="Minimum RS Rating"),
    max_rs: Optional[float] = Query(None, ge=0, le=100, description="Maximum RS Rating"),
    min_rs_1m: Optional[float] = Query(None, ge=0, le=100, description="Minimum RS 1M"),
    max_rs_1m: Optional[float] = Query(None, ge=0, le=100, description="Maximum RS 1M"),
    min_rs_3m: Optional[float] = Query(None, ge=0, le=100, description="Minimum RS 3M"),
    max_rs_3m: Optional[float] = Query(None, ge=0, le=100, description="Maximum RS 3M"),
    min_rs_12m: Optional[float] = Query(None, ge=0, le=100, description="Minimum RS 12M"),
    max_rs_12m: Optional[float] = Query(None, ge=0, le=100, description="Maximum RS 12M"),
    # Price & Growth
    min_price: Optional[float] = Query(None, description="Minimum price"),
    max_price: Optional[float] = Query(None, description="Maximum price"),
    min_adr: Optional[float] = Query(None, description="Minimum ADR %"),
    max_adr: Optional[float] = Query(None, description="Maximum ADR %"),
    min_eps_growth: Optional[float] = Query(None, description="Minimum EPS growth Q/Q (%)"),
    max_eps_growth: Optional[float] = Query(None, description="Maximum EPS growth Q/Q (%)"),
    min_sales_growth: Optional[float] = Query(None, description="Minimum sales growth Q/Q (%)"),
    max_sales_growth: Optional[float] = Query(None, description="Maximum sales growth Q/Q (%)"),
    min_eps_growth_yy: Optional[float] = Query(None, description="Minimum EPS growth Y/Y (%)"),
    min_sales_growth_yy: Optional[float] = Query(None, description="Minimum sales growth Y/Y (%)"),
    max_peg: Optional[float] = Query(None, description="Maximum PEG ratio"),
    # EPS Rating filter
    min_eps_rating: Optional[int] = Query(None, ge=0, le=99, description="Minimum EPS Rating (0-99)"),
    max_eps_rating: Optional[int] = Query(None, ge=0, le=99, description="Maximum EPS Rating (0-99)"),
    # Volume & Market Cap filters
    min_volume: Optional[int] = Query(None, description="Minimum volume"),
    min_market_cap: Optional[int] = Query(None, description="Minimum market cap"),
    # VCP filters
    min_vcp_score: Optional[float] = Query(None, description="Minimum VCP score"),
    max_vcp_score: Optional[float] = Query(None, description="Maximum VCP score"),
    min_vcp_pivot: Optional[float] = Query(None, description="Minimum VCP pivot price"),
    max_vcp_pivot: Optional[float] = Query(None, description="Maximum VCP pivot price"),
    vcp_detected: Optional[bool] = Query(None, description="VCP detected filter"),
    vcp_ready: Optional[bool] = Query(None, description="VCP ready for breakout filter"),
    # Boolean filters
    ma_alignment: Optional[bool] = Query(None, description="MA alignment filter"),
    # Performance filters (price change %)
    min_perf_day: Optional[float] = Query(None, description="Minimum 1-day % change"),
    max_perf_day: Optional[float] = Query(None, description="Maximum 1-day % change"),
    min_perf_week: Optional[float] = Query(None, description="Minimum 5-day % change"),
    max_perf_week: Optional[float] = Query(None, description="Maximum 5-day % change"),
    min_perf_month: Optional[float] = Query(None, description="Minimum 21-day % change"),
    max_perf_month: Optional[float] = Query(None, description="Maximum 21-day % change"),
    # EMA distance filters (% above/below)
    min_ema_10: Optional[float] = Query(None, description="Minimum % from EMA10"),
    max_ema_10: Optional[float] = Query(None, description="Maximum % from EMA10"),
    min_ema_20: Optional[float] = Query(None, description="Minimum % from EMA20"),
    max_ema_20: Optional[float] = Query(None, description="Maximum % from EMA20"),
    min_ema_50: Optional[float] = Query(None, description="Minimum % from EMA50"),
    max_ema_50: Optional[float] = Query(None, description="Maximum % from EMA50"),
    # 52-week distance filters
    min_52w_high: Optional[float] = Query(None, description="Minimum % below 52-week high"),
    max_52w_high: Optional[float] = Query(None, description="Maximum % below 52-week high"),
    min_52w_low: Optional[float] = Query(None, description="Minimum % above 52-week low"),
    max_52w_low: Optional[float] = Query(None, description="Maximum % above 52-week low"),
    # IPO date filter (presets: 6m, 1y, 2y, 3y, 5y or YYYY-MM-DD)
    ipo_after: Optional[str] = Query(None, description="IPO after date (presets: 6m, 1y, 2y, 3y, 5y or YYYY-MM-DD)"),
    # Beta and Beta-Adjusted RS filters
    min_beta: Optional[float] = Query(None, description="Minimum Beta"),
    max_beta: Optional[float] = Query(None, description="Maximum Beta"),
    min_beta_adj_rs: Optional[float] = Query(None, ge=0, le=100, description="Minimum Beta-adjusted RS"),
    max_beta_adj_rs: Optional[float] = Query(None, ge=0, le=100, description="Maximum Beta-adjusted RS"),
    min_beta_adj_rs_1m: Optional[float] = Query(None, ge=0, le=100, description="Minimum Beta-adjusted RS 1M"),
    min_beta_adj_rs_3m: Optional[float] = Query(None, ge=0, le=100, description="Minimum Beta-adjusted RS 3M"),
    min_beta_adj_rs_12m: Optional[float] = Query(None, ge=0, le=100, description="Minimum Beta-adjusted RS 12M"),
    # Qullamaggie extended performance filters
    min_perf_3m: Optional[float] = Query(None, description="Minimum 3-month % change"),
    max_perf_3m: Optional[float] = Query(None, description="Maximum 3-month % change"),
    min_perf_6m: Optional[float] = Query(None, description="Minimum 6-month % change"),
    max_perf_6m: Optional[float] = Query(None, description="Maximum 6-month % change"),
    # Episodic Pivot filters
    min_gap_percent: Optional[float] = Query(None, description="Minimum gap up %"),
    max_gap_percent: Optional[float] = Query(None, description="Maximum gap up %"),
    min_volume_surge: Optional[float] = Query(None, description="Minimum volume surge ratio"),
    max_volume_surge: Optional[float] = Query(None, description="Maximum volume surge ratio"),
    # Sparkline data control (for performance optimization)
    include_sparklines: bool = Query(True, description="Include sparkline data in response (set to false to reduce payload)"),
    db: Session = Depends(get_db)
):
    """
    Get scan results with pagination, sorting, and filtering.

    Args:
        scan_id: Scan UUID
        page: Page number (1-indexed)
        per_page: Results per page (max 100)
        sort_by: Field to sort by
        sort_order: asc or desc
        min_score: Minimum score filter
        stage: Stage filter
        passes_only: Show only passing stocks
        min_eps_growth: Minimum EPS growth Q/Q filter (%)
        min_sales_growth: Minimum sales growth Q/Q filter (%)
        db: Database session

    Returns:
        Paginated scan results
    """
    try:
        # Verify scan exists
        scan = db.query(Scan).filter(Scan.scan_id == scan_id).first()
        if not scan:
            raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")

        # Build query with LEFT JOIN to stock_universe to get company names
        query = db.query(ScanResult, StockUniverse.name).outerjoin(
            StockUniverse, ScanResult.symbol == StockUniverse.symbol
        ).filter(ScanResult.scan_id == scan_id)

        # Apply filters

        # Text search - symbol pattern matching
        if symbol_search:
            query = query.filter(ScanResult.symbol.ilike(f"%{symbol_search}%"))

        # Minervini score range
        if min_score is not None:
            query = query.filter(ScanResult.minervini_score >= min_score)
        if max_score is not None:
            query = query.filter(ScanResult.minervini_score <= max_score)

        # Stage filter
        if stage is not None:
            query = query.filter(ScanResult.stage == stage)

        # Rating filter (comma-separated list)
        if ratings:
            rating_list = [r.strip() for r in ratings.split(',')]
            query = query.filter(ScanResult.rating.in_(rating_list))

        # IBD industry filter (comma-separated list) with include/exclude mode
        if ibd_industries:
            industry_list = [i.strip() for i in ibd_industries.split(',')]
            if ibd_industries_mode == 'exclude':
                query = query.filter(~ScanResult.ibd_industry_group.in_(industry_list))
            else:
                query = query.filter(ScanResult.ibd_industry_group.in_(industry_list))

        # GICS sector filter (comma-separated list) with include/exclude mode
        if gics_sectors:
            sector_list = [s.strip() for s in gics_sectors.split(',')]
            if gics_sectors_mode == 'exclude':
                query = query.filter(~ScanResult.gics_sector.in_(sector_list))
            else:
                query = query.filter(ScanResult.gics_sector.in_(sector_list))

        # Composite score range
        if min_composite is not None:
            query = query.filter(ScanResult.composite_score >= min_composite)
        if max_composite is not None:
            query = query.filter(ScanResult.composite_score <= max_composite)

        # Individual screener score ranges
        if min_canslim is not None:
            query = query.filter(ScanResult.canslim_score >= min_canslim)
        if max_canslim is not None:
            query = query.filter(ScanResult.canslim_score <= max_canslim)
        if min_ipo is not None:
            query = query.filter(ScanResult.ipo_score >= min_ipo)
        if max_ipo is not None:
            query = query.filter(ScanResult.ipo_score <= max_ipo)
        if min_custom is not None:
            query = query.filter(ScanResult.custom_score >= min_custom)
        if max_custom is not None:
            query = query.filter(ScanResult.custom_score <= max_custom)
        if min_vol_breakthrough is not None:
            query = query.filter(ScanResult.volume_breakthrough_score >= min_vol_breakthrough)
        if max_vol_breakthrough is not None:
            query = query.filter(ScanResult.volume_breakthrough_score <= max_vol_breakthrough)

        # RS rating ranges
        if min_rs is not None:
            query = query.filter(ScanResult.rs_rating >= min_rs)
        if max_rs is not None:
            query = query.filter(ScanResult.rs_rating <= max_rs)
        if min_rs_1m is not None:
            query = query.filter(ScanResult.rs_rating_1m >= min_rs_1m)
        if max_rs_1m is not None:
            query = query.filter(ScanResult.rs_rating_1m <= max_rs_1m)
        if min_rs_3m is not None:
            query = query.filter(ScanResult.rs_rating_3m >= min_rs_3m)
        if max_rs_3m is not None:
            query = query.filter(ScanResult.rs_rating_3m <= max_rs_3m)
        if min_rs_12m is not None:
            query = query.filter(ScanResult.rs_rating_12m >= min_rs_12m)
        if max_rs_12m is not None:
            query = query.filter(ScanResult.rs_rating_12m <= max_rs_12m)

        # Price range
        if min_price is not None:
            query = query.filter(ScanResult.price >= min_price)
        if max_price is not None:
            query = query.filter(ScanResult.price <= max_price)

        # Growth metrics filters (Q/Q)
        if min_eps_growth is not None:
            query = query.filter(ScanResult.eps_growth_qq >= min_eps_growth)
        if max_eps_growth is not None:
            query = query.filter(ScanResult.eps_growth_qq <= max_eps_growth)
        if min_sales_growth is not None:
            query = query.filter(ScanResult.sales_growth_qq >= min_sales_growth)
        if max_sales_growth is not None:
            query = query.filter(ScanResult.sales_growth_qq <= max_sales_growth)

        # Growth metrics filters (Y/Y)
        if min_eps_growth_yy is not None:
            query = query.filter(ScanResult.eps_growth_yy >= min_eps_growth_yy)
        if min_sales_growth_yy is not None:
            query = query.filter(ScanResult.sales_growth_yy >= min_sales_growth_yy)

        # Valuation filters
        if max_peg is not None:
            query = query.filter(ScanResult.peg_ratio <= max_peg)

        # EPS Rating filter
        if min_eps_rating is not None:
            query = query.filter(ScanResult.eps_rating >= min_eps_rating)
        if max_eps_rating is not None:
            query = query.filter(ScanResult.eps_rating <= max_eps_rating)

        # Volume & Market Cap filters
        if min_volume is not None:
            query = query.filter(ScanResult.volume >= min_volume)
        if min_market_cap is not None:
            query = query.filter(ScanResult.market_cap >= min_market_cap)

        # VCP filters (from JSON details) - use SQLite json_extract function
        from sqlalchemy import Float as SAFloat, cast, and_, func

        # VCP Score filter
        if min_vcp_score is not None:
            json_val = func.json_extract(ScanResult.details, '$.vcp_score')
            query = query.filter(and_(
                json_val.isnot(None),
                cast(json_val, SAFloat) >= min_vcp_score
            ))
        if max_vcp_score is not None:
            json_val = func.json_extract(ScanResult.details, '$.vcp_score')
            query = query.filter(and_(
                json_val.isnot(None),
                cast(json_val, SAFloat) <= max_vcp_score
            ))

        # VCP Pivot filter
        if min_vcp_pivot is not None:
            json_val = func.json_extract(ScanResult.details, '$.vcp_pivot')
            query = query.filter(and_(
                json_val.isnot(None),
                cast(json_val, SAFloat) >= min_vcp_pivot
            ))
        if max_vcp_pivot is not None:
            json_val = func.json_extract(ScanResult.details, '$.vcp_pivot')
            query = query.filter(and_(
                json_val.isnot(None),
                cast(json_val, SAFloat) <= max_vcp_pivot
            ))

        # VCP Detected filter (boolean)
        # SQLite json_extract returns 1/0 for true/false JSON booleans
        if vcp_detected is not None:
            json_val = func.json_extract(ScanResult.details, '$.vcp_detected')
            query = query.filter(and_(
                json_val.isnot(None),
                json_val == (1 if vcp_detected else 0)
            ))

        # VCP Ready filter (boolean)
        if vcp_ready is not None:
            json_val = func.json_extract(ScanResult.details, '$.vcp_ready_for_breakout')
            query = query.filter(and_(
                json_val.isnot(None),
                json_val == (1 if vcp_ready else 0)
            ))

        # ADR filter (indexed column)
        if min_adr is not None:
            query = query.filter(ScanResult.adr_percent >= min_adr)
        if max_adr is not None:
            query = query.filter(ScanResult.adr_percent <= max_adr)

        # MA alignment filter (from JSON details) - use SQLite json_extract
        if ma_alignment is not None:
            json_val = func.json_extract(ScanResult.details, '$.ma_alignment')
            query = query.filter(and_(
                json_val.isnot(None),
                json_val == (1 if ma_alignment else 0)
            ))

        # Performance filters (indexed columns)
        if min_perf_day is not None:
            query = query.filter(ScanResult.price_change_1d >= min_perf_day)
        if max_perf_day is not None:
            query = query.filter(ScanResult.price_change_1d <= max_perf_day)
        if min_perf_week is not None:
            query = query.filter(ScanResult.perf_week >= min_perf_week)
        if max_perf_week is not None:
            query = query.filter(ScanResult.perf_week <= max_perf_week)
        if min_perf_month is not None:
            query = query.filter(ScanResult.perf_month >= min_perf_month)
        if max_perf_month is not None:
            query = query.filter(ScanResult.perf_month <= max_perf_month)

        # EMA distance filters (indexed columns)
        if min_ema_10 is not None:
            query = query.filter(ScanResult.ema_10_distance >= min_ema_10)
        if max_ema_10 is not None:
            query = query.filter(ScanResult.ema_10_distance <= max_ema_10)
        if min_ema_20 is not None:
            query = query.filter(ScanResult.ema_20_distance >= min_ema_20)
        if max_ema_20 is not None:
            query = query.filter(ScanResult.ema_20_distance <= max_ema_20)
        if min_ema_50 is not None:
            query = query.filter(ScanResult.ema_50_distance >= min_ema_50)
        if max_ema_50 is not None:
            query = query.filter(ScanResult.ema_50_distance <= max_ema_50)

        # 52-week distance filters (indexed columns)
        if min_52w_high is not None:
            query = query.filter(ScanResult.week_52_high_distance >= min_52w_high)
        if max_52w_high is not None:
            query = query.filter(ScanResult.week_52_high_distance <= max_52w_high)
        if min_52w_low is not None:
            query = query.filter(ScanResult.week_52_low_distance >= min_52w_low)
        if max_52w_low is not None:
            query = query.filter(ScanResult.week_52_low_distance <= max_52w_low)

        # IPO date filter (indexed column)
        if ipo_after:
            ipo_cutoff = parse_ipo_after_preset(ipo_after)
            if ipo_cutoff:
                # Filter stocks with ipo_date >= cutoff date
                query = query.filter(
                    ScanResult.ipo_date.isnot(None),
                    ScanResult.ipo_date >= ipo_cutoff
                )

        # Beta and Beta-Adjusted RS filters (indexed columns)
        if min_beta is not None:
            query = query.filter(ScanResult.beta >= min_beta)
        if max_beta is not None:
            query = query.filter(ScanResult.beta <= max_beta)
        if min_beta_adj_rs is not None:
            query = query.filter(ScanResult.beta_adj_rs >= min_beta_adj_rs)
        if max_beta_adj_rs is not None:
            query = query.filter(ScanResult.beta_adj_rs <= max_beta_adj_rs)
        if min_beta_adj_rs_1m is not None:
            query = query.filter(ScanResult.beta_adj_rs_1m >= min_beta_adj_rs_1m)
        if min_beta_adj_rs_3m is not None:
            query = query.filter(ScanResult.beta_adj_rs_3m >= min_beta_adj_rs_3m)
        if min_beta_adj_rs_12m is not None:
            query = query.filter(ScanResult.beta_adj_rs_12m >= min_beta_adj_rs_12m)

        # Qullamaggie extended performance filters (indexed columns)
        if min_perf_3m is not None:
            query = query.filter(ScanResult.perf_3m >= min_perf_3m)
        if max_perf_3m is not None:
            query = query.filter(ScanResult.perf_3m <= max_perf_3m)
        if min_perf_6m is not None:
            query = query.filter(ScanResult.perf_6m >= min_perf_6m)
        if max_perf_6m is not None:
            query = query.filter(ScanResult.perf_6m <= max_perf_6m)

        # Episodic Pivot filters (indexed columns)
        if min_gap_percent is not None:
            query = query.filter(ScanResult.gap_percent >= min_gap_percent)
        if max_gap_percent is not None:
            query = query.filter(ScanResult.gap_percent <= max_gap_percent)
        if min_volume_surge is not None:
            query = query.filter(ScanResult.volume_surge >= min_volume_surge)
        if max_volume_surge is not None:
            query = query.filter(ScanResult.volume_surge <= max_volume_surge)

        if passes_only:
            # Filter by rating or score
            query = query.filter(
                or_(
                    ScanResult.rating.in_(["Strong Buy", "Buy"]),
                    ScanResult.minervini_score >= 70
                )
            )

        # Get total count
        total = query.count()

        # Apply sorting
        # Phase 3.3: rs_rating and stage now have indexed columns
        # For other fields in the details JSON, we need to sort after fetching
        sort_in_sql = True
        sort_field = ScanResult.minervini_score  # Default

        if sort_by == "symbol":
            sort_field = ScanResult.symbol
        elif sort_by == "composite_score":
            sort_field = ScanResult.composite_score
        elif sort_by == "minervini_score":
            sort_field = ScanResult.minervini_score
        elif sort_by == "canslim_score":
            sort_field = ScanResult.canslim_score
        elif sort_by == "ipo_score":
            sort_field = ScanResult.ipo_score
        elif sort_by == "custom_score":
            sort_field = ScanResult.custom_score
        elif sort_by == "volume_breakthrough_score":
            sort_field = ScanResult.volume_breakthrough_score
        elif sort_by == "price" or sort_by == "current_price":
            sort_field = ScanResult.price
        elif sort_by == "price_change_1d":
            sort_field = ScanResult.price_change_1d
        elif sort_by == "rs_rating":
            # Phase 3.3: Now indexed column, can sort in SQL
            sort_field = ScanResult.rs_rating
        elif sort_by == "rs_rating_1m":
            sort_field = ScanResult.rs_rating_1m
        elif sort_by == "rs_rating_3m":
            sort_field = ScanResult.rs_rating_3m
        elif sort_by == "rs_rating_12m":
            sort_field = ScanResult.rs_rating_12m
        elif sort_by == "stage":
            # Phase 3.3: Now indexed column, can sort in SQL
            sort_field = ScanResult.stage
        elif sort_by == "eps_growth_qq":
            # Growth metric indexed column
            sort_field = ScanResult.eps_growth_qq
        elif sort_by == "sales_growth_qq":
            # Growth metric indexed column
            sort_field = ScanResult.sales_growth_qq
        elif sort_by == "eps_growth_yy":
            # YoY EPS growth indexed column
            sort_field = ScanResult.eps_growth_yy
        elif sort_by == "sales_growth_yy":
            # YoY sales growth indexed column
            sort_field = ScanResult.sales_growth_yy
        elif sort_by == "peg_ratio" or sort_by == "peg":
            # PEG ratio indexed column
            sort_field = ScanResult.peg_ratio
        elif sort_by == "eps_rating":
            # EPS Rating indexed column
            sort_field = ScanResult.eps_rating
        elif sort_by == "ibd_industry_group":
            # Industry classification indexed column
            sort_field = ScanResult.ibd_industry_group
        elif sort_by == "ibd_group_rank":
            # IBD group rank indexed column
            sort_field = ScanResult.ibd_group_rank
        elif sort_by == "gics_sector":
            # Industry classification indexed column
            sort_field = ScanResult.gics_sector
        elif sort_by == "rs_trend":
            sort_field = ScanResult.rs_trend
        elif sort_by == "volume":
            sort_field = ScanResult.volume
        elif sort_by == "market_cap":
            sort_field = ScanResult.market_cap
        elif sort_by == "ipo_date":
            # IPO date indexed column
            sort_field = ScanResult.ipo_date
        elif sort_by == "beta":
            # Beta indexed column
            sort_field = ScanResult.beta
        elif sort_by == "beta_adj_rs":
            # Beta-adjusted RS indexed column
            sort_field = ScanResult.beta_adj_rs
        elif sort_by == "beta_adj_rs_1m":
            sort_field = ScanResult.beta_adj_rs_1m
        elif sort_by == "beta_adj_rs_3m":
            sort_field = ScanResult.beta_adj_rs_3m
        elif sort_by == "beta_adj_rs_12m":
            sort_field = ScanResult.beta_adj_rs_12m
        elif sort_by in ["stage_name", "ma_alignment", "vcp_detected", "passes_template", "adr_percent"]:
            # These are still in JSON details - need to sort after fetching
            sort_in_sql = False

        if sort_in_sql:
            # Can sort and paginate in SQL if no JSON field filtering needed
            if sort_order == "asc":
                query = query.order_by(asc(sort_field))
            else:
                query = query.order_by(desc(sort_field))

            # Apply pagination for SQL sorting
            offset = (page - 1) * per_page
            query = query.offset(offset).limit(per_page)
            results_with_names = query.all()
        else:
            # For JSON fields, limit fetch to prevent memory issues, then sort in Python
            # Limit to 1000 rows max for performance (reasonable for UI display)
            # Phase 3.3: No stage filtering needed here anymore (handled in SQL)
            results_with_names = query.limit(1000).all()

            # Sort in Python by the JSON field
            def get_sort_key(result_tuple):
                result, company_name = result_tuple
                detail_value = result.details.get(sort_by) if result.details else None
                # Handle None values - put them last
                if detail_value is None:
                    return float('-inf') if sort_order == "desc" else float('inf')
                return detail_value

            results_with_names = sorted(results_with_names, key=get_sort_key, reverse=(sort_order == "desc"))

            # Apply pagination after sorting
            offset = (page - 1) * per_page
            results_with_names = results_with_names[offset:offset + per_page]

        # Convert to response model
        result_items = []
        for result, company_name in results_with_names:
            details = result.details or {}

            item = ScanResultItem(
                symbol=result.symbol,
                company_name=company_name,  # Company name from stock_universe
                composite_score=result.composite_score or 0,
                minervini_score=result.minervini_score,
                canslim_score=result.canslim_score,
                ipo_score=result.ipo_score,
                custom_score=result.custom_score,
                volume_breakthrough_score=result.volume_breakthrough_score,
                rs_rating=result.rs_rating,
                rs_rating_1m=result.rs_rating_1m,
                rs_rating_3m=result.rs_rating_3m,
                rs_rating_12m=result.rs_rating_12m,
                stage=result.stage,
                stage_name=details.get('stage_name'),
                current_price=result.price,
                volume=result.volume,
                market_cap=result.market_cap,
                ma_alignment=details.get('ma_alignment'),
                vcp_detected=details.get('vcp_detected'),
                vcp_score=details.get('vcp_score'),
                vcp_pivot=details.get('vcp_pivot'),
                vcp_ready_for_breakout=details.get('vcp_ready_for_breakout'),
                vcp_contraction_ratio=details.get('vcp_contraction_ratio'),
                vcp_atr_score=details.get('vcp_atr_score'),
                passes_template=details.get('passes_template', False),
                rating=result.rating or "Pass",
                # Growth metrics - use indexed columns for better performance
                adr_percent=result.adr_percent,
                eps_growth_qq=result.eps_growth_qq,
                sales_growth_qq=result.sales_growth_qq,
                eps_growth_yy=result.eps_growth_yy,
                sales_growth_yy=result.sales_growth_yy,
                # Valuation metrics
                peg_ratio=result.peg_ratio,
                # EPS Rating
                eps_rating=result.eps_rating,
                # Multi-screener metadata
                screeners_run=details.get('screeners_run'),
                # Phase 4: Industry classifications
                ibd_industry_group=result.ibd_industry_group,
                ibd_group_rank=result.ibd_group_rank,
                gics_sector=result.gics_sector,
                gics_industry=result.gics_industry,
                # RS Sparkline data (conditionally included for performance)
                rs_sparkline_data=result.rs_sparkline_data if include_sparklines else None,
                rs_trend=result.rs_trend,
                # Price Sparkline data (conditionally included for performance)
                price_sparkline_data=result.price_sparkline_data if include_sparklines else None,
                price_change_1d=result.price_change_1d,
                price_trend=result.price_trend,
                # IPO date
                ipo_date=result.ipo_date,
                # Beta and Beta-Adjusted RS metrics
                beta=result.beta,
                beta_adj_rs=result.beta_adj_rs,
                beta_adj_rs_1m=result.beta_adj_rs_1m,
                beta_adj_rs_3m=result.beta_adj_rs_3m,
                beta_adj_rs_12m=result.beta_adj_rs_12m,
            )
            result_items.append(item)

        # Calculate pages
        pages = (total + per_page - 1) // per_page

        return ScanResultsResponse(
            scan_id=scan_id,
            total=total,
            page=page,
            per_page=per_page,
            pages=pages,
            results=result_items
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting scan results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting scan results: {str(e)}")


@router.delete("/{scan_id}")
async def delete_scan(
    scan_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a scan and all its results.

    Args:
        scan_id: Scan UUID
        db: Database session

    Returns:
        Success message
    """
    try:
        scan = db.query(Scan).filter(Scan.scan_id == scan_id).first()
        if not scan:
            raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")

        # Delete results first (foreign key constraint)
        db.query(ScanResult).filter(ScanResult.scan_id == scan_id).delete()

        # Delete scan
        db.delete(scan)
        db.commit()

        logger.info(f"Deleted scan {scan_id}")

        return {'message': f'Scan {scan_id} deleted successfully'}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting scan: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting scan: {str(e)}")


@router.get("/{scan_id}/export")
async def export_scan_results(
    scan_id: str,
    format: str = Query(default="csv", regex="^(csv|excel)$"),
    min_score: Optional[float] = Query(default=None, ge=0, le=100),
    stage: Optional[int] = Query(default=None, ge=1, le=4),
    passes_only: Optional[bool] = Query(default=None),
    db: Session = Depends(get_db)
):
    """
    Export scan results to CSV or Excel format.

    Args:
        scan_id: Scan UUID
        format: Export format (csv or excel)
        min_score: Minimum Minervini score filter
        stage: Stage filter (1-4)
        passes_only: Filter for passing stocks only
        db: Database session

    Returns:
        File download (CSV or Excel)
    """
    try:
        # Verify scan exists
        scan = db.query(Scan).filter(Scan.scan_id == scan_id).first()
        if not scan:
            raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")

        # Build query with filters
        query = db.query(ScanResult).filter(ScanResult.scan_id == scan_id)

        if min_score is not None:
            query = query.filter(ScanResult.minervini_score >= min_score)

        if stage is not None:
            query = query.filter(ScanResult.details['stage'].astext.cast(db.Integer) == stage)

        if passes_only:
            query = query.filter(ScanResult.details['passes_template'].astext.cast(db.Boolean) == True)

        # Order by composite score descending (fallback to minervini for old scans)
        query = query.order_by(desc(ScanResult.composite_score))

        results = query.all()

        if format == "csv":
            # Create CSV in memory
            output = io.StringIO()
            writer = csv.writer(output)

            # Write header
            writer.writerow([
                'Symbol',
                'Composite Score',
                'Minervini Score',
                'CANSLIM Score',
                'IPO Score',
                'Custom Score',
                'Volume Breakthrough Score',
                'RS Rating',
                'RS Rating 1M',
                'RS Rating 3M',
                'RS Rating 12M',
                'Stage',
                'Stage Name',
                'Current Price',
                'MA Alignment',
                'VCP Detected',
                'VCP Score',
                'VCP Pivot',
                'VCP Ready',
                'VCP Contraction',
                'VCP ATR Score',
                'Passes Template',
                'Rating',
                '52W Position',
                'Volume Trend',
                'ADR %',
                'EPS Growth Q/Q %',
                'Sales Growth Q/Q %'
            ])

            # Helper function to format percentage values
            def format_pct(value):
                """Format percentage or return N/A"""
                if value is None:
                    return 'N/A'
                return f"{round(value, 2)}%"

            # Write data rows
            for result in results:
                details = result.details or {}
                writer.writerow([
                    result.symbol,
                    round(result.composite_score, 2) if result.composite_score else '',
                    round(result.minervini_score, 2) if result.minervini_score else '',
                    round(result.canslim_score, 2) if result.canslim_score else '',
                    round(result.ipo_score, 2) if result.ipo_score else '',
                    round(result.custom_score, 2) if result.custom_score else '',
                    round(result.volume_breakthrough_score, 2) if result.volume_breakthrough_score else '',
                    round(details.get('rs_rating', 0), 2) if details.get('rs_rating') else '',
                    round(result.rs_rating_1m, 2) if result.rs_rating_1m else '',
                    round(result.rs_rating_3m, 2) if result.rs_rating_3m else '',
                    round(result.rs_rating_12m, 2) if result.rs_rating_12m else '',
                    details.get('stage', ''),
                    details.get('stage_name', ''),
                    round(details.get('current_price', 0), 2) if details.get('current_price') else '',
                    'Yes' if details.get('ma_alignment') else 'No',
                    'Yes' if details.get('vcp_detected') else 'No',
                    round(details.get('vcp_score', 0), 2) if details.get('vcp_score') else '',
                    round(details.get('vcp_pivot', 0), 2) if details.get('vcp_pivot') else '',
                    'Yes' if details.get('vcp_ready_for_breakout') else 'No',
                    round(details.get('vcp_contraction_ratio', 0), 2) if details.get('vcp_contraction_ratio') else '',
                    round(details.get('vcp_atr_score', 0), 2) if details.get('vcp_atr_score') else '',
                    'Yes' if details.get('passes_template') else 'No',
                    result.rating or '',
                    round(details.get('position_52week', 0), 2) if details.get('position_52week') else '',
                    details.get('volume_trend', ''),
                    format_pct(result.adr_percent),
                    format_pct(details.get('eps_growth_qq')),
                    format_pct(details.get('sales_growth_qq')),
                ])

            # Prepare response
            output.seek(0)
            filename = f"scan_{scan_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"'
                }
            )

        else:  # Excel format
            # For now, return CSV with .xlsx extension
            # To implement true Excel, would need to use openpyxl or xlsxwriter
            raise HTTPException(
                status_code=501,
                detail="Excel export not yet implemented. Please use CSV format."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting scan results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error exporting scan results: {str(e)}")


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
    db: Session = Depends(get_db)
):
    """
    Get all stocks in the same IBD industry group as the given symbol.

    Returns ALL stocks in the industry group that were scanned,
    not just those that passed the scan criteria.

    Args:
        scan_id: Scan UUID
        symbol: Stock symbol to find peers for
        db: Database session

    Returns:
        List of peer stocks with their metrics
    """
    try:
        # Get the symbol's industry group
        stock = db.query(ScanResult).filter(
            ScanResult.scan_id == scan_id,
            ScanResult.symbol == symbol.upper()
        ).first()

        if not stock:
            raise HTTPException(
                status_code=404,
                detail=f"Stock {symbol} not found in scan {scan_id}"
            )

        if not stock.ibd_industry_group:
            logger.info(f"No industry group for {symbol}, returning empty list")
            return []  # No industry group data

        # Get all stocks in the same industry group
        peers = db.query(ScanResult, StockUniverse.name).outerjoin(
            StockUniverse, ScanResult.symbol == StockUniverse.symbol
        ).filter(
            ScanResult.scan_id == scan_id,
            ScanResult.ibd_industry_group == stock.ibd_industry_group
        ).order_by(
            ScanResult.composite_score.desc()
        ).all()

        # Convert to response model
        result_items = []
        for result, company_name in peers:
            details = result.details or {}

            item = ScanResultItem(
                symbol=result.symbol,
                company_name=company_name,
                composite_score=result.composite_score or 0,
                minervini_score=result.minervini_score,
                canslim_score=result.canslim_score,
                ipo_score=result.ipo_score,
                custom_score=result.custom_score,
                volume_breakthrough_score=result.volume_breakthrough_score,
                rs_rating=result.rs_rating,
                rs_rating_1m=result.rs_rating_1m,
                rs_rating_3m=result.rs_rating_3m,
                rs_rating_12m=result.rs_rating_12m,
                stage=result.stage,
                stage_name=details.get('stage_name'),
                current_price=result.price,
                volume=result.volume,
                market_cap=result.market_cap,
                ma_alignment=details.get('ma_alignment'),
                vcp_detected=details.get('vcp_detected'),
                vcp_score=details.get('vcp_score'),
                vcp_pivot=details.get('vcp_pivot'),
                vcp_ready_for_breakout=details.get('vcp_ready_for_breakout'),
                vcp_contraction_ratio=details.get('vcp_contraction_ratio'),
                vcp_atr_score=details.get('vcp_atr_score'),
                passes_template=details.get('passes_template', False),
                rating=result.rating or "Pass",
                adr_percent=result.adr_percent,
                eps_growth_qq=details.get('eps_growth_qq'),
                sales_growth_qq=details.get('sales_growth_qq'),
                eps_rating=result.eps_rating,
                screeners_run=details.get('screeners_run'),
                ibd_industry_group=result.ibd_industry_group,
                ibd_group_rank=result.ibd_group_rank,
                gics_sector=result.gics_sector,
                gics_industry=result.gics_industry,
                # RS Sparkline data
                rs_sparkline_data=result.rs_sparkline_data,
                rs_trend=result.rs_trend,
                # Price Sparkline data
                price_sparkline_data=result.price_sparkline_data,
                price_change_1d=result.price_change_1d,
                price_trend=result.price_trend,
                # IPO date
                ipo_date=result.ipo_date,
                # Beta and Beta-Adjusted RS metrics
                beta=result.beta,
                beta_adj_rs=result.beta_adj_rs,
                beta_adj_rs_1m=result.beta_adj_rs_1m,
                beta_adj_rs_3m=result.beta_adj_rs_3m,
                beta_adj_rs_12m=result.beta_adj_rs_12m,
            )
            result_items.append(item)

        logger.info(
            f"Found {len(result_items)} peers for {symbol} "
            f"in group '{stock.ibd_industry_group}'"
        )

        return result_items

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting peers: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting peers: {str(e)}"
        )
