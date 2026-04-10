"""
Ticker Validation Report API endpoints.

Provides access to invalid ticker logs for manual review.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional
import logging

from ...database import get_db
from ...wiring.bootstrap import get_ticker_validation_service
from ...schemas.ticker_validation import (
    ValidationReportResponse,
    ValidationSummary,
    SymbolHistoryResponse,
    ResolveRequest,
    ResolveResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/report", response_model=ValidationReportResponse)
async def get_validation_report(
    limit: int = Query(100, ge=1, le=500, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    error_type: Optional[str] = Query(
        None,
        regex="^(no_data|delisted|api_error|invalid_response|empty_info)$",
        description="Filter by error type"
    ),
    triggered_by: Optional[str] = Query(
        None,
        description="Filter by trigger source (e.g., fundamentals_refresh, cache_warmup)"
    ),
    min_failures: int = Query(1, ge=1, description="Minimum consecutive failures to include"),
    days_back: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    db: Session = Depends(get_db)
):
    """
    Get paginated list of unresolved ticker validation failures.

    Use this to review tickers that failed during data fetching.
    Tickers remain active - manual deactivation required if confirmed invalid.
    """
    try:
        ticker_validation_service = get_ticker_validation_service()
        failures = ticker_validation_service.get_unresolved_failures(
            db=db,
            limit=limit,
            offset=offset,
            error_type=error_type,
            triggered_by=triggered_by,
            min_consecutive_failures=min_failures,
            days_back=days_back,
        )

        return ValidationReportResponse(
            count=len(failures),
            offset=offset,
            limit=limit,
            failures=failures
        )

    except Exception as e:
        logger.error(f"Error getting validation report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary", response_model=ValidationSummary)
async def get_validation_summary(
    days_back: int = Query(7, ge=1, le=90, description="Number of days to include in summary"),
    db: Session = Depends(get_db)
):
    """
    Get summary statistics of ticker validation failures.

    Returns aggregate counts by error type, data source, and trigger.
    """
    try:
        ticker_validation_service = get_ticker_validation_service()
        summary = ticker_validation_service.get_failure_summary(db=db, days_back=days_back)
        return summary

    except Exception as e:
        logger.error(f"Error getting validation summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/symbol/{symbol}", response_model=SymbolHistoryResponse)
async def get_symbol_history(
    symbol: str,
    limit: int = Query(20, ge=1, le=100, description="Maximum results to return"),
    db: Session = Depends(get_db)
):
    """
    Get validation failure history for a specific symbol.

    Useful for investigating why a particular ticker keeps failing.
    """
    try:
        ticker_validation_service = get_ticker_validation_service()
        history = ticker_validation_service.get_symbol_history(
            db=db,
            symbol=symbol.upper(),
            limit=limit
        )

        return SymbolHistoryResponse(
            symbol=symbol.upper(),
            history=history
        )

    except Exception as e:
        logger.error(f"Error getting symbol history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resolve/{log_id}", response_model=ResolveResponse)
async def resolve_failure(
    log_id: int,
    request: ResolveRequest,
    db: Session = Depends(get_db)
):
    """
    Mark a validation failure as resolved.

    Use after manual investigation to clear the report.
    """
    try:
        ticker_validation_service = get_ticker_validation_service()
        success = ticker_validation_service.resolve_failure(
            db=db,
            log_id=log_id,
            resolution_notes=request.resolution_notes
        )

        if success:
            return ResolveResponse(message=f"Failure {log_id} marked as resolved")
        else:
            raise HTTPException(status_code=404, detail=f"Log entry {log_id} not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving failure: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resolve-symbol/{symbol}", response_model=ResolveResponse)
async def resolve_all_for_symbol(
    symbol: str,
    request: ResolveRequest,
    db: Session = Depends(get_db)
):
    """
    Resolve all unresolved failures for a symbol.

    Use after confirming a ticker is valid (perhaps after temporary API issues).
    """
    try:
        ticker_validation_service = get_ticker_validation_service()
        count = ticker_validation_service.bulk_resolve_by_symbol(
            db=db,
            symbol=symbol.upper(),
            resolution_notes=request.resolution_notes
        )

        return ResolveResponse(
            message=f"Resolved {count} failures for {symbol.upper()}",
            count=count
        )

    except Exception as e:
        logger.error(f"Error bulk resolving: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
