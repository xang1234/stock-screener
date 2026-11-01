"""Validation loop API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ...database import get_db
from ...schemas.validation import ValidationOverviewResponse, ValidationSourceKind
from ...services.benchmark_registry_service import benchmark_registry
from ...services.validation_service import ValidationService

router = APIRouter()


def _get_validation_service() -> ValidationService:
    return ValidationService()


@router.get("/overview", response_model=ValidationOverviewResponse)
def get_validation_overview(
    source_kind: ValidationSourceKind = Query(ValidationSourceKind.SCAN_PICK),
    lookback_days: int = Query(90, ge=30, le=365),
    market: str = Query("US"),
    db: Session = Depends(get_db),
    service: ValidationService = Depends(_get_validation_service),
):
    """Return deterministic validation metrics for one supported signal source."""

    normalized_market = market.strip().upper()
    supported = benchmark_registry.supported_markets()
    if normalized_market not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported market '{market}'. Expected one of: {', '.join(supported)}.",
        )
    return service.get_overview(
        db,
        source_kind=source_kind,
        lookback_days=lookback_days,
        market=normalized_market,
    )
