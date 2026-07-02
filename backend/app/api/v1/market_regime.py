"""Market regime and sector health endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...database import get_db

router = APIRouter()


@router.get("/market/regime")
def market_regime(db: Session = Depends(get_db)):
    from ...services.market_regime import get_market_regime
    return get_market_regime(db)


@router.get("/market/sector-health")
def sector_health(db: Session = Depends(get_db)):
    from ...services.sector_phase_service import get_sector_phase_health
    return get_sector_phase_health(db)
