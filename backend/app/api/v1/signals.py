"""Signal archive API endpoints."""
from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from ...database import get_db
from ...models.signal_archive import SignalArchive

router = APIRouter()


@router.get("/signals")
def list_signals(
    db: Session = Depends(get_db),
    outcome: Optional[str] = Query(None, description="Filter by outcome: open, stop_hit, target_hit"),
    screener: Optional[str] = Query(None),
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
):
    query = db.query(SignalArchive).order_by(desc(SignalArchive.signal_date))
    if outcome:
        query = query.filter(SignalArchive.outcome == outcome)
    if screener:
        query = query.filter(SignalArchive.screener == screener)
    total = query.count()
    rows = query.offset(offset).limit(limit).all()
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "signals": [
            {
                "id": r.id,
                "symbol": r.symbol,
                "signal_date": r.signal_date.isoformat() if r.signal_date else None,
                "entry_price": r.entry_price,
                "stop_loss": r.stop_loss,
                "target_price": r.target_price,
                "screener": r.screener,
                "composite_score": r.composite_score,
                "sector": r.sector,
                "stage": r.stage,
                "outcome": r.outcome,
                "outcome_date": r.outcome_date.isoformat() if r.outcome_date else None,
                "outcome_price": r.outcome_price,
                "pct_return": r.pct_return,
                "days_held": r.days_held,
            }
            for r in rows
        ],
    }


@router.get("/signals/stats")
def signal_stats(db: Session = Depends(get_db)):
    total = db.query(SignalArchive).count()
    open_count = db.query(SignalArchive).filter(SignalArchive.outcome == "open").count()
    stop_count = db.query(SignalArchive).filter(SignalArchive.outcome == "stop_hit").count()
    target_count = db.query(SignalArchive).filter(SignalArchive.outcome == "target_hit").count()

    closed = db.query(SignalArchive).filter(SignalArchive.pct_return.isnot(None)).all()
    avg_return = sum(r.pct_return for r in closed) / len(closed) if closed else None
    win_rate = sum(1 for r in closed if r.pct_return and r.pct_return > 0) / len(closed) if closed else None

    return {
        "total": total,
        "open": open_count,
        "stop_hit": stop_count,
        "target_hit": target_count,
        "closed_count": len(closed),
        "avg_return_pct": round(avg_return, 2) if avg_return is not None else None,
        "win_rate": round(win_rate * 100, 1) if win_rate is not None else None,
    }
