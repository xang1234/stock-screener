"""Market regime gate — suppresses buy signals when SPY or breadth is unfavorable."""
from __future__ import annotations

import logging
from typing import Dict

import yfinance as yf
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def get_market_regime(db: Session) -> Dict:
    """Return regime dict including buy_allowed flag."""
    try:
        spy_data = yf.Ticker("SPY").history(period="1y", interval="1d")
        if spy_data.empty:
            raise ValueError("No SPY data returned")
        from .phase_service import classify_phase
        spy_phase_info = classify_phase(spy_data)
        spy_phase = spy_phase_info.get("phase", 0)
    except Exception as e:
        logger.warning("Could not fetch SPY for regime gate: %s", e)
        spy_phase = 2

    try:
        from ..models.stock import StockTechnical
        total = db.query(StockTechnical).count()
        phase2_count = db.query(StockTechnical).filter(StockTechnical.stage == 2).count()
        pct_phase2 = phase2_count / total if total > 0 else 0.0
    except Exception as e:
        logger.warning("Could not query StockTechnical for breadth: %s", e)
        pct_phase2 = 1.0

    buy_allowed = spy_phase == 2 and pct_phase2 >= 0.15

    return {
        "spy_phase": spy_phase,
        "pct_stocks_phase2": round(pct_phase2, 3),
        "buy_allowed": buy_allowed,
        "reason": None if buy_allowed else f"SPY Phase {spy_phase}, {pct_phase2:.0%} stocks in Phase 2",
    }
