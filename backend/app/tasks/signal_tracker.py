"""Nightly task to track open signal outcomes."""
from __future__ import annotations

import logging
from datetime import date

from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task(name="app.tasks.signal_tracker.track_open_signals")
def track_open_signals() -> dict:
    """Update outcome for all open signals in signal_archive."""
    from ..database import SessionLocal
    from ..models.signal_archive import SignalArchive

    db = SessionLocal()
    open_signals = db.query(SignalArchive).filter(SignalArchive.outcome == "open").all()
    updated = 0

    for sig in open_signals:
        try:
            import yfinance as yf
            ticker = yf.Ticker(sig.symbol)
            hist = ticker.history(period="2d", interval="1d")
            if hist.empty:
                continue
            current_price = float(hist["Close"].iloc[-1])
            sig.days_held = (date.today() - sig.signal_date).days

            if sig.stop_loss and current_price <= sig.stop_loss:
                sig.outcome = "stop_hit"
                sig.outcome_price = sig.stop_loss
                sig.outcome_date = date.today()
                sig.pct_return = round((sig.stop_loss - sig.entry_price) / sig.entry_price * 100, 2)
            elif sig.target_price and current_price >= sig.target_price:
                sig.outcome = "target_hit"
                sig.outcome_price = current_price
                sig.outcome_date = date.today()
                sig.pct_return = round((current_price - sig.entry_price) / sig.entry_price * 100, 2)
            updated += 1
        except Exception as e:
            logger.warning("track_open_signals: error for %s: %s", sig.symbol, e)

    db.commit()
    db.close()
    return {"updated": updated, "total_open": len(open_signals)}
