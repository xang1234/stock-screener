"""Daily ATM implied-volatility history, used to compute IV Rank (IVR).

Previously this was a rolling ~252-session hash in Redis (7-30 day TTL keys
elsewhere in this codebase made that an easy footgun -- a Redis flush or
cache-clear script silently reset every ticker's IV Rank to "building up
history again"). Persisting to Postgres instead makes the history durable
across cache clears/restarts, consistent with how max_pain_snapshots and
gex_snapshots already persist their daily options data.
"""

from datetime import datetime

from sqlalchemy import Column, Date, DateTime, Float, Index, Integer, String, UniqueConstraint

from ..database import Base


class IvHistory(Base):
    """One row per ticker per trading day: that day's ATM implied volatility.

    IV Rank is `(current_iv - min(atm_iv)) / (max(atm_iv) - min(atm_iv))` over
    the trailing ~252 rows for a ticker -- see compute_ivr() in
    options_metrics.py.
    """

    __tablename__ = "iv_history"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(20), index=True, nullable=False)
    trading_date = Column(Date, nullable=False)
    atm_iv = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("ticker", "trading_date", name="uq_iv_history_ticker_trading_date"),
        Index("ix_iv_history_ticker_trading_date", "ticker", "trading_date"),
    )
