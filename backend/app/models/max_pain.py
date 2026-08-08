"""
Max Pain snapshot models for storing options analysis data.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Index
from sqlalchemy.orm import relationship

from ..database import Base


class MaxPainSnapshot(Base):
    """Stores a single max pain snapshot for a ticker at a specific time"""
    __tablename__ = "max_pain_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(10), index=True, nullable=False)
    company_name = Column(String(255), nullable=True)
    status = Column(String(20), default="OK", nullable=False)  # OK, FAILED
    error = Column(String(500), nullable=True)
    
    # Options data
    max_pain_strike = Column(Float, nullable=True)
    expiration = Column(Date, nullable=True)
    call_oi = Column(Integer, nullable=True)
    put_oi = Column(Integer, nullable=True)
    put_call_ratio = Column(Float, nullable=True)
    last_price = Column(Float, nullable=True)
    distance_pct = Column(Float, nullable=True)

    # Metadata
    fetched_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    # US/Eastern trading day this snapshot belongs to (derived from fetched_at
    # at write time, not from a raw UTC date_trunc -- see migration 20260805_0028).
    # Used for weekly/monthly/quarterly/yearly history rollups.
    trading_date = Column(Date, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    batch_id = Column(String(50), nullable=True, index=True)  # Group snapshots by batch run

    # Indexes for common queries
    __table_args__ = (
        Index('ix_max_pain_ticker_fetched', 'ticker', 'fetched_at'),
        Index('ix_max_pain_batch_fetched', 'batch_id', 'fetched_at'),
        Index('ix_max_pain_snapshots_ticker_trading_date', 'ticker', 'trading_date'),
    )


class MaxPainBatch(Base):
    """Tracks batch runs of the max pain analysis"""
    __tablename__ = "max_pain_batches"

    id = Column(String(50), primary_key=True, index=True)
    status = Column(String(20), default="running", nullable=False)  # running, completed, failed
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    # Celery task ID of the update_max_pain run that owns this row -- lets
    # the Operations "Force Stop" cancel path find and fail this row when
    # the task is revoked, instead of leaving it stuck at status='running'
    # forever (see migration 20260805_0029).
    celery_task_id = Column(String(155), nullable=True, index=True)

    # Summary stats
    tickers_ok = Column(Integer, default=0)
    tickers_failed = Column(Integer, default=0)
    avg_put_call_ratio = Column(Float, nullable=True)
    closest_to_max_pain = Column(String(10), nullable=True)
    
    # Config used
    strike_range_pct = Column(Float, default=20.0)
    max_strikes = Column(Integer, default=30)
    
    # Error tracking
    error_message = Column(String(1000), nullable=True)
    
    __table_args__ = (
        Index('ix_max_pain_batch_status', 'status'),
        Index('ix_max_pain_batch_completed', 'completed_at'),
    )
