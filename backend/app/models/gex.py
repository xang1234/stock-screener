"""GEX snapshot models for storing options gamma exposure data."""

from datetime import datetime

from sqlalchemy import Column, Date, DateTime, Float, Index, Integer, String

from ..database import Base


class GexSnapshot(Base):
    """Stores a single GEX snapshot for a ticker at a specific time."""

    __tablename__ = "gex_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(20), index=True, nullable=False)
    company_name = Column(String(255), nullable=True)
    status = Column(String(20), default="OK", nullable=False)  # OK, FAILED
    error = Column(String(1000), nullable=True)

    expiration = Column(Date, nullable=True)
    spot_price = Column(Float, nullable=True)
    call_oi = Column(Integer, nullable=True)
    put_oi = Column(Integer, nullable=True)
    call_gex = Column(Float, nullable=True)
    put_gex = Column(Float, nullable=True)
    total_gex = Column(Float, nullable=True)
    flip_level = Column(Float, nullable=True)
    distance_to_flip_pct = Column(Float, nullable=True)

    fetched_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    # US/Eastern trading day this snapshot belongs to (derived from fetched_at
    # at write time, not from a raw UTC date_trunc -- see migration 20260805_0028).
    # Used for weekly/monthly/quarterly/yearly history rollups.
    trading_date = Column(Date, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    batch_id = Column(String(50), nullable=True, index=True)

    __table_args__ = (
        Index("ix_gex_ticker_fetched", "ticker", "fetched_at"),
        Index("ix_gex_batch_fetched", "batch_id", "fetched_at"),
        Index("ix_gex_snapshots_ticker_trading_date", "ticker", "trading_date"),
    )


class GexBatch(Base):
    """Tracks batch runs of the GEX analysis."""

    __tablename__ = "gex_batches"

    id = Column(String(50), primary_key=True, index=True)
    status = Column(String(20), default="running", nullable=False)  # running, completed, failed
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    # Celery task ID of the update_gex run that owns this row -- lets the
    # Operations "Force Stop" cancel path find and fail this row when the
    # task is revoked, instead of leaving it stuck at status='running'
    # forever (see migration 20260805_0029).
    celery_task_id = Column(String(155), nullable=True, index=True)

    tickers_total = Column(Integer, default=0)
    tickers_ok = Column(Integer, default=0)
    tickers_failed = Column(Integer, default=0)
    avg_total_gex = Column(Float, nullable=True)
    closest_to_flip = Column(String(20), nullable=True)

    strike_range_pct = Column(Float, default=20.0)
    max_strikes = Column(Integer, nullable=True)

    error_message = Column(String(1000), nullable=True)

    __table_args__ = (
        Index("ix_gex_batch_status", "status"),
        Index("ix_gex_batch_completed", "completed_at"),
    )
