"""Stock universe database models for scannable symbol lifecycle management."""
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    Index,
    Text,
    UniqueConstraint,
)
from sqlalchemy.sql import func
from ..database import Base

UNIVERSE_STATUS_ACTIVE = "active"
UNIVERSE_STATUS_INACTIVE_MISSING_SOURCE = "inactive_missing_source"
UNIVERSE_STATUS_INACTIVE_NO_DATA = "inactive_no_data"
UNIVERSE_STATUS_INACTIVE_MANUAL = "inactive_manual"


class StockUniverse(Base):
    """
    Stock universe table storing all scannable stocks.

    Supports fetching from finviz and manual additions.
    Users can activate/deactivate symbols for scanning.
    """

    __tablename__ = "stock_universe"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, unique=True, index=True)
    name = Column(String(255))
    market = Column(String(8), nullable=False, default="US", index=True)  # US, HK, JP, TW
    exchange = Column(String(20), index=True)  # NYSE, NASDAQ, AMEX
    currency = Column(String(8), nullable=False, default="USD")
    timezone = Column(String(64), nullable=False, default="America/New_York")
    local_code = Column(String(32), nullable=True)  # Exchange-local identifier when different from symbol
    sector = Column(String(100), index=True)
    industry = Column(String(100))
    market_cap = Column(Float, nullable=True)
    is_active = Column(Boolean, default=True, index=True)  # Derived from lifecycle status
    status = Column(String(32), nullable=False, default=UNIVERSE_STATUS_ACTIVE, index=True)
    status_reason = Column(String(255))
    is_sp500 = Column(Boolean, default=False, index=True)  # S&P 500 membership
    source = Column(String(20), default="finviz")  # finviz, manual
    added_at = Column(DateTime(timezone=True), server_default=func.now())
    first_seen_at = Column(DateTime(timezone=True), server_default=func.now())
    last_seen_in_source_at = Column(DateTime(timezone=True))
    deactivated_at = Column(DateTime(timezone=True))
    consecutive_fetch_failures = Column(Integer, nullable=False, default=0)
    last_fetch_success_at = Column(DateTime(timezone=True))
    last_fetch_failure_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_universe_exchange_active", "exchange", "is_active"),
        Index("idx_universe_sector_active", "sector", "is_active"),
        Index("idx_universe_exchange_status", "exchange", "status"),
        Index("idx_universe_status_active", "status", "is_active"),
    )

    def __repr__(self):
        return (
            f"<StockUniverse(symbol='{self.symbol}', exchange='{self.exchange}', "
            f"status='{self.status}', active={self.is_active})>"
        )

    @classmethod
    def active_filter(cls):
        """Return the authoritative DB predicate for active-universe membership."""
        return cls.is_active.is_(True)


class StockUniverseStatusEvent(Base):
    """Audit log for universe lifecycle state transitions."""

    __tablename__ = "stock_universe_status_events"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    old_status = Column(String(32), nullable=True)
    new_status = Column(String(32), nullable=False, index=True)
    trigger_source = Column(String(64), nullable=False)
    reason = Column(String(255))
    payload_json = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    __table_args__ = (
        Index("idx_universe_status_events_symbol_created", "symbol", "created_at"),
        Index("idx_universe_status_events_status_created", "new_status", "created_at"),
    )


class StockUniverseReconciliationRun(Base):
    """Immutable reconciliation artifact metadata per market snapshot."""

    __tablename__ = "stock_universe_reconciliation_runs"

    id = Column(Integer, primary_key=True, index=True)
    market = Column(String(8), nullable=False, index=True)
    source_name = Column(String(64), nullable=False)
    snapshot_id = Column(String(128), nullable=False, index=True)
    previous_snapshot_id = Column(String(128), nullable=True)
    total_current = Column(Integer, nullable=False, default=0)
    total_previous = Column(Integer, nullable=False, default=0)
    added_count = Column(Integer, nullable=False, default=0)
    removed_count = Column(Integer, nullable=False, default=0)
    changed_count = Column(Integer, nullable=False, default=0)
    unchanged_count = Column(Integer, nullable=False, default=0)
    artifact_hash = Column(String(64), nullable=False, index=True)
    artifact_json = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    __table_args__ = (
        UniqueConstraint("market", "snapshot_id", name="uq_universe_reconciliation_market_snapshot"),
        Index("idx_universe_reconciliation_market_created", "market", "created_at"),
    )
