"""Per-market telemetry event model (bead asia.10.1).

Append-only event log for the 5 telemetry categories: freshness lag,
universe drift, benchmark age, extraction success by language, and
fundamentals completeness distribution. Each row carries a versioned JSON
payload so metric definitions can evolve without breaking historical reads.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, Column, DateTime, Index, SmallInteger, String
from sqlalchemy.sql import func

from ..database import Base
from .types import JsonColumn


class MarketTelemetryEvent(Base):
    """One emission of a per-market telemetry metric."""

    __tablename__ = "market_telemetry_events"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    market = Column(String(8), nullable=False)             # US/HK/JP/TW/SHARED
    metric_key = Column(String(64), nullable=False)        # freshness_lag/universe_drift/etc.
    schema_version = Column(SmallInteger, nullable=False)
    payload = Column(JsonColumn, nullable=False)
    recorded_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("ix_market_telemetry_market_recorded", "market", "recorded_at"),
        Index("ix_market_telemetry_metric_recorded", "metric_key", "recorded_at"),
        # Supports the 15d retention DELETE in PerMarketTelemetry.cleanup_old_events.
        Index("ix_market_telemetry_recorded_at", "recorded_at"),
    )
