"""Per-market telemetry alert model (bead asia.10.2).

Stateful alert log: an alert opens on threshold breach, may upgrade severity
or transition to acknowledged, and closes on recovery. The partial unique
index ``ux_telemetry_alerts_active`` (see migration 20260415_0013) enforces
at most one active alert per (market, metric_key).
"""

from __future__ import annotations

from sqlalchemy import BigInteger, Column, DateTime, Index, String, Text
from sqlalchemy.sql import func

from ..database import Base
from .types import JsonColumn


class AlertSeverity:
    WARNING = "warning"
    CRITICAL = "critical"


class AlertState:
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    CLOSED = "closed"


class MarketTelemetryAlert(Base):
    """One alert lifecycle row for a (market, metric_key) breach."""

    __tablename__ = "market_telemetry_alerts"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    market = Column(String(8), nullable=False)
    metric_key = Column(String(64), nullable=False)
    severity = Column(String(10), nullable=False)
    state = Column(String(12), nullable=False)
    owner = Column(String(64))
    title = Column(String(200), nullable=False)
    description = Column(Text)
    metrics = Column(JsonColumn)
    opened_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    acknowledged_at = Column(DateTime(timezone=True))
    acknowledged_by = Column(String(64))
    closed_at = Column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_telemetry_alerts_state_opened", "state", "opened_at"),
        Index("ix_telemetry_alerts_market_state", "market", "state", "opened_at"),
    )
