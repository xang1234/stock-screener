"""Database models for provider-backed fundamentals snapshots."""

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.sql import func

from ..database import Base


class ProviderSnapshotRun(Base):
    """Metadata for a snapshot build or publish run."""

    __tablename__ = "provider_snapshot_runs"

    id = Column(Integer, primary_key=True, index=True)
    snapshot_key = Column(String(64), nullable=False, index=True)
    run_mode = Column(String(16), nullable=False)  # preview | publish
    status = Column(String(32), nullable=False, index=True)
    source_revision = Column(String(128), nullable=False)
    coverage_stats_json = Column(Text)
    parity_stats_json = Column(Text)
    warnings_json = Column(Text)
    symbols_total = Column(Integer, nullable=False, default=0)
    symbols_published = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    published_at = Column(DateTime(timezone=True))

    __table_args__ = (
        Index("idx_provider_snapshot_runs_key_created", "snapshot_key", "created_at"),
        Index("idx_provider_snapshot_runs_key_status", "snapshot_key", "status"),
        UniqueConstraint("snapshot_key", "source_revision", name="uq_provider_snapshot_revision"),
    )


class ProviderSnapshotRow(Base):
    """One normalized provider snapshot row for a symbol within a run."""

    __tablename__ = "provider_snapshot_rows"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("provider_snapshot_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    exchange = Column(String(20), index=True)
    row_hash = Column(String(64), nullable=False)
    normalized_payload_json = Column(Text, nullable=False)
    raw_payload_json = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("run_id", "symbol", name="uq_provider_snapshot_row_run_symbol"),
        Index("idx_provider_snapshot_rows_run_exchange", "run_id", "exchange"),
    )


class ProviderSnapshotPointer(Base):
    """Atomic pointer to the currently published snapshot run."""

    __tablename__ = "provider_snapshot_pointers"

    snapshot_key = Column(String(64), primary_key=True)
    run_id = Column(Integer, ForeignKey("provider_snapshot_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
