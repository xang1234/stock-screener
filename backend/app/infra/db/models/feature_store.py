"""
Feature Store SQLAlchemy models.

Stores pre-computed daily screening snapshots decoupled from individual scans.
Uses a pointer-based atomic publishing pattern: compute a new FeatureRun,
populate StockFeatureDaily rows, then atomically swap FeatureRunPointer
to make the new data visible to readers.

Tables:
    feature_runs              — One row per screening computation
    feature_run_universe_symbols — Symbols included in each run
    stock_feature_daily       — Per-symbol screening scores
    feature_run_pointers      — Atomic publish pointers (e.g. 'latest_published')
"""
from sqlalchemy import (
    Column, Integer, Float, Text, Date, DateTime, JSON,
    ForeignKey, Index,
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.database import Base


class FeatureRun(Base):
    """A single screening computation run."""

    __tablename__ = "feature_runs"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    as_of_date = Column(Date, nullable=False, index=True)
    run_type = Column(Text, nullable=False)  # daily_snapshot / backfill / manual
    status = Column(Text, nullable=False, index=True)  # running / completed / failed / quarantined / published
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    code_version = Column(Text, nullable=True)
    universe_hash = Column(Text, nullable=True)
    input_hash = Column(Text, nullable=True)
    config_json = Column(JSON, nullable=True)
    correlation_id = Column(Text, nullable=True, index=True)
    stats_json = Column(JSON, nullable=True)
    warnings_json = Column(JSON, nullable=True)

    # Relationships
    universe_symbols = relationship(
        "FeatureRunUniverseSymbol",
        back_populates="run",
        cascade="all, delete-orphan",
    )
    features = relationship(
        "StockFeatureDaily",
        back_populates="run",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_feature_runs_date_status", "as_of_date", "status"),
    )


class FeatureRunUniverseSymbol(Base):
    """Symbol included in a feature run's universe."""

    __tablename__ = "feature_run_universe_symbols"

    run_id = Column(
        Integer,
        ForeignKey("feature_runs.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )
    symbol = Column(Text, nullable=False, primary_key=True, index=True)

    # Relationships
    run = relationship("FeatureRun", back_populates="universe_symbols")


class StockFeatureDaily(Base):
    """Pre-computed screening scores for a single stock on a single day."""

    __tablename__ = "stock_feature_daily"

    run_id = Column(
        Integer,
        ForeignKey("feature_runs.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )
    symbol = Column(Text, nullable=False, primary_key=True, index=True)
    as_of_date = Column(Date, nullable=False, index=True)  # denormalized from FeatureRun
    composite_score = Column(Float, nullable=True, index=True)
    overall_rating = Column(Integer, nullable=True, index=True)
    passes_count = Column(Integer, nullable=True)
    details_json = Column(JSON, nullable=True)

    # Relationships
    run = relationship("FeatureRun", back_populates="features")

    __table_args__ = (
        Index("ix_stock_feature_daily_run_score", "run_id", "composite_score"),
        Index("ix_stock_feature_daily_run_rating", "run_id", "overall_rating"),
    )


class FeatureRunPointer(Base):
    """Named pointer to a feature run for atomic publishing."""

    __tablename__ = "feature_run_pointers"

    key = Column(Text, primary_key=True)  # e.g. 'latest_published'
    run_id = Column(
        Integer,
        ForeignKey("feature_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
