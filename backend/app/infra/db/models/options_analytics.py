"""Relational persistence models for published options analytics."""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class OptionsAnalyticsRun(Base):
    __tablename__ = "options_analytics_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market = Column(String(8), nullable=False, index=True)
    origin = Column(String(32), nullable=False)
    source_feature_run_id = Column(
        Integer, ForeignKey("feature_runs.id", ondelete="RESTRICT"), nullable=True
    )
    external_source_feature_run_key = Column(String(255), nullable=True)
    calculation_version = Column(String(64), nullable=False)
    schema_version = Column(String(64), nullable=False)
    provider = Column(String(32), nullable=False)
    input_signature = Column(String(64), nullable=False)
    attempt_number = Column(Integer, nullable=False)
    status = Column(String(32), nullable=False, index=True)
    as_of_date = Column(Date, nullable=False, index=True)
    risk_free_rate = Column(Float, nullable=True)
    expected_count = Column(Integer, nullable=False)
    current_count = Column(Integer, nullable=False)
    continuity_count = Column(Integer, nullable=False)
    completed_count = Column(Integer, nullable=False)
    core_valid_current_count = Column(Integer, nullable=False)
    failed_count = Column(Integer, nullable=False)
    retried_count = Column(Integer, nullable=False)
    coverage = Column(Float, nullable=False)
    assumptions_json = Column(JSON, nullable=True)
    warnings_json = Column(JSON, nullable=True)
    diagnostics_json = Column(JSON, nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)

    items = relationship(
        "OptionsAnalyticsRunItem",
        back_populates="run",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint(
            "input_signature",
            "attempt_number",
            name="uq_options_run_signature_attempt",
        ),
        CheckConstraint(
            "(origin = 'history_transfer' AND external_source_feature_run_key IS NOT NULL) "
            "OR (origin <> 'history_transfer' AND source_feature_run_id IS NOT NULL)",
            name="ck_options_run_source_identity",
        ),
        Index(
            "ix_options_runs_market_version_status",
            "market",
            "calculation_version",
            "status",
        ),
    )


class OptionsAnalyticsRunItem(Base):
    __tablename__ = "options_analytics_run_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(
        Integer,
        ForeignKey("options_analytics_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    security_symbol = Column(Text, nullable=False, index=True)
    candidate_kind = Column(String(16), nullable=False)
    candidate_rank = Column(Integer, nullable=True)
    leader_rank = Column(Integer, nullable=True)
    spot_price = Column(Float, nullable=True)
    expiration = Column(Date, nullable=True)
    observation_state = Column(String(32), nullable=False)
    core_valid = Column(Boolean, nullable=False, default=False)
    observation_at = Column(DateTime(timezone=True), nullable=True)
    max_pain = Column(Float, nullable=True)
    net_gex = Column(Float, nullable=True)
    gamma_flip = Column(Float, nullable=True)
    call_wall = Column(Float, nullable=True)
    put_wall = Column(Float, nullable=True)
    atm_iv = Column(Float, nullable=True)
    skew_25_delta = Column(Float, nullable=True)
    realized_volatility = Column(Float, nullable=True)
    vrp = Column(Float, nullable=True)
    activity_intensity = Column(Float, nullable=True)
    iv_percentile = Column(Float, nullable=True)
    iv_rank = Column(Float, nullable=True)
    max_pain_change_5 = Column(Float, nullable=True)
    net_gex_change_5 = Column(Float, nullable=True)
    gamma_flip_change_5 = Column(Float, nullable=True)
    atm_iv_change_5 = Column(Float, nullable=True)
    skew_25_delta_change_5 = Column(Float, nullable=True)
    realized_volatility_change_5 = Column(Float, nullable=True)
    vrp_change_5 = Column(Float, nullable=True)
    activity_intensity_change_5 = Column(Float, nullable=True)
    activity_rank = Column(Integer, nullable=True)
    call_open_interest = Column(BigInteger, nullable=True)
    put_open_interest = Column(BigInteger, nullable=True)
    call_volume = Column(BigInteger, nullable=True)
    put_volume = Column(BigInteger, nullable=True)
    call_put_volume_ratio = Column(Float, nullable=True)
    volume_oi_ratio = Column(Float, nullable=True)
    near_spot_volume_concentration = Column(Float, nullable=True)
    near_spot_open_interest_concentration = Column(Float, nullable=True)
    highest_contract_activity_ratio = Column(Float, nullable=True)
    short_history_observation_count = Column(Integer, nullable=False, default=0)
    iv_history_observation_count = Column(Integer, nullable=False, default=0)
    lifetime_observation_count = Column(Integer, nullable=False, default=0)
    retry_count = Column(Integer, nullable=False, default=0)
    evidence_json = Column(JSON, nullable=True)
    assumptions_json = Column(JSON, nullable=True)
    warnings_json = Column(JSON, nullable=True)
    reasons_json = Column(JSON, nullable=True)

    run = relationship("OptionsAnalyticsRun", back_populates="items")
    strike_points = relationship(
        "OptionsAnalyticsStrikePoint",
        back_populates="item",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint(
            "run_id", "security_symbol", name="uq_options_run_item_symbol"
        ),
        Index(
            "ix_options_items_run_kind_activity",
            "run_id",
            "candidate_kind",
            "activity_rank",
        ),
    )


class OptionsAnalyticsStrikePoint(Base):
    __tablename__ = "options_analytics_strike_points"

    id = Column(Integer, primary_key=True, autoincrement=True)
    item_id = Column(
        Integer,
        ForeignKey("options_analytics_run_items.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    strike = Column(Float, nullable=False)
    call_open_interest = Column(Integer, nullable=True)
    put_open_interest = Column(Integer, nullable=True)
    call_volume = Column(Integer, nullable=True)
    put_volume = Column(Integer, nullable=True)
    call_iv = Column(Float, nullable=True)
    put_iv = Column(Float, nullable=True)
    estimated_call_gex = Column(Float, nullable=True)
    estimated_put_gex = Column(Float, nullable=True)

    item = relationship("OptionsAnalyticsRunItem", back_populates="strike_points")

    __table_args__ = (
        UniqueConstraint("item_id", "strike", name="uq_options_strike_item_strike"),
    )


class OptionsAnalyticsPointer(Base):
    __tablename__ = "options_analytics_pointers"

    market = Column(String(8), primary_key=True)
    calculation_version = Column(String(64), primary_key=True)
    run_id = Column(
        Integer,
        ForeignKey("options_analytics_runs.id", ondelete="RESTRICT"),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
