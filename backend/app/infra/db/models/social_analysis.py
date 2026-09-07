"""Durable semantic work and the separate installation-wide dollar ledger."""
from sqlalchemy import (Boolean, CheckConstraint, Column, Date, DateTime, ForeignKey,
                        Integer, JSON, Numeric, Text, UniqueConstraint)
from sqlalchemy.sql import func
from app.database import Base


class SocialExtractionWork(Base):
    __tablename__ = "social_extraction_work"
    id = Column(Integer, primary_key=True)
    content_item_id = Column(Integer, ForeignKey("content_items.id", ondelete="RESTRICT"), nullable=False)
    input_hash = Column(Text, nullable=False)
    prompt_version = Column(Text, nullable=False)
    schema_version = Column(Text, nullable=False)
    selected_model = Column(Text, nullable=False)
    input_snapshot_json = Column(JSON, nullable=False)
    actual_provider = Column(Text)
    actual_model = Column(Text)
    result_json = Column(JSON)
    error_code = Column(Text)
    state = Column(Text, nullable=False, default="pending")
    claim_token = Column(Text)
    claim_expires_at = Column(DateTime(timezone=True))
    requested_by_admin = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    __table_args__ = (
        UniqueConstraint("content_item_id", "input_hash", "prompt_version", "schema_version", "selected_model", name="uq_social_extraction_identity"),
        CheckConstraint("state IN ('pending','running','waiting_budget','succeeded','failed_retryable','failed_terminal','outside_window')", name="ck_social_work_state"),
    )


class SocialRunWork(Base):
    __tablename__ = "social_run_work"
    run_id = Column(Text, ForeignKey("social_signal_runs.id", ondelete="RESTRICT"), primary_key=True)
    work_id = Column(Integer, ForeignKey("social_extraction_work.id", ondelete="RESTRICT"), primary_key=True)
    input_hash = Column(Text, nullable=False)
    included_at = Column(DateTime(timezone=True), nullable=False)
    __table_args__ = (UniqueConstraint("run_id", "work_id", name="uq_social_run_work"),)


class SocialLLMBudgetDay(Base):
    __tablename__ = "social_llm_budget_days"
    id = Column(Integer, primary_key=True)
    budget_date = Column(Date, nullable=False)
    timezone = Column(Text, nullable=False)
    period_start_utc = Column(DateTime(timezone=True), nullable=False)
    period_end_utc = Column(DateTime(timezone=True), nullable=False)
    limit_usd = Column(Numeric(24, 12), nullable=False)
    reserved_usd = Column(Numeric(24, 12), nullable=False, default=0)
    actual_usd = Column(Numeric(24, 12), nullable=False, default=0)
    version = Column(Integer, nullable=False, default=1)
    __table_args__ = (
        UniqueConstraint("period_start_utc", "period_end_utc", name="uq_social_llm_period"),
        CheckConstraint("limit_usd >= 0 AND reserved_usd >= 0 AND actual_usd >= 0", name="ck_social_budget_money"),
    )


class SocialLLMAttempt(Base):
    __tablename__ = "social_llm_attempts"
    id = Column(Integer, primary_key=True)
    idempotency_key = Column(Text, nullable=False, unique=True)
    budget_day_id = Column(Integer, ForeignKey("social_llm_budget_days.id", ondelete="RESTRICT"), nullable=False)
    work_ids = Column(JSON, nullable=False)
    estimated_usd = Column(Numeric(24, 12), nullable=False)
    actual_usd = Column(Numeric(24, 12))
    pricing_version = Column(Text, nullable=False)
    input_token_limit = Column(Integer, nullable=False)
    output_token_limit = Column(Integer, nullable=False)
    actual_input_tokens = Column(Integer)
    actual_output_tokens = Column(Integer)
    state = Column(Text, nullable=False)
    provider_request_id = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True))
    __table_args__ = (
        CheckConstraint("state IN ('reserved','dispatched','reconciled','uncertain','released')", name="ck_social_attempt_state"),
        CheckConstraint("estimated_usd >= 0 AND (actual_usd IS NULL OR actual_usd >= 0)", name="ck_social_attempt_money"),
    )
