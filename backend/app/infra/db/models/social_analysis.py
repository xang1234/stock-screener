"""Durable semantic work and the separate installation-wide dollar ledger."""
from sqlalchemy import (Boolean, CheckConstraint, Column, Date, DateTime, ForeignKey,
                        Integer, JSON, Numeric, Text, UniqueConstraint)
from sqlalchemy.sql import func
from sqlalchemy import event
from sqlalchemy.orm import Session
from app.database import Base


class SocialThemeAssociation(Base):
    __tablename__ = "social_theme_associations"
    id = Column(Integer, primary_key=True)
    theme_cluster_id = Column(Integer, ForeignKey("theme_clusters.id", ondelete="RESTRICT"), nullable=False)
    company_key = Column(Text)
    market = Column(Text, nullable=False)
    canonical_symbol = Column(Text, nullable=False)
    state = Column(Text, nullable=False, default="proposed")
    origin = Column(Text, nullable=False, default="social")
    decision_owner = Column(Text, nullable=False, default="system")
    evidence_work_ids = Column(JSON, nullable=False, default=list)
    policy_version = Column(Text, nullable=False)
    version = Column(Integer, nullable=False, default=1)
    first_seen_at = Column(DateTime(timezone=True), nullable=False)
    accepted_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), nullable=False)
    __table_args__ = (
        UniqueConstraint("theme_cluster_id", "market", "canonical_symbol", name="uq_social_theme_listing"),
        CheckConstraint("state IN ('proposed','accepted','rejected')", name="ck_social_theme_state"),
        CheckConstraint("origin IN ('social','legacy')", name="ck_social_theme_origin"),
        CheckConstraint("decision_owner IN ('system','admin')", name="ck_social_theme_owner"),
        CheckConstraint("market IN ('US','HK','CN','JP','TW')", name="ck_social_theme_market"),
        CheckConstraint("version >= 1", name="ck_social_theme_version"),
    )


class SocialThemeDecision(Base):
    __tablename__ = "social_theme_decisions"
    id = Column(Integer, primary_key=True)
    association_id = Column(Integer, ForeignKey("social_theme_associations.id", ondelete="RESTRICT"), nullable=False)
    run_id = Column(Text, ForeignKey("social_signal_runs.id", ondelete="RESTRICT"))
    actor = Column(Text, nullable=False)
    reason = Column(Text, nullable=False)
    before_state = Column(Text, nullable=False)
    after_state = Column(Text, nullable=False)
    policy_version = Column(Text, nullable=False)
    evidence_work_ids = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)


@event.listens_for(Session, "before_flush")
def _protect_theme_decision_history(session, flush_context, instances):
    if any(isinstance(row, SocialThemeDecision) for row in session.dirty | session.deleted):
        raise ValueError("social_theme_decision_append_only")


@event.listens_for(Session, "do_orm_execute")
def _protect_theme_decision_bulk_history(state):
    if (state.is_update or state.is_delete) and state.bind_mapper is not None and state.bind_mapper.class_ is SocialThemeDecision:
        raise ValueError("social_theme_decision_append_only")


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
