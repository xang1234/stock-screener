"""Persisted social evidence and installation-wide source registry."""
from sqlalchemy import (
    BigInteger, CheckConstraint, Column, Date, DateTime, ForeignKey, Index,
    Integer, JSON, Numeric, Text, UniqueConstraint, event, inspect, select,
)
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from app.database import Base


class SocialSourceRegistry(Base):
    __tablename__ = "social_source_registry"
    id = Column(Integer, primary_key=True)
    version = Column(Integer, nullable=False, server_default="1")
    mode = Column(Text, nullable=False, server_default="off")
    provider = Column(Text, nullable=False, server_default="disabled")
    official_budget_day = Column(Date)
    official_reserved_posts = Column(Integer, nullable=False, server_default="0")
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    __table_args__ = (
        CheckConstraint("id = 1", name="ck_social_registry_singleton"),
        CheckConstraint("version >= 1", name="ck_social_registry_version"),
        CheckConstraint("mode IN ('off','validation','live')", name="ck_social_registry_mode"),
        CheckConstraint("provider IN ('disabled','official','xui')", name="ck_social_registry_provider"),
        CheckConstraint("official_reserved_posts >= 0", name="ck_social_registry_official_reserved_posts"),
    )


class ContentPipelineEligibility(Base):
    __tablename__ = "content_pipeline_eligibility"
    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    content_item_id = Column(Integer, ForeignKey("content_items.id", ondelete="RESTRICT"), nullable=False)
    pipeline = Column(Text, nullable=False)
    channel = Column(Text, nullable=False)
    originating_source_id = Column(Integer, ForeignKey("content_sources.id", ondelete="RESTRICT"), nullable=True)
    observed_at = Column(DateTime(timezone=True), nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "content_item_id", "pipeline", "channel", "originating_source_id",
            name="uq_content_eligibility_source",
        ),
        CheckConstraint("channel IN ('legacy','social')", name="ck_content_eligibility_channel"),
        CheckConstraint("pipeline IN ('technical','fundamental')", name="ck_content_eligibility_pipeline"),
    )


class SocialSourceConfiguration(Base):
    __tablename__ = "social_source_configurations"
    content_source_id = Column(Integer, ForeignKey("content_sources.id", ondelete="RESTRICT"), primary_key=True)
    x_list_id = Column(Text, nullable=False, unique=True)
    lifecycle_state = Column(Text, nullable=False)
    provenance = Column(Text, nullable=False)
    tested_provider = Column(Text)
    test_status = Column(Text)
    test_sample_count = Column(Integer)
    tested_at = Column(DateTime(timezone=True))
    test_request_id = Column(Text, unique=True)
    test_request_version = Column(Integer)
    test_registry_version = Column(Integer)
    last_successful_collection_at = Column(DateTime(timezone=True))
    archived_at = Column(DateTime(timezone=True))
    version = Column(Integer, nullable=False, server_default="1")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    __table_args__ = (
        CheckConstraint("lifecycle_state IN ('pending','enabled','disabled','archived')", name="ck_social_source_lifecycle"),
        CheckConstraint("provenance IN ('system_seed','admin')", name="ck_social_source_provenance"),
        CheckConstraint("tested_provider IN ('official','xui')", name="ck_social_source_test_provider"),
        CheckConstraint("test_status IN ('queued','running','passed','failed','rate_limited','reauthentication_required','provider_error')", name="ck_social_source_test_status"),
        CheckConstraint("test_sample_count >= 0 AND test_sample_count <= 5", name="ck_social_source_sample_count"),
        CheckConstraint("version >= 1", name="ck_social_source_version"),
        CheckConstraint("(lifecycle_state = 'archived') = (archived_at IS NOT NULL)", name="ck_social_source_archive_time"),
    )


class SocialSourceAuditEvent(Base):
    __tablename__ = "social_source_audit_events"
    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    scope = Column(Text, nullable=False)
    registry_id = Column(Integer, ForeignKey("social_source_registry.id", ondelete="RESTRICT"), nullable=False)
    content_source_id = Column(Integer, ForeignKey("content_sources.id", ondelete="RESTRICT"))
    action = Column(Text, nullable=False)
    actor = Column(Text, nullable=False)
    before_json = Column(JSON)
    after_json = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    __table_args__ = (
        CheckConstraint("(scope = 'source' AND content_source_id IS NOT NULL) OR (scope = 'runtime' AND content_source_id IS NULL)", name="ck_social_audit_scope"),
        CheckConstraint("action IN ('created','renamed','test_requested','test_completed','enabled','disabled','archived','runtime_changed')", name="ck_social_audit_action"),
        Index("ix_social_audit_source_created", "content_source_id", "created_at"),
    )


class SocialPostSource(Base):
    __tablename__ = "social_post_sources"
    content_item_id = Column(Integer, ForeignKey("content_items.id", ondelete="RESTRICT"), primary_key=True)
    content_source_id = Column(Integer, ForeignKey("content_sources.id", ondelete="RESTRICT"), primary_key=True)
    observed_at = Column(DateTime(timezone=True), nullable=False)
    __table_args__ = (UniqueConstraint("content_item_id", "content_source_id", name="uq_social_post_source"),)


class SocialContentMetrics(Base):
    __tablename__ = "social_content_metrics"
    content_item_id = Column(Integer, ForeignKey("content_items.id", ondelete="RESTRICT"), primary_key=True)
    provider = Column(Text, nullable=False)
    provider_post_id = Column(Text, nullable=False)
    likes = Column(BigInteger)
    reposts = Column(BigInteger)
    replies = Column(BigInteger)
    quotes = Column(BigInteger)
    bookmarks = Column(BigInteger)
    views = Column(BigInteger)
    observed_at = Column(DateTime(timezone=True), nullable=False)
    __table_args__ = (
        UniqueConstraint("content_item_id", name="uq_social_metrics_content"),
        CheckConstraint("provider IN ('official','xui')", name="ck_social_metrics_provider"),
        *(CheckConstraint(f"{field} >= 0", name=f"ck_social_metrics_{field}") for field in ("likes", "reposts", "replies", "quotes", "bookmarks", "views")),
    )


class SocialPostTicker(Base):
    __tablename__ = "social_post_tickers"
    id = Column(Integer, primary_key=True)
    content_item_id = Column(Integer, ForeignKey("content_items.id", ondelete="RESTRICT"), nullable=False)
    candidate_key = Column(Text, nullable=False)
    raw_token = Column(Text, nullable=False)
    stock_universe_id = Column(Integer, ForeignKey("stock_universe.id", ondelete="RESTRICT"))
    canonical_symbol = Column(Text)
    market = Column(Text)
    mic = Column(Text)
    local_code = Column(Text)
    resolution_state = Column(Text, nullable=False)
    resolution_policy_version = Column(Text, nullable=False)
    explanation_json = Column(JSON, nullable=False)
    __table_args__ = (
        UniqueConstraint("content_item_id", "candidate_key", name="uq_social_post_candidate"),
        CheckConstraint("market IN ('US','HK','CN','JP','TW')", name="ck_social_ticker_market"),
        CheckConstraint("resolution_state IN ('resolved','unresolved')", name="ck_social_ticker_resolution"),
        CheckConstraint("(resolution_state = 'resolved' AND stock_universe_id IS NOT NULL AND canonical_symbol IS NOT NULL AND market IS NOT NULL) OR (resolution_state = 'unresolved' AND stock_universe_id IS NULL AND canonical_symbol IS NULL AND market IS NULL)", name="ck_social_ticker_identity"),
    )


class SocialSignalRun(Base):
    __tablename__ = "social_signal_runs"
    id = Column(Text, primary_key=True)
    registry_id = Column(Integer, ForeignKey("social_source_registry.id", ondelete="RESTRICT"), nullable=False)
    registry_version = Column(Integer, nullable=False)
    mode = Column(Text, nullable=False)
    provider = Column(Text, nullable=False)
    status = Column(Text, nullable=False)
    source_outcomes_json = Column(JSON, nullable=False)
    application_progress_json = Column(JSON, nullable=False)
    feature_run_ids_json = Column(JSON, nullable=False)
    exposure_dates_json = Column(JSON, nullable=False)
    coverage_json = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    published_at = Column(DateTime(timezone=True))
    __table_args__ = (
        CheckConstraint("mode IN ('validation','live')", name="ck_social_run_mode"),
        CheckConstraint("provider IN ('official','xui')", name="ck_social_run_provider"),
        CheckConstraint("status IN ('running','staged','completed','failed','published')", name="ck_social_run_status"),
    )


class SocialSignalSnapshot(Base):
    __tablename__ = "social_signal_snapshots"
    id = Column(Integer, primary_key=True)
    run_id = Column(Text, ForeignKey("social_signal_runs.id", ondelete="RESTRICT"), nullable=False)
    window_days = Column(Integer, nullable=False)
    candidate_key = Column(Text, nullable=False)
    stock_universe_id = Column(Integer, ForeignKey("stock_universe.id", ondelete="RESTRICT"))
    canonical_symbol = Column(Text)
    market = Column(Text)
    mic = Column(Text)
    local_code = Column(Text)
    state = Column(Text, nullable=False)
    social_score = Column(Numeric(12, 6))
    confirmation_score = Column(Numeric(12, 6))
    queue_score = Column(Numeric(12, 6))
    explanation_json = Column(JSON, nullable=False)
    coverage_json = Column(JSON, nullable=False)
    resolution_policy_version = Column(Text, nullable=False)
    formula_version = Column(Text, nullable=False)
    latest_mention = Column(DateTime(timezone=True))
    mention_count = Column(Integer, nullable=False)
    observed_list_count = Column(Integer, nullable=False)
    enabled_list_count = Column(Integer, nullable=False)
    normalization_scope = Column(Text, nullable=False)
    __table_args__ = (
        UniqueConstraint("run_id", "window_days", "candidate_key", name="uq_social_snapshot_candidate"),
        CheckConstraint("window_days IN (1,7,14)", name="ck_social_snapshot_window"),
        CheckConstraint("market IN ('US','HK','CN','JP','TW')", name="ck_social_snapshot_market"),
        CheckConstraint("state IN ('actionable','watch','risk_off','context','unresolved')", name="ck_social_snapshot_state"),
        CheckConstraint("(social_score IS NOT NULL AND confirmation_score IS NOT NULL AND queue_score IS NOT NULL) OR ((social_score IS NULL OR confirmation_score IS NULL) AND queue_score IS NULL)", name="ck_social_snapshot_nullable_scores"),
        *(CheckConstraint(f"{field} >= 0 AND {field} <= 100", name=f"ck_social_snapshot_{field}") for field in ("social_score", "confirmation_score", "queue_score")),
    )


class SocialSignalRunPointer(Base):
    __tablename__ = "social_signal_run_pointers"
    key = Column(Text, primary_key=True)
    run_id = Column(Text, ForeignKey("social_signal_runs.id", ondelete="RESTRICT"), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    __table_args__ = (CheckConstraint("key = 'latest_published'", name="ck_social_pointer_key"),)


@event.listens_for(Session, "before_flush")
def _protect_social_history(session, flush_context, instances):
    from app.infra.db.models.social_analysis import SocialRunWork
    links = [row for row in session.new | session.dirty | session.deleted if isinstance(row, SocialRunWork)]
    if links:
        from app.services.social_theme_projection_service import _lock_registry
        with session.no_autoflush:
            _lock_registry(session)
            run_ids = {row.run_id for row in links}
            for row in links:
                run_ids.update(inspect(row).attrs.run_id.history.deleted)
            statuses = session.execute(select(SocialSignalRun.status).where(SocialSignalRun.id.in_(run_ids))).scalars()
            if any(status != "running" for status in statuses):
                raise ValueError("social_terminal_work_manifest_immutable")
    for row in session.new:
        if isinstance(row, SocialSignalSnapshot):
            run = session.get(SocialSignalRun, row.run_id)
            if run is not None:
                prior = inspect(run).attrs.status.history.deleted
                status = prior[0] if prior else run.status
                if status in {"staged", "completed", "failed", "published"}:
                    raise ValueError("social_run_snapshots_immutable")
    for row in session.dirty | session.deleted:
        if isinstance(row, SocialSourceAuditEvent):
            raise ValueError("social_audit_append_only")
        if isinstance(row, SocialSourceConfiguration) and inspect(row).attrs.x_list_id.history.has_changes():
            raise ValueError("social_list_id_immutable")
        if isinstance(row, SocialSignalSnapshot):
            raise ValueError("social_snapshot_immutable")
        if isinstance(row, SocialSignalRun):
            state = inspect(row)
            old_status = state.attrs.status.history.deleted
            terminal = old_status[0] if old_status else row.status
            if terminal in {"staged", "completed", "failed", "published"}:
                changed = {attr.key for attr in state.attrs if attr.history.has_changes()}
                publication_transition = (terminal == "staged" and row.status == "published"
                    and changed <= {"status", "published_at"} and row not in session.deleted)
                if not publication_transition and (changed or row in session.deleted):
                    raise ValueError("social_run_immutable")


@event.listens_for(Session, "do_orm_execute")
def _protect_social_bulk_history(state):
    from app.infra.db.models.social_analysis import SocialRunWork
    if (state.is_insert or state.is_update or state.is_delete) and state.bind_mapper is not None and state.bind_mapper.class_ is SocialRunWork:
        raise ValueError("social_work_manifest_immutable_bulk")
    if state.is_insert and state.bind_mapper is not None and state.bind_mapper.class_ is SocialSignalSnapshot:
        raise ValueError("social_publication_immutable_bulk")
    if (state.is_update or state.is_delete) and state.bind_mapper is not None:
        if state.bind_mapper.class_ is SocialSourceAuditEvent:
            raise ValueError("social_audit_append_only")
        if state.bind_mapper.class_ in {SocialSignalSnapshot, SocialSignalRun}:
            raise ValueError("social_publication_immutable_bulk")
