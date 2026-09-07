"""Persisted social evidence and installation-wide source registry."""
from sqlalchemy import (
    BigInteger, CheckConstraint, Column, Date, DateTime, ForeignKey, Index,
    Integer, JSON, Numeric, Text, UniqueConstraint, event, inspect,
)
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from sqlalchemy.orm import declarative_base
from alembic import op
import json

Base = declarative_base()
revision = "20260906_0035"
down_revision = "20260904_0034"
branch_labels = None
depends_on = None


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
    content_item_id = Column(Integer, ForeignKey("content_items.id", ondelete="RESTRICT"), primary_key=True)
    pipeline = Column(Text, primary_key=True)
    channel = Column(Text, primary_key=True)
    originating_source_id = Column(Integer, ForeignKey("content_sources.id", ondelete="RESTRICT"), nullable=True)
    observed_at = Column(DateTime(timezone=True), nullable=False)
    __table_args__ = (
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



# External tables are reflected only to resolve foreign keys; never created here.
_TABLE_NAMES = tuple(Base.metadata.tables)


def upgrade():
    bind = op.get_bind()
    from sqlalchemy import MetaData, Table, select
    for name in ("content_sources", "content_items", "stock_universe"):
        Table(name, Base.metadata, autoload_with=bind, extend_existing=True)
    for name in _TABLE_NAMES:
        Base.metadata.tables[name].create(bind)
    bind.execute(SocialSourceRegistry.__table__.insert().values(id=1))
    items = Base.metadata.tables["content_items"]
    sources = Base.metadata.tables["content_sources"]
    rows = bind.execute(select(items.c.id, sources.c.id.label("originating_source_id"), items.c.fetched_at, sources.c.pipelines).select_from(items.outerjoin(sources, items.c.source_id == sources.c.id)))
    for row in rows:
        pipelines = row.pipelines
        if isinstance(pipelines, str):
            try:
                pipelines = json.loads(pipelines)
            except ValueError:
                pipelines = pipelines.split(",")
        if not isinstance(pipelines, (list, tuple)):
            pipelines = ["technical", "fundamental"]
        pipelines = {str(value).strip().lower() for value in pipelines} & {"technical", "fundamental"}
        for pipeline in pipelines or {"technical", "fundamental"}:
            bind.execute(ContentPipelineEligibility.__table__.insert().values(
                content_item_id=row.id, pipeline=pipeline, channel="legacy",
                originating_source_id=row.originating_source_id,
                observed_at=row.fetched_at or bind.execute(select(func.now())).scalar_one(),
            ))


def downgrade():
    bind = op.get_bind()
    for name in reversed(_TABLE_NAMES):
        Base.metadata.tables[name].drop(bind)
