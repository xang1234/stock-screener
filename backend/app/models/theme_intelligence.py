"""Audited, non-destructive theme grouping and source-bound event history."""

from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    event,
)
from sqlalchemy.orm import Session, relationship
from sqlalchemy.sql import func

from app.database import Base


class ThemeEquivalenceOperation(Base):
    __tablename__ = "theme_equivalence_operations"
    id = Column(Integer, primary_key=True)
    operation_key = Column(String(120), nullable=False, unique=True)
    source_id = Column(
        Integer, ForeignKey("theme_clusters.id", ondelete="RESTRICT"), nullable=False
    )
    requested_target_id = Column(Integer, nullable=False)
    target_id = Column(
        Integer, ForeignKey("theme_clusters.id", ondelete="RESTRICT"), nullable=False
    )
    pipeline = Column(String(20), nullable=False, index=True)
    member_ids = Column(JSON, nullable=False)
    aliases = Column(JSON, nullable=False)
    actor = Column(String(120), nullable=False)
    reason = Column(Text, nullable=False)
    active = Column(Boolean, nullable=False, default=True)
    refresh_pending = Column(Boolean, nullable=False, default=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    undone_at = Column(DateTime(timezone=True))
    undone_by = Column(String(120))
    undo_reason = Column(Text)


class ThemeDevelopmentEvent(Base):
    __tablename__ = "theme_development_events"
    id = Column(Integer, primary_key=True)
    pipeline = Column(String(20), nullable=False, index=True)
    event_key = Column(String(64), nullable=False)
    canonical_event_key = Column(String(64), unique=True)
    development_identity = Column(Uuid(as_uuid=True), unique=True)
    identity = Column(JSON, nullable=False)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    __table_args__ = (
        UniqueConstraint("pipeline", "event_key", name="uq_theme_event_key"),
    )


class ThemeDevelopmentObservation(Base):
    __tablename__ = "theme_development_observations"
    id = Column(Integer, primary_key=True)
    event_id = Column(
        Integer,
        ForeignKey("theme_development_events.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    content_item_id = Column(
        Integer,
        ForeignKey("content_items.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    pipeline = Column(String(20), nullable=False)
    analysis_channel = Column(String(20), nullable=False)
    development_support = Column(String(20), nullable=False, default="present")
    source_family_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("economic_source_families.id", ondelete="RESTRICT"),
    )
    revision = Column(String(64), nullable=False)
    observation_key = Column(String(64), nullable=False, unique=True)
    theme_links = relationship(
        "ThemeDevelopmentTheme", cascade="all, delete-orphan", lazy="selectin"
    )
    economic_theme_links = relationship(
        "EconomicThemeDevelopment", cascade="all, delete-orphan", lazy="selectin"
    )

    @property
    def theme_ids(self):
        return sorted(link.theme_id for link in self.theme_links)

    facts = Column(JSON, nullable=False)
    citations = Column(JSON, nullable=False)
    classification = Column(String(30), nullable=False)
    published_at = Column(DateTime(timezone=True))
    available_at = Column(DateTime(timezone=True), nullable=False)
    recorded_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    superseded = Column(Boolean, nullable=False, default=False)

    __table_args__ = (
        CheckConstraint(
            "analysis_channel IN ('technical','fundamental','narrative')",
            name="ck_theme_development_analysis_channel",
        ),
        CheckConstraint(
            "development_support IN ('present','absent','unresolved')",
            name="ck_theme_development_support",
        ),
    )


class ThemeDevelopmentWork(Base):
    __tablename__ = "theme_development_work"
    id = Column(Integer, primary_key=True)
    content_item_id = Column(
        Integer, ForeignKey("content_items.id", ondelete="RESTRICT"), nullable=False
    )
    pipeline = Column(String(20), nullable=False)
    revision = Column(String(64), nullable=False)
    checked_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    status = Column(String(20), nullable=False, default="pending", index=True)
    attempts = Column(Integer, nullable=False, default=0)
    error_code = Column(String(80))
    next_attempt_at = Column(DateTime(timezone=True))
    claim_token = Column(String(64))
    lease_until = Column(DateTime(timezone=True))
    __table_args__ = (
        UniqueConstraint(
            "content_item_id", "pipeline", "revision", name="uq_theme_development_work"
        ),
    )


class ThemeDevelopmentTheme(Base):
    __tablename__ = "theme_development_themes"
    observation_id = Column(
        Integer,
        ForeignKey("theme_development_observations.id", ondelete="CASCADE"),
        primary_key=True,
    )
    theme_id = Column(
        Integer,
        ForeignKey("theme_clusters.id", ondelete="RESTRICT"),
        primary_key=True,
        index=True,
    )


class EconomicThemeDevelopment(Base):
    __tablename__ = "economic_theme_developments"

    observation_id = Column(
        Integer,
        ForeignKey("theme_development_observations.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    economic_theme_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("economic_themes.id", ondelete="RESTRICT"),
        primary_key=True,
        index=True,
    )
    link_origin = Column(String(32), nullable=False, default="economic_native")
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "link_origin IN ('economic_native','legacy_mapping','compatibility')",
            name="ck_economic_theme_development_origin",
        ),
    )


class EconomicDevelopmentBackfill(Base):
    """Which taxonomy version the ``legacy_mapping`` development links are for (#513).

    One row: a backfill for another version replaces the links and this
    marker. Observations above ``through_observation_id`` are not covered.
    """

    __tablename__ = "economic_development_backfills"

    taxonomy_version_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("economic_taxonomy_versions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    through_observation_id = Column(Integer, nullable=False)
    legacy_link_count = Column(Integer, nullable=False)
    legacy_link_max_observation_id = Column(Integer)
    completed_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class LegacyDevelopmentEventMapping(Base):
    __tablename__ = "legacy_development_event_mappings"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    old_event_id = Column(
        Integer,
        ForeignKey("theme_development_events.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    )
    canonical_event_id = Column(
        Integer,
        ForeignKey("theme_development_events.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    old_pipeline = Column(String(20), nullable=False)
    migration_run_id = Column(Uuid(as_uuid=True), nullable=False, index=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


@event.listens_for(Session, "before_flush")
def _protect_legacy_development_event_mappings(session, _flush_context, _instances):
    for row in session.dirty:
        if isinstance(row, LegacyDevelopmentEventMapping) and session.is_modified(
            row, include_collections=False
        ):
            raise ValueError("legacy_development_event_mapping_append_only")
    for row in session.deleted:
        if isinstance(row, LegacyDevelopmentEventMapping):
            raise ValueError(  # noqa: TRY004 -- Domain immutability violation.
                "legacy_development_event_mapping_append_only"
            )
