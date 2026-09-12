"""Audited, non-destructive theme grouping and source-bound event history."""

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
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
    revision = Column(String(64), nullable=False)
    observation_key = Column(String(64), nullable=False, unique=True)
    theme_links = relationship(
        "ThemeDevelopmentTheme", cascade="all, delete-orphan", lazy="selectin"
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
