"""Published UI bootstrap snapshot models."""

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..database import Base


class UIViewSnapshot(Base):
    """Persisted snapshot payload for a single page/bootstrap variant."""

    __tablename__ = "ui_view_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    view_key = Column(String(64), nullable=False, index=True)
    variant_key = Column(String(128), nullable=False, index=True)
    source_revision = Column(String(256), nullable=False, index=True)
    payload_json = Column(JSON, nullable=False)
    published_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "view_key",
            "variant_key",
            "source_revision",
            name="uq_ui_view_snapshots_revision",
        ),
    )


class UIViewSnapshotPointer(Base):
    """Pointer to the currently published snapshot for a variant."""

    __tablename__ = "ui_view_snapshot_pointers"

    view_key = Column(String(64), primary_key=True)
    variant_key = Column(String(128), primary_key=True)
    snapshot_id = Column(
        Integer,
        ForeignKey("ui_view_snapshots.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    snapshot = relationship("UIViewSnapshot", lazy="joined")
