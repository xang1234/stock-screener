"""Unit tests for published UI snapshot persistence semantics."""

from __future__ import annotations

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from app.db_migrations.ui_view_snapshot_migration import migrate_ui_view_snapshot_tables
from app.models.ui_view_snapshot import UIViewSnapshot
from app.services.ui_snapshot_service import UISnapshotService


def test_ui_snapshot_service_marks_outdated_pointer_reads_as_stale_and_prunes_old_revisions():
    engine = create_engine("sqlite:///:memory:")
    Session = sessionmaker(bind=engine)
    migrate_ui_view_snapshot_tables(engine)

    service = UISnapshotService(Session)

    with Session() as db:
        first = service._publish(  # noqa: SLF001 - unit test exercises persistence semantics directly
            db=db,
            view_key="test_view",
            variant_key="default",
            source_revision="rev-1",
            payload={"value": 1},
        )
        fresh = service._get_snapshot(  # noqa: SLF001
            db=db,
            view_key="test_view",
            variant_key="default",
            source_revision="rev-1",
        )
        stale = service._get_snapshot(  # noqa: SLF001
            db=db,
            view_key="test_view",
            variant_key="default",
            source_revision="rev-2",
        )

        assert first.is_stale is False
        assert fresh is not None and fresh.is_stale is False
        assert stale is not None and stale.is_stale is True
        assert stale.source_revision == "rev-1"
        assert stale.payload == {"value": 1}

        second = service._publish(  # noqa: SLF001
            db=db,
            view_key="test_view",
            variant_key="default",
            source_revision="rev-2",
            payload={"value": 2},
        )
        third = service._publish(  # noqa: SLF001
            db=db,
            view_key="test_view",
            variant_key="default",
            source_revision="rev-3",
            payload={"value": 3},
        )

        kept_revisions = {
            row.source_revision
            for row in db.query(UIViewSnapshot)
            .filter(
                UIViewSnapshot.view_key == "test_view",
                UIViewSnapshot.variant_key == "default",
            )
            .all()
        }

        assert second.snapshot_revision != third.snapshot_revision
        assert kept_revisions == {"rev-2", "rev-3"}
        assert db.query(func.count(UIViewSnapshot.id)).scalar() == 2
