"""Tests for chunked pipeline-state backfill service."""

from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.theme import (
    ContentItem,
    ContentItemPipelineState,
    ContentSource,
    ThemeMention,
)
from app.services.theme_pipeline_state_backfill_service import ThemePipelineStateBackfillService


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def test_backfill_chunk_resume_and_idempotency():
    db = _session()
    try:
        source_both = ContentSource(
            name="Both",
            source_type="news",
            url="https://example.com/both",
            pipelines=["technical", "fundamental"],
            is_active=True,
        )
        source_tech = ContentSource(
            name="Tech",
            source_type="news",
            url="https://example.com/tech",
            pipelines=["technical"],
            is_active=True,
        )
        db.add_all([source_both, source_tech])
        db.flush()

        item1 = ContentItem(
            source_id=source_both.id,
            source_type=source_both.source_type,
            source_name=source_both.name,
            title="Pending item",
            content="No processing yet",
            published_at=datetime.utcnow() - timedelta(days=1),
            is_processed=False,
        )
        item2 = ContentItem(
            source_id=source_both.id,
            source_type=source_both.source_type,
            source_name=source_both.name,
            title="Ambiguous processed item",
            content="Legacy processed without mentions",
            published_at=datetime.utcnow() - timedelta(days=1),
            is_processed=True,
            processed_at=datetime.utcnow() - timedelta(hours=5),
        )
        item3 = ContentItem(
            source_id=source_tech.id,
            source_type=source_tech.source_type,
            source_name=source_tech.name,
            title="Mentioned tech item",
            content="AI mention",
            published_at=datetime.utcnow() - timedelta(hours=6),
            is_processed=True,
            extraction_error="legacy timeout",
            processed_at=datetime.utcnow() - timedelta(hours=4),
        )
        db.add_all([item1, item2, item3])
        db.flush()

        db.add(
            ThemeMention(
                content_item_id=item3.id,
                source_type=source_tech.source_type,
                source_name=source_tech.name,
                raw_theme="AI",
                canonical_theme="AI",
                pipeline="technical",
                sentiment="neutral",
                confidence=0.8,
                mentioned_at=datetime.utcnow() - timedelta(hours=4),
            )
        )
        db.add(
            ThemeMention(
                content_item_id=item3.id,
                source_type=source_tech.source_type,
                source_name=source_tech.name,
                raw_theme="Legacy marker",
                canonical_theme="Legacy marker",
                pipeline="legacy_pipeline",
                sentiment="neutral",
                confidence=0.3,
                mentioned_at=datetime.utcnow() - timedelta(hours=4),
            )
        )
        db.commit()

        service = ThemePipelineStateBackfillService(db)

        first = service.process_chunk(after_content_item_id=0, limit=2, dry_run=False)
        assert first.rows_read == 2
        assert first.rows_written == 4
        assert first.conflicts == 0

        second = service.process_chunk(after_content_item_id=first.next_cursor, limit=2, dry_run=False)
        assert second.rows_read == 1
        assert second.rows_written == 1
        assert second.conflicts == 0

        rerun = service.process_chunk(after_content_item_id=0, limit=10, dry_run=False)
        assert rerun.rows_read == 3
        assert rerun.rows_written == 0
        assert rerun.conflicts == 5

        total_rows = db.query(ContentItemPipelineState).count()
        assert total_rows == 5
    finally:
        db.close()


def test_backfill_dry_run_does_not_write_rows():
    db = _session()
    try:
        source = ContentSource(
            name="DryRun",
            source_type="news",
            url="https://example.com/dry",
            pipelines=["technical", "fundamental"],
            is_active=True,
        )
        db.add(source)
        db.flush()
        item = ContentItem(
            source_id=source.id,
            source_type=source.source_type,
            source_name=source.name,
            title="Dry run item",
            content="Test",
            published_at=datetime.utcnow(),
            is_processed=False,
        )
        db.add(item)
        db.commit()

        service = ThemePipelineStateBackfillService(db)
        result = service.process_chunk(after_content_item_id=0, limit=10, dry_run=True)
        assert result.rows_read == 1
        assert result.rows_written == 2
        assert db.query(ContentItemPipelineState).count() == 0
    finally:
        db.close()
