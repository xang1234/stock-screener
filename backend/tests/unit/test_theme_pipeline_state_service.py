"""Tests for pipeline-state reconcile and observability helpers."""

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
from app.services.theme_pipeline_state_service import (
    compute_pipeline_state_health,
    reconcile_source_pipeline_change,
    validate_pipeline_selection,
)


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def test_reconcile_add_pipeline_creates_pending_rows():
    db = _session()
    try:
        source = ContentSource(
            name="Src",
            source_type="news",
            url="https://example.com/rss",
            pipelines=["technical"],
            is_active=True,
        )
        db.add(source)
        db.flush()

        item = ContentItem(
            source_id=source.id,
            source_type=source.source_type,
            source_name=source.name,
            title="Item 1",
            content="Test",
            published_at=datetime.utcnow(),
            is_processed=False,
        )
        db.add(item)
        db.commit()

        result = reconcile_source_pipeline_change(
            db,
            source_id=source.id,
            old_pipelines=["technical"],
            new_pipelines=["technical", "fundamental"],
        )

        assert result["created_pending_rows"] == 1
        row = db.query(ContentItemPipelineState).filter(
            ContentItemPipelineState.content_item_id == item.id,
            ContentItemPipelineState.pipeline == "fundamental",
        ).first()
        assert row is not None
        assert row.status == "pending"
    finally:
        db.close()


def test_reconcile_remove_pipeline_preserves_mentions_and_marks_in_progress():
    db = _session()
    try:
        source = ContentSource(
            name="Src",
            source_type="news",
            url="https://example.com/rss2",
            pipelines=["technical", "fundamental"],
            is_active=True,
        )
        db.add(source)
        db.flush()

        item = ContentItem(
            source_id=source.id,
            source_type=source.source_type,
            source_name=source.name,
            title="Item 2",
            content="Test",
            published_at=datetime.utcnow(),
            is_processed=False,
        )
        db.add(item)
        db.flush()

        mention = ThemeMention(
            content_item_id=item.id,
            source_type=source.source_type,
            source_name=source.name,
            raw_theme="AI",
            canonical_theme="AI",
            pipeline="fundamental",
            sentiment="neutral",
            confidence=0.5,
            mentioned_at=datetime.utcnow(),
        )
        state = ContentItemPipelineState(
            content_item_id=item.id,
            pipeline="fundamental",
            status="in_progress",
            attempt_count=1,
            last_attempt_at=datetime.utcnow() - timedelta(minutes=5),
        )
        db.add_all([mention, state])
        db.commit()

        result = reconcile_source_pipeline_change(
            db,
            source_id=source.id,
            old_pipelines=["technical", "fundamental"],
            new_pipelines=["technical"],
        )

        assert result["removed_in_progress_rows"] == 1
        refreshed_mention = db.query(ThemeMention).filter(ThemeMention.id == mention.id).first()
        refreshed_state = db.query(ContentItemPipelineState).filter(
            ContentItemPipelineState.content_item_id == item.id,
            ContentItemPipelineState.pipeline == "fundamental",
        ).first()
        assert refreshed_mention.pipeline == "fundamental"
        assert refreshed_state.status == "failed_retryable"
        assert refreshed_state.error_code == "source_pipeline_removed"
    finally:
        db.close()


def test_compute_pipeline_state_health_reports_core_metrics():
    db = _session()
    try:
        source = ContentSource(
            name="Src",
            source_type="news",
            url="https://example.com/rss3",
            pipelines=["technical", "fundamental"],
            is_active=True,
        )
        db.add(source)
        db.flush()

        now = datetime.utcnow()
        items = []
        for idx in range(5):
            item = ContentItem(
                source_id=source.id,
                source_type=source.source_type,
                source_name=source.name,
                title=f"Item {idx}",
                content="Test",
                published_at=now - timedelta(hours=idx),
                is_processed=True,
            )
            db.add(item)
            db.flush()
            items.append(item)

        db.add_all(
            [
                ContentItemPipelineState(
                    content_item_id=items[0].id,
                    pipeline="technical",
                    status="pending",
                    created_at=now - timedelta(hours=10),
                ),
                ContentItemPipelineState(
                    content_item_id=items[1].id,
                    pipeline="technical",
                    status="pending",
                    created_at=now - timedelta(hours=2),
                ),
                ContentItemPipelineState(
                    content_item_id=items[2].id,
                    pipeline="technical",
                    status="failed_retryable",
                    error_code="json_decode_error",
                    updated_at=now - timedelta(hours=3),
                ),
                ContentItemPipelineState(
                    content_item_id=items[3].id,
                    pipeline="technical",
                    status="processed",
                    processed_at=now - timedelta(hours=1),
                ),
                ContentItemPipelineState(
                    content_item_id=items[4].id,
                    pipeline="technical",
                    status="processed",
                    processed_at=now - timedelta(hours=1),
                ),
            ]
        )
        db.flush()

        db.add(
            ThemeMention(
                content_item_id=items[4].id,
                source_type=source.source_type,
                source_name=source.name,
                raw_theme="AI infra",
                canonical_theme="AI Infrastructure",
                pipeline="technical",
                sentiment="bullish",
                confidence=0.8,
                mentioned_at=now,
            )
        )
        db.commit()

        report = compute_pipeline_state_health(db, pipeline="technical", max_age_days=30)
        assert len(report["pipelines"]) == 1
        health = report["pipelines"][0]
        assert health["counts"]["pending"] == 2
        assert health["counts"]["failed_retryable"] == 1
        assert health["counts"]["parse_error"] == 1
        assert health["counts"]["processed"] == 2
        assert health["counts"]["processed_without_mentions"] == 1
        assert health["rates"]["processed_without_mentions_ratio"] == 0.5
        assert health["pending_age_hours"]["p95"] >= health["pending_age_hours"]["p50"]
    finally:
        db.close()


def test_validate_pipeline_selection_rejects_invalid_values():
    try:
        validate_pipeline_selection(["technical", "legacy"])
        assert False, "Expected ValueError for invalid pipeline"
    except ValueError as exc:
        assert "Invalid pipelines" in str(exc)


def test_validate_pipeline_selection_rejects_empty():
    try:
        validate_pipeline_selection([])
        assert False, "Expected ValueError for empty pipelines"
    except ValueError as exc:
        assert "At least one valid pipeline" in str(exc)


def test_validate_pipeline_selection_normalizes_and_dedupes():
    pipelines = validate_pipeline_selection(["Technical", "technical", "fundamental"])
    assert pipelines == ["technical", "fundamental"]
