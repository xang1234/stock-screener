"""Live attachment persistence, delayed evidence and idempotent preparation."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from app.database import Base
from app.models.theme import ContentAttachment, ContentItem, ContentItemPipelineState
from app.services.live_attachment_service import (
    attachment_snapshot,
    build_live_grounding,
    prepare_pending_attachment,
    reconcile_attachment_revisions,
    record_attachments,
)
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

NOW = datetime(2026, 9, 11, tzinfo=timezone.utc)


@pytest.fixture
def sessions():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def seed(sessions):
    with sessions.begin() as db:
        item = ContentItem(
            source_type="twitter",
            external_id="1",
            content="$NBIS expansion",
            url="https://x.com/a/status/1",
        )
        db.add(item)
        db.flush()
        record_attachments(
            db,
            item,
            [{"kind": "image", "url": "https://pbs.twimg.com/media/a.jpg"}],
            NOW,
        )
        return item.id


def prepared(kind, url):
    return SimpleNamespace(
        text="Nebius expands AI infrastructure.",
        original_text="Nebius expands AI infrastructure.",
        content_sha256="a" * 64,
        final_url=url,
        status="complete",
        provenance={"model": "kimi-k2.6"},
    )


def test_duplicate_ingestion_and_success_do_not_repeat_model(sessions):
    item_id = seed(sessions)
    with sessions.begin() as db:
        item = db.get(ContentItem, item_id)
        record_attachments(
            db,
            item,
            [{"kind": "image", "url": "https://pbs.twimg.com/media/a.jpg"}],
            NOW,
        )
        assert len(db.scalars(select(ContentAttachment)).all()) == 1
        assert attachment_snapshot(db, item_id)["status"] == "pending"
    calls = []

    def process(kind, url):
        calls.append(url)
        return prepared(kind, url)

    assert prepare_pending_attachment(sessions, process, now=NOW)
    assert not prepare_pending_attachment(sessions, process, now=NOW)
    assert len(calls) == 1
    with sessions() as db:
        snap = attachment_snapshot(db, item_id)
        assert snap["status"] == "complete"
        assert len(snap["evidence"]) == 1
        assert snap["evidence"][0]["text"].startswith("Nebius")
        assert (
            attachment_snapshot(db, item_id, as_of=NOW - timedelta(seconds=1))[
                "evidence"
            ]
            == []
        )
        context = build_live_grounding(db, db.get(ContentItem, item_id), now=NOW)
        assert context.evidence[0].relation == "attached_image"
        assert context.evidence[0].source_url.endswith("a.jpg")


def test_failure_is_visible_and_retry_is_bounded(sessions):
    item_id = seed(sessions)

    def fail(*args):
        raise OSError("secret provider payload must not be persisted")

    for n in range(3):
        assert prepare_pending_attachment(sessions, fail, now=NOW + timedelta(hours=n))
    assert not prepare_pending_attachment(sessions, fail, now=NOW + timedelta(days=1))
    with sessions() as db:
        row = db.scalar(select(ContentAttachment))
        assert row.status == "failed"
        assert row.attempt_count == 3
        assert "secret" not in row.error_code
        assert attachment_snapshot(db, item_id)["status"] == "failed"


def test_late_evidence_invalidates_processed_state_but_not_inflight(sessions):
    item_id = seed(sessions)
    with sessions.begin() as db:
        old = attachment_snapshot(db, item_id)["revision"]
        db.add(
            ContentItemPipelineState(
                content_item_id=item_id,
                pipeline="technical",
                status="processed",
                evidence_revision=old,
            )
        )
        db.add(
            ContentItemPipelineState(
                content_item_id=item_id,
                pipeline="fundamental",
                status="in_progress",
                evidence_revision=old,
            )
        )
    prepare_pending_attachment(sessions, prepared, now=NOW)
    with sessions.begin() as db:
        reconcile_attachment_revisions(db, item_id)
        states = {s.pipeline: s for s in db.scalars(select(ContentItemPipelineState))}
        assert states["technical"].status == "pending"
        assert states["fundamental"].status == "in_progress"
        # A stale extraction finishing after evidence arrived is caught next sweep.
        states["fundamental"].status = "processed"
    with sessions.begin() as db:
        reconcile_attachment_revisions(db, item_id)
        assert (
            db.scalar(
                select(ContentItemPipelineState).where(
                    ContentItemPipelineState.pipeline == "fundamental"
                )
            ).status
            == "pending"
        )


def test_reextraction_replaces_mentions_atomically_and_preserves_other_pipeline(
    sessions,
):
    from unittest.mock import MagicMock

    from app.models.theme import ThemeCluster, ThemeMention
    from app.services.theme_extraction_service import ThemeExtractionService

    item_id = seed(sessions)
    with sessions.begin() as db:
        cluster = ThemeCluster(
            name="AI",
            canonical_key="ai",
            display_name="AI",
            pipeline="technical",
            parent_cluster_id=1,
        )
        db.add(cluster)
        db.flush()
        cluster.parent_cluster_id = cluster.id
        for pipeline in ("technical", "fundamental"):
            db.add(
                ThemeMention(
                    content_item_id=item_id,
                    theme_cluster_id=cluster.id,
                    pipeline=pipeline,
                    raw_theme="AI",
                    source_type="twitter",
                    tickers=[],
                )
            )
    with sessions() as db:
        item, cluster = db.get(ContentItem, item_id), db.scalar(select(ThemeCluster))
        svc = ThemeExtractionService.__new__(ThemeExtractionService)
        svc.db, svc.pipeline = db, "technical"
        svc.extract_from_content = MagicMock(return_value=[])
        svc._extract_and_store_mentions(item)
        db.commit()
        assert [m.pipeline for m in db.scalars(select(ThemeMention))] == ["fundamental"]
        svc.extract_from_content.side_effect = RuntimeError("provider unavailable")
        svc.pipeline = "fundamental"
        with pytest.raises(RuntimeError):
            svc._extract_and_store_mentions(item)
        db.rollback()
        assert len(db.scalars(select(ThemeMention)).all()) == 1


def test_partial_translation_is_visible_and_retried(sessions):
    item_id = seed(sessions)

    def partial(kind, url):
        result = prepared(kind, url)
        result.status = "partial"
        result.provenance = {
            "translation": {"retry": {"needed": True, "codes": ["model_timeout"]}}
        }
        return result

    prepare_pending_attachment(sessions, partial, now=NOW)
    with sessions() as db:
        assert attachment_snapshot(db, item_id)["status"] == "partial"
    assert not prepare_pending_attachment(sessions, prepared, now=NOW)
    assert prepare_pending_attachment(sessions, prepared, now=NOW + timedelta(hours=1))
    with sessions() as db:
        assert attachment_snapshot(db, item_id)["status"] == "complete"


def test_stale_lease_cannot_overwrite_newer_success(sessions):
    item_id = seed(sessions)

    def slow(kind, url):
        assert prepare_pending_attachment(
            sessions, prepared, now=NOW + timedelta(minutes=16)
        )
        stale = prepared(kind, url)
        stale.text = "wrong stale result"
        return stale

    assert prepare_pending_attachment(sessions, slow, now=NOW)
    with sessions() as db:
        assert (
            attachment_snapshot(db, item_id)["evidence"][0]["text"]
            == "Nebius expands AI infrastructure."
        )


def test_same_url_on_different_post_is_not_assumed_same_content(sessions):
    seed(sessions)
    prepare_pending_attachment(sessions, prepared, now=NOW)
    with sessions.begin() as db:
        second = ContentItem(
            source_type="twitter", external_id="2", content="new information"
        )
        db.add(second)
        db.flush()
        record_attachments(
            db,
            second,
            [{"kind": "image", "url": "https://pbs.twimg.com/media/a.jpg"}],
            NOW,
        )
    calls = []

    def changed(kind, url):
        calls.append(url)
        return prepared(kind, url)

    assert prepare_pending_attachment(sessions, changed, now=NOW)
    assert len(calls) == 1


def test_task_respects_enablement_and_dispatches_after_preparation(
    sessions, monkeypatch
):
    from app import database
    from app.config import settings
    from app.infra.db.models.social_signals import ContentPipelineEligibility
    from app.interfaces.tasks import social_signal_tasks
    from app.models.theme import ContentSource
    from app.services import live_attachment_preparation
    from app.tasks import live_attachment_tasks

    item_id = seed(sessions)
    monkeypatch.setenv("LIVE_ATTACHMENT_PREPARATION_ENABLED", "false")
    assert live_attachment_tasks.prepare_live_attachments.run()["status"] == "disabled"
    with sessions.begin() as db:
        source = ContentSource(
            name="Investment research", source_type="substack", is_active=True
        )
        db.add(source)
        db.flush()
        db.add(
            ContentPipelineEligibility(
                content_item_id=item_id,
                pipeline="technical",
                channel="legacy",
                originating_source_id=source.id,
                observed_at=NOW,
            )
        )
    monkeypatch.setenv("LIVE_ATTACHMENT_PREPARATION_ENABLED", "true")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "fake-key")
    monkeypatch.setattr(settings, "feature_themes", True)
    monkeypatch.setattr(database, "SessionLocal", sessions)
    monkeypatch.setattr(
        live_attachment_preparation, "LiveAttachmentPreparer", lambda key: prepared
    )
    dispatched = []
    monkeypatch.setattr(
        live_attachment_tasks.refresh_attachment_themes,
        "delay",
        lambda ids: dispatched.append(("legacy", ids)),
    )
    monkeypatch.setattr(
        social_signal_tasks.refresh_social_signals,
        "apply_async",
        lambda **kwargs: dispatched.append("social"),
    )
    result = live_attachment_tasks.prepare_live_attachments.run()
    assert result["processed"] == 1
    assert dispatched == [("legacy", [item_id]), "social"]
    with sessions() as db:
        assert attachment_snapshot(db, item_id)["status"] == "complete"


def test_disabled_source_does_not_prepare(sessions, monkeypatch):
    from app import database
    from app.tasks import live_attachment_tasks

    seed(sessions)  # No active source grant.
    monkeypatch.setenv("LIVE_ATTACHMENT_PREPARATION_ENABLED", "true")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "fake-key")
    monkeypatch.setattr(database, "SessionLocal", sessions)
    assert live_attachment_tasks.prepare_live_attachments.run()["processed"] == 0


def test_fragment_variants_and_identical_content_are_one_grounding_source(sessions):
    item_id = seed(sessions)
    with sessions.begin() as db:
        record_attachments(
            db,
            db.get(ContentItem, item_id),
            [
                {"kind": "image", "url": "https://pbs.twimg.com/media/a.jpg#one"},
                {"kind": "image", "url": "https://pbs.twimg.com/media/a.jpg#two"},
                {"kind": "image", "url": "https://pbs.twimg.com/media/copy.jpg"},
            ],
            NOW,
        )
        assert len(db.scalars(select(ContentAttachment)).all()) == 2
    prepare_pending_attachment(sessions, prepared, now=NOW)
    prepare_pending_attachment(sessions, prepared, now=NOW)
    with sessions() as db:
        assert len(attachment_snapshot(db, item_id)["evidence"]) == 1


def test_authorization_rechecked_inside_claim_and_before_publish(sessions):
    seed(sessions)
    calls = []
    assert prepare_pending_attachment(
        sessions,
        lambda *_: calls.append("external"),
        now=NOW,
        authorize=lambda *_: False,
    )
    assert calls == []
    checks = iter([True, False])
    assert prepare_pending_attachment(
        sessions,
        prepared,
        now=NOW + timedelta(minutes=16),
        authorize=lambda *_: next(checks),
    )
    with sessions() as db:
        row = db.scalar(select(ContentAttachment))
        assert row.prepared_text is None
        assert row.status == "pending"


def test_attachment_migration_roundtrip():
    import importlib.util
    from pathlib import Path

    import sqlalchemy as sa
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    path = (
        Path(__file__).parents[2]
        / "alembic/versions/20260911_0041_add_content_attachments.py"
    )
    spec = importlib.util.spec_from_file_location("attachment_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE content_items (id INTEGER PRIMARY KEY)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE content_item_pipeline_state (id INTEGER PRIMARY KEY)"
        )
        migration.op = Operations(MigrationContext.configure(connection))
        migration.upgrade()
        assert "content_attachments" in sa.inspect(connection).get_table_names()
        assert "attachment_revision" in {
            c["name"] for c in sa.inspect(connection).get_columns("content_items")
        }
        migration.downgrade()
        assert "content_attachments" not in sa.inspect(connection).get_table_names()
        assert "attachment_revision" not in {
            c["name"] for c in sa.inspect(connection).get_columns("content_items")
        }


def test_new_attachment_update_is_not_dropped_by_large_recovery_backlog(
    sessions, monkeypatch
):
    from app import database
    from app.config import settings
    from app.infra.db.models.social_signals import ContentPipelineEligibility
    from app.interfaces.tasks import social_signal_tasks
    from app.models.theme import ContentSource
    from app.services import live_attachment_preparation
    from app.tasks import live_attachment_tasks

    with sessions.begin() as db:
        source = ContentSource(name="Research", source_type="substack", is_active=True)
        db.add(source)
        db.flush()
        for index in range(101):
            item = ContentItem(
                source_type="twitter",
                external_id=f"backlog-{index}",
                content="Research",
                attachment_revision="a" * 64,
            )
            db.add(item)
            db.flush()
            db.add(
                ContentPipelineEligibility(
                    content_item_id=item.id,
                    pipeline="technical",
                    channel="legacy",
                    originating_source_id=source.id,
                    observed_at=NOW,
                )
            )
            db.add(
                ContentItemPipelineState(
                    content_item_id=item.id,
                    pipeline="technical",
                    status="pending",
                    updated_at=NOW,
                )
            )
        newest = item.id
        record_attachments(
            db,
            item,
            [{"kind": "image", "url": "https://pbs.twimg.com/media/new.jpg"}],
            NOW,
        )
    monkeypatch.setenv("LIVE_ATTACHMENT_PREPARATION_ENABLED", "true")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "fake")
    monkeypatch.setattr(settings, "feature_themes", True)
    monkeypatch.setattr(database, "SessionLocal", sessions)
    monkeypatch.setattr(
        live_attachment_preparation, "LiveAttachmentPreparer", lambda key: prepared
    )
    targeted = []
    monkeypatch.setattr(
        live_attachment_tasks.refresh_attachment_themes,
        "delay",
        lambda ids: targeted.append(ids),
    )
    monkeypatch.setattr(
        social_signal_tasks.refresh_social_signals, "apply_async", lambda **kwargs: None
    )
    live_attachment_tasks.prepare_live_attachments.run()
    assert newest in targeted[0]
    assert len(targeted[0]) == 100


def test_deferred_unauthorized_parent_does_not_block_later_attachment(sessions):
    first = seed(sessions)
    with sessions.begin() as db:
        item = ContentItem(
            source_type="twitter", external_id="next", content="Research"
        )
        db.add(item)
        db.flush()
        record_attachments(
            db,
            item,
            [{"kind": "image", "url": "https://pbs.twimg.com/media/next.jpg"}],
            NOW,
        )
        second = item.id
    allow = lambda db, item_id: item_id != first
    assert prepare_pending_attachment(sessions, prepared, now=NOW, authorize=allow)
    assert prepare_pending_attachment(sessions, prepared, now=NOW, authorize=allow)
    with sessions() as db:
        assert attachment_snapshot(db, second)["status"] == "complete"
