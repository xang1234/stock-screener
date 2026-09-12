"""Durable source observations; synthetic data and disposable repositories only."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.domain.social_signals.records import SocialPostRecord, SocialReadRequest, SocialSourceBatch, SocialSourceOutcome
from app.infra.db.models.social_signals import SocialContentMetrics, SocialPostSource, ContentPipelineEligibility
from app.models.theme import ContentItem
from app.services.social_source_admin_service import SocialSourceAdminService

NOW = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'publication.sqlite'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as db:
        admin = SocialSourceAdminService(db)
        admin.ensure_seed_sources()
    with factory.begin() as db:
        from app.infra.db.models.social_signals import SocialSourceRegistry
        registry = db.get(SocialSourceRegistry, 1)
        registry.mode, registry.provider = "live", "official"
    yield factory
    engine.dispose()


def batch(source=1, *, post_id="100", likes=10, replies=2, age=0, success=True):
    at = NOW + timedelta(hours=age)
    request = SocialReadRequest(f"request-{source}-{post_id}-{age}", str(source),
        "1522014550211457024" if source == 1 else "1986290701492232693", "initial", at, 1000, at - timedelta(days=14))
    post = SocialPostRecord("official", post_id, str(source), "$AAA supplies cooling", f"https://x.com/a/status/{post_id}", "a", NOW-timedelta(days=1), at, likes=likes, replies=replies)
    outcome = SocialSourceOutcome("success" if success else "failed", "pending", "limited", ("post_cap",), (),
        post.created_at, post.created_at, 1, None, None if success else "provider_error", proposed_progress="cursor" if success else None)
    return SocialSourceBatch(request, (post,), outcome)


def writer(factory):
    from app.infra.db.repositories.social_signal_writer import SocialSignalWriter
    return SocialSignalWriter(factory, clock=lambda: NOW)


def test_cross_list_duplicate_retains_memberships_and_only_social_eligibility(store):
    w = writer(store)
    w.persist_observations(batch())
    w.persist_observations(batch(2))
    w.persist_observations(batch())
    with store() as db:
        assert db.query(ContentItem).count() == 1
        assert db.query(SocialPostSource).count() == 2
        assert {r.channel for r in db.query(ContentPipelineEligibility)} == {"social"}


def test_social_writer_reuses_legacy_twitter_content_identity(store):
    legacy_external_id = hashlib.md5(b"twitter:100").hexdigest()
    with store.begin() as db:
        db.add(ContentItem(
            source_type="twitter",
            external_id=legacy_external_id,
            content="legacy copy",
            url="https://x.com/a/status/100",
            published_at=NOW - timedelta(days=1),
        ))

    writer(store).persist_observations(batch())

    with store() as db:
        assert db.query(ContentItem).count() == 1
        assert db.query(SocialPostSource).count() == 1


def test_social_attachment_is_persisted_as_parent_bound_child_evidence(store):
    """Dropping attachment persistence would leave a later social refresh text-only."""
    from app.domain.social_signals.records import SocialAttachmentRef
    from app.models.theme import ContentAttachment

    original = batch()
    attached = replace(
        original,
        posts=(replace(original.posts[0], attachments=(SocialAttachmentRef(
            "image", "https://pbs.twimg.com/media/example.jpg"
        ),)),),
    )

    writer(store).persist_observations(attached)

    with store() as db:
        evidence = db.query(ContentAttachment).one()
        assert (evidence.kind, evidence.url, evidence.status) == (
            "image", "https://pbs.twimg.com/media/example.jpg", "pending"
        )


def test_next_social_generation_rehydrates_prepared_evidence_for_a_retained_parent(store):
    """Ignoring retained parents would strand a late image result until X repeats the post."""
    from app.domain.social_signals.records import SocialAttachmentRef
    from app.infra.db.repositories.social_refresh_support import SocialScoringEvidenceReader
    from app.models.theme import ContentAttachment

    w = writer(store)
    w.create_run("old", NOW)
    initial = batch()
    initial = replace(initial, posts=(replace(initial.posts[0], attachments=(SocialAttachmentRef(
        "image", "https://pbs.twimg.com/media/nebius.jpg"
    ),)),))
    w.persist_observations(initial, run_id="old")
    w.persist_observations(batch(2, post_id="200"), run_id="old")
    with store.begin() as db:
        attachment = db.query(ContentAttachment).one()
        attachment.status = "complete"
        attachment.prepared_at = NOW + timedelta(minutes=30)
        attachment.original_text = "Nebius slide"
        attachment.prepared_text = "Nebius ($NBIS) is expanding AI cloud capacity."
        attachment.content_sha256 = "a" * 64
        attachment.final_url = attachment.url
        attachment.provenance = {"policy_version": "live-attachment-v1"}

    later = NOW + timedelta(hours=1)
    w.clock = lambda: later + timedelta(minutes=1)
    w.create_run("new", later)
    w.persist_observations(batch(1, post_id="300", age=1), run_id="new")
    w.persist_observations(batch(2, post_id="400", age=1), run_id="new")

    retained = SocialScoringEvidenceReader(store).retained_posts("new", later)

    retained_post = next(post for _, post in retained if post.provider_post_id == "100")
    assert retained_post.evidence_digest is not None
    assert retained_post.prepared_evidence[0].text == "Nebius ($NBIS) is expanding AI cloud capacity."


def test_newer_partial_metrics_preserve_missing_and_older_cannot_overwrite(store):
    w = writer(store)
    w.persist_observations(batch())
    w.clock = lambda: NOW + timedelta(hours=1)
    w.persist_observations(batch(likes=20, replies=None, age=1))
    w.persist_observations(batch(likes=1, replies=1))
    with store() as db:
        metrics = db.query(SocialContentMetrics).one()
        assert (metrics.likes, metrics.replies, metrics.reposts) == (20, 2, None)


def test_crash_rolls_back_observations_and_progress(store, monkeypatch):
    w = writer(store)
    run = w.create_run("run", NOW)
    from sqlalchemy import event
    def crash(session):
        raise RuntimeError("synthetic crash")
    event.listen(store.class_, "before_commit", crash)
    try:
        with pytest.raises(RuntimeError, match="synthetic crash"):
            w.persist_observations(batch(), run_id=run)
    finally:
        event.remove(store.class_, "before_commit", crash)
    with store() as db:
        assert db.query(ContentItem).count() == 0
    assert w.latest_committed_progress("1", "official") is None
    saved = w.persist_observations(batch(), run_id=run)
    assert saved.outcome.committed_progress == "cursor"
    assert w.latest_committed_progress("1", "official") == "cursor"
    progress = w.latest_collection_progress("1", "official")
    assert progress.initial_complete is False
    assert progress.cursor == "cursor"


def test_cursorless_success_is_a_durable_completed_initial_marker(store):
    w = writer(store)
    w.create_run("run", NOW)
    value = batch()
    value = replace(
        value,
        outcome=replace(value.outcome, proposed_progress=None),
    )

    w.persist_observations(value, run_id="run")

    progress = w.latest_collection_progress("1", "official")
    assert progress.initial_complete is True
    assert progress.cursor is None


def test_failed_and_diagnostic_reads_never_commit_progress(store):
    w = writer(store)
    saved = w.persist_observations(batch(success=False))
    assert saved.outcome.committed_progress is None
    diagnostic = replace(batch(), request=replace(batch().request, intent="test", limit=5))
    with pytest.raises(ValueError, match="diagnostic"):
        w.persist_observations(diagnostic)


def test_observed_cashtag_mapping_is_deduplicated_without_fabricated_market(store):
    from app.infra.db.models.social_signals import SocialPostTicker
    w = writer(store)
    w.persist_observations(batch())
    w.persist_observations(batch(2))
    with store() as db:
        mapping = db.query(SocialPostTicker).one()
        assert mapping.raw_token == "$AAA"
        assert mapping.resolution_state == "unresolved"
        assert mapping.market is mapping.stock_universe_id is None


def test_changed_duplicate_delivery_cannot_replace_frozen_input(store):
    w = writer(store)
    w.create_run("run", NOW)
    w.persist_observations(batch(), run_id="run")
    with pytest.raises(ValueError, match="duplicate"):
        w.persist_observations(batch(likes=999), run_id="run")


def test_progress_order_uses_observation_time_and_survives_budget_pause(store):
    w = writer(store)
    w.clock = lambda: NOW + timedelta(hours=1)
    w.create_run("new", NOW)
    newer = replace(batch(age=1), outcome=replace(batch(age=1).outcome, proposed_progress="new-cursor"))
    w.persist_observations(newer, run_id="new")
    w.create_run("old", NOW-timedelta(hours=1))
    w.persist_observations(batch(), run_id="old")
    assert w.latest_committed_progress("1", "official") == "new-cursor"
    with store() as db:
        from app.infra.db.models.social_signals import SocialSignalRunPointer
        assert db.query(SocialSignalRunPointer).count() == 0


def test_disabled_source_cannot_commit_delayed_observation(store):
    from app.infra.db.models.social_signals import SocialSourceConfiguration
    w = writer(store)
    with store.begin() as db:
        db.get(SocialSourceConfiguration, 1).lifecycle_state = "disabled"
    with pytest.raises(ValueError, match="source.*enabled"):
        w.persist_observations(batch())
