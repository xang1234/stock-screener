"""Durable source observations; synthetic data and disposable repositories only."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

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
