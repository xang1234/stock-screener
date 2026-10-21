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
    factory = sessionmaker(engine, expire_on_commit=False, autoflush=False)
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


def test_a_batch_repeating_a_post_records_it_once(store):
    single = batch()
    post = single.posts[0]
    repeated = replace(single, posts=(post, replace(post, likes=11)),
        outcome=replace(single.outcome, received_count=2))

    writer(store).persist_observations(repeated)

    with store() as db:
        assert db.query(ContentItem).count() == 1
        assert db.query(SocialPostSource).count() == 1
        assert db.query(SocialContentMetrics).one().likes == 10


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


def test_collection_progress_reports_the_latest_successful_read_of_any_intent(store):
    w = writer(store)
    w.create_run("initial", NOW)
    w.persist_observations(batch(), run_id="initial")
    later = NOW + timedelta(hours=6)
    w.clock = lambda: later
    w.create_run("incremental", later)
    incremental = batch(age=6)
    w.persist_observations(replace(incremental, request=replace(incremental.request, intent="incremental")),
                           run_id="incremental")
    w.create_run("failed", later + timedelta(minutes=1))
    w.persist_observations(batch(age=6, success=False, post_id="101"), run_id="failed")

    assert w.latest_collection_progress("1", "official").last_read_at == later


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


def test_reobserved_post_replaces_stale_ticker_rows_and_skips_money(store):
    """A row resolved before the universe loaded must not linger as 'unresolved'."""
    from app.infra.db.models.social_signals import SocialPostTicker
    from app.models.stock_universe import StockUniverse
    w = writer(store)
    first = batch()
    money = replace(first, posts=(replace(first.posts[0], text="$AAA raised $10K, $1.5B, $680M and $300."),))
    w.persist_observations(money)
    with store.begin() as db:
        db.add(StockUniverse(symbol="AAA", name="AAA Corp", market="US", exchange="NASDAQ", is_active=True))
    w.clock = lambda: NOW + timedelta(hours=1)
    w.persist_observations(replace(money, posts=(replace(money.posts[0], observed_at=NOW + timedelta(hours=1)),)))
    with store() as db:
        rows = db.query(SocialPostTicker).all()
        assert [(row.candidate_key, row.resolution_state) for row in rows] == [("US:AAA", "resolved")]


def test_scoring_reader_ignores_a_stale_unresolved_mapping(store):
    """Posts that are never re-observed keep old rows; the reader resolves the text fresh."""
    from app.infra.db.repositories.social_refresh_support import SocialScoringEvidenceReader
    from app.models.stock_universe import StockUniverse
    w = writer(store)
    w.create_run("run", NOW)
    first = batch()
    w.persist_observations(replace(first, posts=(replace(first.posts[0], text="$AAA hit $50M"),)), run_id="run")
    w.persist_observations(batch(2, post_id="200"), run_id="run")
    with store.begin() as db:
        db.add(StockUniverse(symbol="AAA", name="AAA Corp", market="US", exchange="NASDAQ", is_active=True))
    w.select_current_inputs("run")
    keys = {item.candidate_key for item in SocialScoringEvidenceReader(store).read("run", NOW)}
    assert keys == {"US:AAA"}


def test_a_failed_work_does_not_hide_a_succeeded_one_for_the_same_input(store):
    from app.infra.db.models.social_analysis import SocialExtractionWork, SocialRunWork
    from app.services.social_extraction_service import SocialExtractionService

    w = writer(store)
    w.create_run("run", NOW)
    saved = w.persist_observations(batch(), run_id="run")
    w.persist_observations(batch(2, post_id="200"), run_id="run")
    content_id = dict(w.read_run_inputs("run").content_ids)["100"]
    input_hash = SocialExtractionService.input_hash(saved.posts)
    with store.begin() as db:
        for model, state in (("model-a", "succeeded"), ("model-b", "failed_terminal")):
            work = SocialExtractionWork(
                content_item_id=content_id, input_hash=input_hash, prompt_version="v1",
                schema_version="v1", selected_model=model, input_snapshot_json={},
                state=state, created_at=NOW, updated_at=NOW,
            )
            db.add(work)
            db.flush()
            db.add(SocialRunWork(run_id="run", work_id=work.id, input_hash=input_hash, included_at=NOW))

    manifest = w.select_current_inputs("run").current_manifest

    assert manifest.unavailable_inputs == ()
    assert (content_id, input_hash) in {(v.content_item_id, v.input_hash) for v in manifest.required_inputs}


def test_retained_work_that_aged_out_of_analysis_does_not_hold_the_generation(store):
    from app.infra.db.models.social_analysis import SocialExtractionWork, SocialRunWork

    w = writer(store)
    w.create_run("run", NOW)
    w.persist_observations(batch(), run_id="run")
    w.persist_observations(batch(2, post_id="200"), run_id="run")
    created = NOW - timedelta(days=14, hours=12)
    with store.begin() as db:
        item = ContentItem(source_type="twitter", external_id="retained", url="https://x.com/a/status/9",
                           content="$AAA", published_at=created)
        db.add(item)
        db.flush()
        work = SocialExtractionWork(
            content_item_id=item.id, input_hash="retained-hash", prompt_version="v1",
            schema_version="v1", selected_model="model",
            input_snapshot_json={"created_at": created.isoformat()},
            state="outside_window", error_code="outside_signal_window", created_at=NOW, updated_at=NOW,
        )
        db.add(work)
        db.flush()
        db.add(SocialRunWork(run_id="run", work_id=work.id, input_hash="retained-hash", included_at=NOW))
        work_id = work.id

    manifest = w.select_current_inputs("run").current_manifest

    assert (manifest.scoring_work_ids, work_id in manifest.audit_work_ids) == ((), True)


def test_unavailable_revision_does_not_hide_another_revision_of_the_same_post(store):
    from app.infra.db.models.social_analysis import SocialExtractionWork, SocialRunWork
    from app.infra.db.repositories.social_refresh_support import SocialScoringEvidenceReader
    from app.models.stock_universe import StockUniverse
    from app.services.social_extraction_service import SocialExtractionService

    w = writer(store)
    w.create_run("run", NOW)
    first = w.persist_observations(batch(), run_id="run")
    edited = batch(2)
    edited = replace(edited, posts=(replace(edited.posts[0], text="$AAA supplies cooling, edited"),))
    second = w.persist_observations(edited, run_id="run")
    content_id = dict(w.read_run_inputs("run").content_ids)["100"]
    with store.begin() as db:
        db.add(StockUniverse(symbol="AAA", name="AAA Corp", market="US", exchange="NASDAQ", is_active=True))
        for saved, state in ((first, "failed_terminal"), (second, "succeeded")):
            input_hash = SocialExtractionService.input_hash(saved.posts)
            work = SocialExtractionWork(
                content_item_id=content_id, input_hash=input_hash, prompt_version="v1",
                schema_version="v1", selected_model="model", input_snapshot_json={},
                state=state, result_json={"claims": [], "judgments": []} if state == "succeeded" else None,
                created_at=NOW, updated_at=NOW,
            )
            db.add(work)
            db.flush()
            db.add(SocialRunWork(run_id="run", work_id=work.id, input_hash=input_hash, included_at=NOW))
    w.select_current_inputs("run")

    evidence = SocialScoringEvidenceReader(store).read("run", NOW)

    assert [post.text for item in evidence for post in item.posts] == ["$AAA supplies cooling, edited"]
