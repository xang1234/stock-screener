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


def saved_work(db, content_id, post, claims=None, *, model="model"):
    """A succeeded row whose saved result revalidates against ``post``.

    ``claims`` are claim dicts as the parser emits them. A claim quoting text the
    post does not contain is what a stricter parser leaves behind in old rows.
    """
    from dataclasses import asdict
    from app.domain.social_signals.records import ExtractionClaim, ExtractionPostJudgment, ExtractionResult
    from app.infra.db.models.social_analysis import SocialExtractionWork
    from app.infra.db.repositories.social_signal_writer import serialized
    from app.services.social_extraction_service import SocialExtractionService

    if claims is None:
        claims = [flagged_claim("$AAA")]
    input_hash = SocialExtractionService.input_hash((post,))
    result = ExtractionResult(input_hash, "synthetic", "actual", "v1", "v1", tuple(
        ExtractionClaim(**{**claim, "post_id": post.provider_post_id,
                           "duplicate_of_post_ids": tuple(claim["duplicate_of_post_ids"])})
        for claim in claims
    ), None, None, (ExtractionPostJudgment(post.provider_post_id, True, "aaa-cooling"),))
    work = SocialExtractionWork(
        content_item_id=content_id, input_hash=input_hash, prompt_version="v1", schema_version="v1",
        selected_model=model, actual_provider="synthetic", actual_model="actual",
        input_snapshot_json=serialized(post), result_json=asdict(result), state="succeeded",
        created_at=NOW, updated_at=NOW,
    )
    db.add(work)
    db.flush()
    return work


def link(db, run_id, work, at=NOW):
    from app.infra.db.models.social_analysis import SocialRunWork
    db.add(SocialRunWork(run_id=run_id, work_id=work.id, input_hash=work.input_hash, included_at=at))


def unquoted_excerpt(store, work_id):
    """Rewrite the saved claims to quote text the post never contained."""
    from app.infra.db.models.social_analysis import SocialExtractionWork
    with store.begin() as db:
        work = db.get(SocialExtractionWork, work_id)
        work.result_json = {**work.result_json, "claims": [
            {**claim, "excerpt": "not in the post"} for claim in work.result_json["claims"]]}


def work_state(store, work_id):
    from app.infra.db.models.social_analysis import SocialExtractionWork
    with store() as db:
        work = db.get(SocialExtractionWork, work_id)
        return work.state, work.error_code


def empty_batch(source):
    observed = batch(source)
    return replace(observed, posts=(), outcome=replace(
        observed.outcome, received_count=0, observed_oldest_at=None, observed_newest_at=None))


def repeated_post_batch():
    single = batch()
    post = single.posts[0]
    return replace(single, posts=(post, replace(post, likes=11)),
        outcome=replace(single.outcome, received_count=2))


def test_a_batch_repeating_a_post_records_it_once(store):
    from app.infra.db.models.social_signals import SocialPostTicker

    writer(store).persist_observations(repeated_post_batch())

    with store() as db:
        assert db.query(ContentItem).count() == 1
        assert db.query(SocialPostSource).count() == 1
        assert db.query(SocialPostTicker).count() == 1
        assert db.query(SocialContentMetrics).one().likes == 10


def test_a_redelivered_batch_repeating_a_post_is_accepted_as_the_same_delivery(store):
    from app.infra.db.models.social_signals import SocialSignalRun

    w = writer(store)
    run = w.create_run("run", NOW)
    w.persist_observations(repeated_post_batch(), run_id=run)

    saved = w.persist_observations(repeated_post_batch(), run_id=run)

    assert saved.outcome.committed_progress == "cursor"
    with store() as db:
        inputs = db.get(SocialSignalRun, run).application_progress_json["observations"]["1"]["inputs"]
    assert [value["post"]["provider_post_id"] for value in inputs] == ["100"]


def test_a_redelivery_whose_repeated_copy_names_another_source_is_rejected(store):
    w = writer(store)
    run = w.create_run("run", NOW)
    w.persist_observations(repeated_post_batch(), run_id=run)
    changed = repeated_post_batch()
    changed = replace(changed, posts=(changed.posts[0], replace(changed.posts[1], source_id="2")))

    with pytest.raises(ValueError, match="post_source_identity_mismatch"):
        w.persist_observations(changed, run_id=run)


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


def test_collection_progress_reads_progress_fields_not_the_runs_posts(store):
    from sqlalchemy import event
    w = writer(store)
    w.create_run("run", NOW)
    w.persist_observations(batch(), run_id="run")
    w.create_run("replay", NOW + timedelta(hours=1))
    with store.begin() as db:
        from app.infra.db.models.social_signals import SocialSignalRun
        original = db.get(SocialSignalRun, "run")
        observation = {**original.application_progress_json["observations"]["1"],
                       "committed_progress": "replayed-cursor"}
        replay = db.get(SocialSignalRun, "replay")
        replay.application_progress_json = {
            **original.application_progress_json, "observations": {"1": observation},
            "replay": {"saved_run_id": "run"},
        }
        replay.source_outcomes_json = {"1": {**original.source_outcomes_json["1"],
                                       "committed_progress": "replayed-cursor"}}
    engine = store.kw["bind"]
    fetched = []
    def record(conn, cursor, statement, parameters, context, executemany):
        fetched.extend(column[0] for column in cursor.description or ())
    event.listen(engine, "after_cursor_execute", record)
    try:
        progress = w.latest_collection_progress("1", "official")
    finally:
        event.remove(engine, "after_cursor_execute", record)

    assert (progress.cursor, progress.initial_complete, progress.last_read_at) == ("cursor", False, NOW)
    assert {"application_progress_json", "source_outcomes_json"} & set(fetched) == set()


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
    from app.infra.db.models.social_analysis import SocialExtractionWork
    from app.services.social_extraction_service import SocialExtractionService

    w = writer(store)
    w.create_run("run", NOW)
    saved = w.persist_observations(batch(), run_id="run")
    w.persist_observations(batch(2, post_id="200"), run_id="run")
    content_id = dict(w.read_run_inputs("run").content_ids)["100"]
    input_hash = SocialExtractionService.input_hash(saved.posts)
    with store.begin() as db:
        link(db, "run", saved_work(db, content_id, saved.posts[0], model="model-a"))
        failed = SocialExtractionWork(
            content_item_id=content_id, input_hash=input_hash, prompt_version="v1",
            schema_version="v1", selected_model="model-b", input_snapshot_json={},
            state="failed_terminal", created_at=NOW, updated_at=NOW,
        )
        db.add(failed)
        db.flush()
        link(db, "run", failed)

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


def flagged_claim(token, duplicate_of=()):
    return {"post_id": "100", "theme_key": "cooling", "raw_theme": "cooling", "company_token": token,
            "relationship": "supplier", "excerpt": "supplies cooling", "support": "supported",
            "duplicate_of_post_ids": list(duplicate_of), "evidence_id": None}


@pytest.mark.parametrize(("claims", "expected"), [
    ([flagged_claim("$AAA", ["99"])], (("official:100", "1"),)),
    ([flagged_claim("$AAA", ["99"]), flagged_claim("$AAA")], ()),
    ([flagged_claim("$BBB", ["99"])], ()),
    ([], ()),
])
def test_scoring_evidence_marks_a_post_whose_claims_about_a_ticker_are_all_flagged(store, claims, expected):
    from app.infra.db.repositories.social_refresh_support import SocialScoringEvidenceReader
    from app.models.stock_universe import StockUniverse

    w = writer(store)
    w.create_run("run", NOW)
    observed = batch()
    observed = replace(observed, posts=(replace(observed.posts[0], text="$AAA supplies cooling to $BBB"),))
    saved = w.persist_observations(observed, run_id="run")
    content_id = dict(w.read_run_inputs("run").content_ids)["100"]
    with store.begin() as db:
        for symbol in ("AAA", "BBB"):
            db.add(StockUniverse(symbol=symbol, name=f"{symbol} Corp", market="US", exchange="NASDAQ", is_active=True))
        link(db, "run", saved_work(db, content_id, saved.posts[0], claims))
    w.select_current_inputs("run")

    evidence = {item.candidate_key: item for item in SocialScoringEvidenceReader(store).read("run", NOW)}

    assert evidence["US:AAA"].duplicate_posts == expected


def test_a_flag_on_an_earlier_revision_does_not_exclude_the_edited_post(store):
    from app.infra.db.repositories.social_refresh_support import SocialScoringEvidenceReader
    from app.models.stock_universe import StockUniverse

    w = writer(store)
    w.create_run("old", NOW)
    original = w.persist_observations(batch(), run_id="old")
    later = NOW + timedelta(hours=1)
    w.clock = lambda: later + timedelta(minutes=1)
    w.create_run("new", later)
    edited = batch(age=1)
    edited = w.persist_observations(
        replace(edited, posts=(replace(edited.posts[0], text="$AAA supplies cooling, edited"),)), run_id="new")
    content_id = dict(w.read_run_inputs("new").content_ids)["100"]
    with store.begin() as db:
        db.add(StockUniverse(symbol="AAA", name="AAA Corp", market="US", exchange="NASDAQ", is_active=True))
        for saved, claims in ((original, [flagged_claim("$AAA", ["99"])]), (edited, [])):
            link(db, "new", saved_work(db, content_id, saved.posts[0], claims), at=later)
    w.select_current_inputs("new")

    evidence = {item.candidate_key: item for item in SocialScoringEvidenceReader(store).read("new", later)}

    assert [post.text for post in evidence["US:AAA"].posts] == ["$AAA supplies cooling, edited"]
    assert evidence["US:AAA"].duplicate_posts == ()


def test_each_list_revision_of_a_post_carries_its_own_flag(store):
    from app.infra.db.repositories.social_refresh_support import SocialScoringEvidenceReader
    from app.models.stock_universe import StockUniverse

    w = writer(store)
    w.create_run("run", NOW)
    first = w.persist_observations(batch(), run_id="run")
    edited = batch(2)
    edited = replace(edited, posts=(replace(edited.posts[0], text="$AAA supplies cooling, edited"),))
    second = w.persist_observations(edited, run_id="run")
    content_id = dict(w.read_run_inputs("run").content_ids)["100"]
    with store.begin() as db:
        db.add(StockUniverse(symbol="AAA", name="AAA Corp", market="US", exchange="NASDAQ", is_active=True))
        for saved, claims in ((first, [flagged_claim("$AAA", ["99"])]), (second, [])):
            link(db, "run", saved_work(db, content_id, saved.posts[0], claims))
    w.select_current_inputs("run")

    evidence = {item.candidate_key: item for item in SocialScoringEvidenceReader(store).read("run", NOW)}

    assert evidence["US:AAA"].duplicate_posts == (("official:100", "1"),)


def test_unavailable_revision_does_not_hide_another_revision_of_the_same_post(store):
    from app.infra.db.models.social_analysis import SocialExtractionWork
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
        failed = SocialExtractionWork(
            content_item_id=content_id, input_hash=SocialExtractionService.input_hash(first.posts),
            prompt_version="v1", schema_version="v1", selected_model="model", input_snapshot_json={},
            state="failed_terminal", created_at=NOW, updated_at=NOW,
        )
        db.add(failed)
        db.flush()
        link(db, "run", failed)
        link(db, "run", saved_work(db, content_id, second.posts[0], []))
    w.select_current_inputs("run")

    evidence = SocialScoringEvidenceReader(store).read("run", NOW)

    assert [post.text for item in evidence for post in item.posts] == ["$AAA supplies cooling, edited"]


def test_a_carry_in_result_that_no_longer_revalidates_is_closed_not_carried(store):
    w = writer(store)
    w.create_run("run", NOW)
    aged = batch()
    created = NOW - timedelta(days=20)
    aged = replace(aged, posts=(replace(aged.posts[0], created_at=created),),
                   outcome=replace(aged.outcome, observed_oldest_at=created, observed_newest_at=created))
    saved = w.persist_observations(aged, run_id="run")
    w.persist_observations(batch(2, post_id="200"), run_id="run")
    content_id = dict(w.read_run_inputs("run").content_ids)["100"]
    with store.begin() as db:
        work_id = saved_work(db, content_id, saved.posts[0], [flagged_claim("$AAA")]).id
    unquoted_excerpt(store, work_id)
    with store.begin() as db:
        from app.infra.db.models.social_analysis import SocialExtractionWork
        link(db, "run", db.get(SocialExtractionWork, work_id))

    manifest = w.select_current_inputs("run").current_manifest

    assert (manifest.carry_in_work_ids, manifest.coverage_reasons) == ((), ("outside_window_judgments_missing",))
    assert work_id in manifest.audit_work_ids
    assert work_state(store, work_id) == ("failed_terminal", "invalid_saved_result")


def test_a_current_result_that_no_longer_revalidates_is_a_coverage_gap_not_a_crash(store):
    from app.domain.social_signals.records import ReplayInput

    w = writer(store)
    w.create_run("run", NOW)
    first = w.persist_observations(batch(), run_id="run")
    second = w.persist_observations(batch(2, post_id="200"), run_id="run")
    content_ids = dict(w.read_run_inputs("run").content_ids)
    with store.begin() as db:
        bad = saved_work(db, content_ids["100"], first.posts[0])
        good = saved_work(db, content_ids["200"], second.posts[0])
        link(db, "run", bad)
        link(db, "run", good)
        bad_id, bad_hash, good_hash = bad.id, bad.input_hash, good.input_hash
    unquoted_excerpt(store, bad_id)

    manifest = w.select_current_inputs("run").current_manifest

    assert w.current_inputs_ready("run") is True
    assert manifest.required_inputs == (ReplayInput(content_ids["200"], good_hash, "current"),)
    assert manifest.unavailable_inputs == (ReplayInput(content_ids["100"], bad_hash, "unavailable"),)
    assert manifest.coverage_reasons == ("analysis_unavailable",)
    assert work_state(store, bad_id) == ("failed_terminal", "invalid_saved_result")


def test_readiness_is_false_for_a_selected_result_that_stopped_revalidating_until_reselected(store):
    w = writer(store)
    w.create_run("run", NOW)
    first = w.persist_observations(batch(), run_id="run")
    second = w.persist_observations(batch(2, post_id="200"), run_id="run")
    content_ids = dict(w.read_run_inputs("run").content_ids)
    with store.begin() as db:
        bad = saved_work(db, content_ids["100"], first.posts[0])
        link(db, "run", bad)
        link(db, "run", saved_work(db, content_ids["200"], second.posts[0]))
        bad_id = bad.id
    w.select_current_inputs("run")
    assert w.current_inputs_ready("run") is True
    unquoted_excerpt(store, bad_id)

    assert w.current_inputs_ready("run") is False

    manifest = w.select_current_inputs("run").current_manifest
    assert (w.current_inputs_ready("run"), manifest.coverage_reasons) == (True, ("analysis_unavailable",))


def test_a_replay_closes_a_saved_result_that_no_longer_revalidates_instead_of_carrying_it(store):
    w = writer(store)
    w.create_run("old", NOW)
    first = w.persist_observations(batch(), run_id="old")
    second = w.persist_observations(batch(2, post_id="200"), run_id="old")
    content_ids = dict(w.read_run_inputs("old").content_ids)
    with store.begin() as db:
        bad = saved_work(db, content_ids["100"], first.posts[0])
        good = saved_work(db, content_ids["200"], second.posts[0])
        link(db, "old", bad)
        link(db, "old", good)
        bad_id, good_id = bad.id, good.id
    unquoted_excerpt(store, bad_id)

    w.create_replay_run("new", "old", NOW + timedelta(hours=1))

    replayed = w.read_run_inputs("new")
    assert replayed.work_ids == (good_id,)
    assert sorted(v.content_item_id for v in replayed.replay_manifest.required_inputs) == sorted(content_ids.values())
    assert work_state(store, bad_id) == ("failed_terminal", "invalid_saved_result")


def test_publish_refuses_a_pinned_result_that_stopped_revalidating_with_a_stable_reason(store):
    from app.infra.db.models.social_signals import SocialSignalRun, SocialSourceRegistry

    w = writer(store)
    w.create_run("run", NOW)
    saved = w.persist_observations(batch(), run_id="run")
    w.persist_observations(empty_batch(2), run_id="run")
    content_id = dict(w.read_run_inputs("run").content_ids)["100"]
    with store.begin() as db:
        work = saved_work(db, content_id, saved.posts[0])
        link(db, "run", work)
        work_id = work.id
    w.prepare_run("run", (), NOW)
    unquoted_excerpt(store, work_id)
    with store() as db:
        version = db.get(SocialSourceRegistry, 1).version

    with pytest.raises(ValueError, match="^pinned_input_invalid$"):
        w.publish("run", version)

    with store() as db:
        run = db.get(SocialSignalRun, "run")
        assert (run.status, run.published_at, run.application_progress_json["failure"]) == (
            "failed", None, {"reason_code": "publication_failed"})
    assert w.read_run_inputs("run").work_ids == (work_id,)


def test_a_failed_close_does_not_replace_the_publication_error(store, monkeypatch):
    w = writer(store)

    def fails(message):
        def raise_(*_args, **_kwargs):
            raise RuntimeError(message)
        return raise_

    monkeypatch.setattr(w, "_publish", fails("publication broke"))
    monkeypatch.setattr(w, "_close_runs", fails("database down"))

    with pytest.raises(RuntimeError, match="^publication broke$"):
        w.publish("run", 1)
