"""Atomic pointer/projection publication from isolated, saved input records."""
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest
from sqlalchemy import update, delete

from tests.unit.repositories.test_social_signal_writer import store, batch, writer, NOW
from app.domain.social_signals.records import SocialSnapshotRecord
from app.infra.db.models.social_signals import SocialSignalRun, SocialSignalRunPointer, SocialSignalSnapshot
from tests.integration.test_social_theme_projection import social_fixture


def row(run, candidate="AAA", score="80", market="US", state="watch"):
    return SocialSnapshotRecord(run, candidate, candidate if market else None, market, state,
        Decimal(score), None, None, (), ("post_cap",), NOW-timedelta(days=1), candidate, candidate)


def empty_batch(source):
    b = batch(source)
    return replace(b, posts=(), outcome=replace(b.outcome, received_count=0,
        observed_oldest_at=None, observed_newest_at=None))


def collected(w, run="run", now=NOW):
    w.create_run(run, now)
    for source in (1, 2):
        w.persist_observations(empty_batch(source), run_id=run)
    return run


def test_no_pointer_is_typed_unavailable(store):
    from app.infra.db.repositories.published_social_signal_reader import PublishedSocialSignalReader, SocialPublicationUnavailable
    with pytest.raises(SocialPublicationUnavailable) as error:
        PublishedSocialSignalReader(store).queue("US", 14, "all", "blended", 1, 20)
    assert error.value.reason == "no_published_run"


def test_limited_history_can_publish_and_reader_freezes_one_run(store):
    from app.infra.db.repositories.published_social_signal_reader import PublishedSocialSignalReader
    w = writer(store)
    run = collected(w)
    w.prepare_run(run, (row(run), row(run, "GLOBAL", "99", None, "unresolved")), NOW)
    with store() as db:
        version = db.get(SocialSignalRun, run).registry_version
    result = w.publish(run, version)
    assert result.published and ("1", "limited") in result.coverage_summary
    page = PublishedSocialSignalReader(store).queue("US", 14, "all", "blended", 1, 20)
    assert {item.run_id for item in page.items} == {run}
    assert page.total == 2
    assert page.items[0].candidate_key == "AAA"
    assert w.publish(run, version).published
    with store() as db:
        assert db.query(SocialSignalSnapshot).count() == 2


@pytest.mark.parametrize("problem", ["missing_source", "failed_read", "unfinished", "no_work_link"])
def test_publication_requires_every_pinned_read_and_exact_finished_inputs(store, problem):
    w = writer(store)
    w.create_run("run", NOW)
    w.persist_observations(empty_batch(1), run_id="run")
    if problem == "failed_read":
        b = empty_batch(2)
        b = replace(b, outcome=replace(b.outcome, read_status="failed", error_code="provider_error", proposed_progress=None))
        w.persist_observations(b, run_id="run")
    elif problem in {"unfinished", "no_work_link"}:
        w.persist_observations(batch(2), run_id="run")
        if problem == "unfinished":
            from app.models.theme import ContentItem
            from app.use_cases.social_signals.process_backlog import ProcessSocialBacklog
            with store() as db:
                item = db.query(ContentItem).one().id
            ProcessSocialBacklog(store).enqueue(item, batch(2).posts[0], selected_model="synthetic/model", now=NOW, run_id="run")
    with pytest.raises(ValueError, match="participation|incomplete|pinned"):
        w.prepare_run("run", (), NOW)
    with store() as db:
        assert db.query(SocialSignalRunPointer).count() == 0


def test_newer_generation_fences_delayed_run_and_published_payload_is_immutable(store):
    w = writer(store)
    for run, now in (("old", NOW), ("new", NOW+timedelta(minutes=1))):
        collected(w, run, now)
        w.prepare_run(run, (row(run),), now)
    with store() as db:
        version = db.get(SocialSignalRun, "new").registry_version
    w.publish("new", version)
    with pytest.raises(ValueError, match="superseded"):
        w.publish("old", version)
    for mutation in ("run", "snapshot", "bulk_update", "bulk_delete"):
        with store() as db:
            with pytest.raises(ValueError, match="immutable"):
                if mutation == "run":
                    db.get(SocialSignalRun, "new").created_at = NOW
                elif mutation == "snapshot":
                    db.query(SocialSignalSnapshot).filter_by(run_id="new").one().social_score = 1
                elif mutation == "bulk_update":
                    db.execute(update(SocialSignalSnapshot).values(social_score=1))
                else:
                    db.execute(delete(SocialSignalRun).where(SocialSignalRun.id == "new"))
                db.commit()


def test_mode_change_rejects_prepared_run_and_validation_never_publishes(store):
    from app.infra.db.models.social_signals import SocialSourceRegistry
    w = writer(store)
    collected(w)
    w.prepare_run("run", (row("run"),), NOW)
    with store.begin() as db:
        registry = db.get(SocialSourceRegistry, 1)
        version = registry.version
        registry.mode, registry.version = "validation", version+1
    with pytest.raises(ValueError, match="version|live"):
        w.publish("run", version)
    collected(w, "validation")
    w.prepare_run("validation", (row("validation"),), NOW)
    assert not w.publish("validation", version+1).published
    with store() as db:
        assert db.query(SocialSignalRunPointer).count() == 0


def test_prepared_run_cannot_be_retimed_after_backlog(store):
    w = writer(store)
    collected(w)
    with pytest.raises(ValueError, match="generation"):
        w.prepare_run("run", (), NOW+timedelta(days=15))


def test_staged_basket_is_adjudicated_without_writes_and_matches_live(social_fixture):
    from app.models.theme import ThemeCluster
    f = social_fixture
    first = f.save(("AAA", "BBB", "CCC"), age=3)
    second = f.save(("AAA", "BBB", "CCC"), age=2)
    projection = f.prepare([first, second])
    prepared = f.service.prepare_application(projection)
    basket = prepared.read("cooling", "US")
    assert tuple(m.canonical_symbol for m in basket.membership) == ("AAA", "BBB", "CCC")
    assert f.db.query(ThemeCluster).count() == 0
    f.service.apply_live(projection, projection.registry_version, prepared=prepared)
    f.db.commit()
    from app.services.social_theme_market_service import LiveAcceptedBasketReader
    assert LiveAcceptedBasketReader(f.db).read("cooling", "US") == basket


def test_raw_proposal_and_copied_evidence_never_enter_staged_accepted_basket(social_fixture):
    f = social_fixture
    projection = f.prepare([f.save(key="copied"), f.save(key="copied")])
    assert f.service.prepare_application(projection).read("cooling", "US").membership == ()


def test_admin_rejection_invalidates_prepared_generation(social_fixture):
    f = social_fixture
    f.apply(f.prepare([f.save(), f.save()]))
    association = f.associations()[0]
    projection = f.prepare([f.save()])
    prepared = f.service.prepare_application(projection)
    f.service.decide(association.id, "rejected", "reviewed business mismatch", "admin", association.version)
    f.db.commit()
    from app.infra.db.models.social_signals import SocialSourceRegistry
    assert f.db.get(SocialSourceRegistry, 1).version == projection.registry_version + 1
    f.db.rollback()
    with f.db.begin(), pytest.raises(ValueError, match="version"):
        f.service.apply_live(projection, projection.registry_version, prepared=prepared)


def save_success(store, w, run):
    from dataclasses import asdict
    from app.domain.social_signals.records import ExtractionClaim, ExtractionPostJudgment, ExtractionResult
    from app.infra.db.models.social_analysis import SocialExtractionWork
    from app.models.theme import ContentItem
    from app.use_cases.social_signals.process_backlog import ProcessSocialBacklog
    b = batch()
    w.persist_observations(b, run_id=run)
    w.persist_observations(empty_batch(2), run_id=run)
    with store() as db:
        item_id = db.query(ContentItem).one().id
    work_id = ProcessSocialBacklog(store).enqueue(item_id, b.posts[0], selected_model="synthetic/model", now=NOW, run_id=run)
    with store.begin() as db:
        work = db.get(SocialExtractionWork, work_id)
        result = ExtractionResult(work.input_hash, "synthetic", "model", work.prompt_version, work.schema_version,
            (ExtractionClaim("100", "cooling", "Cooling", "AAA", "supplies", "$AAA supplies cooling", "supported", ()),),
            10, 10, (ExtractionPostJudgment("100", True, "claim-100"),))
        work.state, work.actual_provider, work.actual_model, work.result_json = "succeeded", "synthetic", "model", asdict(result)
    return work_id


def test_projection_and_pointer_roll_back_together_after_apply(store, monkeypatch):
    from app.models.theme import ThemeCluster
    from app.services.social_theme_projection_service import SocialThemeProjectionService
    w = writer(store)
    collected(w, "prior", NOW-timedelta(minutes=1))
    w.prepare_run("prior", (), NOW-timedelta(minutes=1))
    with store() as db:
        version = db.get(SocialSignalRun, "prior").registry_version
    w.publish("prior", version)
    w.create_run("new", NOW)
    save_success(store, w, "new")
    w.prepare_run("new", (row("new"),), NOW)
    original = SocialThemeProjectionService.apply_live
    def crash(service, *args, **kwargs):
        original(service, *args, **kwargs)
        assert service.db.query(ThemeCluster).count() == 1
        raise RuntimeError("after projection")
    monkeypatch.setattr(SocialThemeProjectionService, "apply_live", crash)
    with pytest.raises(RuntimeError, match="after projection"):
        w.publish("new", version)
    with store() as db:
        assert db.query(ThemeCluster).count() == 0
        assert db.get(SocialSignalRunPointer, "latest_published").run_id == "prior"
    monkeypatch.setattr(SocialThemeProjectionService, "apply_live", original)
    assert w.publish("new", version).published


def test_frozen_replay_inputs_keep_metrics_and_reuse_work_after_new_observation(store):
    w = writer(store)
    w.create_run("run", NOW)
    work_id = save_success(store, w, "run")
    saved = w.read_run_inputs("run")
    w.persist_observations(batch(likes=900, age=1))
    assert saved.batches[0].posts[0].likes == 10
    assert saved.work_ids == (work_id,)
    assert w.read_run_inputs("run") == saved


def test_stale_measured_basket_is_rejected_before_publication(social_fixture):
    from app.services.social_theme_market_service import LiveAcceptedBasketReader
    from app.models.theme import ThemeConstituent
    f = social_fixture
    f.apply(f.prepare([f.save(), f.save()]))
    projection = f.prepare([f.save()])
    prepared = f.service.prepare_application(projection)
    theme_id = f.associations()[0].theme_cluster_id
    f.db.add(ThemeConstituent(theme_cluster_id=theme_id, symbol="BBB", is_active=True))
    f.db.commit()
    with f.db.begin(), pytest.raises(ValueError, match="version"):
        f.service.apply_live(projection, projection.registry_version, prepared=prepared)


def test_measurement_cannot_claim_unadjudicated_basket(store):
    from app.domain.social_signals.records import ThemeMarketEvidence
    w = writer(store)
    w.create_run("run", NOW)
    save_success(store, w, "run")
    evidence = ThemeMarketEvidence("cooling", "US", NOW.date(), "SPY", "untrusted-version", 3, (), (), ())
    with pytest.raises(ValueError, match="basket"):
        w.prepare_run("run", (row("run"),), NOW, theme_evidence=(evidence,))


@pytest.mark.parametrize("history", ["limited", "warming_up"])
def test_short_unknown_history_and_elapsed_fourteen_days_preserve_gaps(store, history):
    w = writer(store)
    gaps = ((NOW-timedelta(days=10), NOW-timedelta(days=9)),)
    for run, at in (("startup", NOW), ("later", NOW+timedelta(days=14))):
        w.create_run(run, at)
        for source in (1, 2):
            b = empty_batch(source)
            b = replace(b, outcome=replace(b.outcome, history_status=history,
                coverage_reason_codes=("unknown_exhaustion",), known_gap_intervals=gaps))
            w.persist_observations(b, run_id=run)
        w.prepare_run(run, (), at)
        saved = w.read_run_inputs(run)
        assert all(b.outcome.history_status == history for b in saved.batches)
        assert all(b.outcome.known_gap_intervals == gaps for b in saved.batches)
        assert all(b.outcome.coverage_reason_codes == ("unknown_exhaustion",) for b in saved.batches)


def test_pending_disabled_archived_sources_not_pinned_and_busy_new_enabled_must_participate(store):
    from app.infra.db.models.social_signals import SocialSourceConfiguration, SocialSourceRegistry
    from app.models.theme import ContentSource
    with store.begin() as db:
        for number, state in ((3, "pending"), (4, "disabled"), (5, "archived"), (6, "enabled")):
            db.add(ContentSource(id=number, source_type="twitter", name=f"synthetic-{number}"))
            db.add(SocialSourceConfiguration(content_source_id=number, x_list_id=str(number), lifecycle_state=state,
                provenance="admin", archived_at=NOW if state == "archived" else None))
        db.get(SocialSourceRegistry, 1).version += 1
    w = writer(store)
    collected(w)
    with store() as db:
        assert set(db.get(SocialSignalRun, "run").application_progress_json["sources"]) == {"1", "2", "6"}
    with pytest.raises(ValueError, match="participation"):
        w.prepare_run("run", (), NOW)


def test_published_run_rejects_appended_snapshot(store):
    w = writer(store)
    collected(w)
    w.prepare_run("run", (row("run"),), NOW)
    with store() as db:
        version = db.get(SocialSignalRun, "run").registry_version
    w.publish("run", version)
    with store() as db:
        original = db.query(SocialSignalSnapshot).one()
        values = {c.name: getattr(original, c.name) for c in SocialSignalSnapshot.__table__.columns if c.name != "id"}
        values["candidate_key"] = "injected"
        db.add(SocialSignalSnapshot(**values))
        with pytest.raises(ValueError, match="immutable"):
            db.commit()


@pytest.mark.parametrize("mode,expected", [("pure_social", ["z", "b", "a", "scored", "missing"]),
    ("blended", ["scored", "z", "b", "a", "missing"])])
def test_reader_uses_shared_rank_mode_ties_and_nulls(store, mode, expected):
    from app.domain.social_signals.scoring import rank_snapshots
    from app.infra.db.repositories.published_social_signal_reader import PublishedSocialSignalReader
    w = writer(store)
    collected(w)
    rows = (replace(row("run", "a"), canonical_symbol="ZZZ", latest_mention=NOW-timedelta(hours=2)),
        replace(row("run", "b"), canonical_symbol="BBB", latest_mention=NOW-timedelta(hours=1)),
        replace(row("run", "z"), canonical_symbol="AAA", latest_mention=NOW-timedelta(hours=1)),
        replace(row("run", "scored", "10"), confirmation_score=Decimal(90), queue_score=Decimal(42)),
        replace(row("run", "missing"), social_score=None, latest_mention=None))
    w.prepare_run("run", rows, NOW)
    with store() as db:
        version = db.get(SocialSignalRun, "run").registry_version
    w.publish("run", version)
    actual = PublishedSocialSignalReader(store).queue("US", 14, "all", mode, 1, 20)
    assert [r.candidate_key for r in actual.items] == expected
    assert actual.items == rank_snapshots(rows, mode)


def test_reader_hydrates_only_page_and_unranked_order_ignores_scores(store):
    from sqlalchemy import event
    from app.infra.db.repositories.published_social_signal_reader import PublishedSocialSignalReader
    w = writer(store)
    collected(w)
    rows = tuple(row("run", f"ranked-{i:03}") for i in range(35)) + (
        row("run", "context-z", "100", state="context"), row("run", "context-a", "1", state="context"),
        row("run", "unresolved-a", "100", None, "unresolved"))
    w.prepare_run("run", rows, NOW)
    with store() as db:
        version = db.get(SocialSignalRun, "run").registry_version
    w.publish("run", version)
    hydrated = []
    def loaded(row, context):
        hydrated.append(row.id)
    event.listen(SocialSignalSnapshot, "load", loaded)
    try:
        page = PublishedSocialSignalReader(store).queue("US", 14, "all", "blended", 8, 5)
    finally:
        event.remove(SocialSignalSnapshot, "load", loaded)
    assert [r.candidate_key for r in page.items] == ["context-a", "context-z", "unresolved-a"]
    assert page.total == 38
    assert len(hydrated) == 3


def test_snapshot_references_run_evidence_without_copying_corpus(store):
    w = writer(store)
    w.create_run("run", NOW)
    save_success(store, w, "run")
    w.prepare_run("run", (row("run"), row("run", "BBB")), NOW)
    with store() as db:
        run = db.get(SocialSignalRun, "run")
        assert run.application_progress_json["observations"]["1"]["inputs"][0]["post"]["text"] == "$AAA supplies cooling"
        for snapshot in db.query(SocialSignalSnapshot):
            assert "run_inputs" not in snapshot.explanation_json
            assert "theme_evidence" not in snapshot.explanation_json
            assert "work_ids" not in snapshot.explanation_json
            assert snapshot.explanation_json["evidence_run_id"] == "run"


@pytest.mark.parametrize("change", ["theme_cluster_id", "is_active", "confidence", "source", "evidence_count"])
def test_qualified_alias_change_fences_prepared_application(social_fixture, change):
    from app.models.theme import ThemeAlias, ThemeCluster
    f = social_fixture
    left = ThemeCluster(name="Hardware", display_name="Hardware", canonical_key="hardware", pipeline="technical", is_active=True)
    right = ThemeCluster(name="Equipment", display_name="Equipment", canonical_key="equipment", pipeline="technical", is_active=True)
    f.db.add_all([left, right])
    f.db.flush()
    alias = ThemeAlias(theme_cluster_id=left.id, pipeline="technical", alias_text="Cooling", alias_key="cooling",
        source="manual", confidence=1, evidence_count=5, is_active=True)
    f.db.add(alias)
    f.db.commit()
    projection = f.prepare([f.save()])
    prepared = f.service.prepare_application(projection)
    setattr(alias, change, {"theme_cluster_id": right.id, "is_active": False, "confidence": 0,
        "source": "unknown", "evidence_count": 1}[change])
    f.db.commit()
    with pytest.raises(ValueError, match="version"):
        with f.db.begin():
            f.service.apply_live(projection, projection.registry_version, prepared=prepared)


@pytest.mark.parametrize("operation", ["enqueue", "insert", "update", "delete", "bulk_insert", "bulk_update", "bulk_delete"])
@pytest.mark.parametrize("status", ["staged", "published"])
def test_terminal_work_links_cannot_change(store, operation, status):
    from sqlalchemy import insert
    from app.infra.db.models.social_analysis import SocialExtractionWork, SocialRunWork
    from app.use_cases.social_signals.process_backlog import ProcessSocialBacklog
    w = writer(store)
    w.create_run("run", NOW)
    work_id = save_success(store, w, "run")
    with store() as db:
        item_id = db.get(SocialExtractionWork, work_id).content_item_id
    spare_id = ProcessSocialBacklog(store).enqueue(item_id, batch().posts[0],
        selected_model="synthetic/other", now=NOW)
    w.prepare_run("run", (), NOW)
    if status == "published":
        with store() as db:
            version = db.get(SocialSignalRun, "run").registry_version
        w.publish("run", version)
    before = w.read_run_inputs("run")
    with store() as db:
        work = db.get(SocialExtractionWork, work_id)
        item_id, input_hash = work.content_item_id, work.input_hash
        count = db.query(SocialExtractionWork).count()
    with pytest.raises(ValueError, match="immutable|terminal"):
        if operation == "enqueue":
            ProcessSocialBacklog(store).enqueue(item_id, replace(batch().posts[0], text="new content"),
                selected_model="synthetic/model", now=NOW, run_id="run")
        else:
            with store.begin() as db:
                if operation == "insert":
                    db.add(SocialRunWork(run_id="run", work_id=spare_id, input_hash=input_hash, included_at=NOW))
                elif operation == "update":
                    db.get(SocialRunWork, ("run", work_id)).included_at = NOW+timedelta(days=1)
                elif operation == "delete":
                    db.delete(db.get(SocialRunWork, ("run", work_id)))
                elif operation == "bulk_insert":
                    db.execute(insert(SocialRunWork).values(run_id="run", work_id=spare_id, input_hash=input_hash, included_at=NOW))
                elif operation == "bulk_update":
                    db.execute(update(SocialRunWork).where(SocialRunWork.run_id == "run").values(input_hash="changed"))
                else:
                    db.execute(delete(SocialRunWork).where(SocialRunWork.run_id == "run"))
    assert w.read_run_inputs("run") == before
    with store() as db:
        assert db.query(SocialExtractionWork).count() == count


def test_terminal_replay_uses_saved_manifest_even_if_links_removed_below_orm(store):
    from app.infra.db.models.social_analysis import SocialRunWork
    w = writer(store)
    w.create_run("run", NOW)
    work_id = save_success(store, w, "run")
    w.prepare_run("run", (), NOW)
    # A privileged Core write bypasses ORM policy. Historical replay must still
    # use its immutable saved manifest, rather than reinterpret mutable links.
    with store.kw["bind"].begin() as connection:
        connection.execute(SocialRunWork.__table__.delete().where(SocialRunWork.run_id == "run"))
    assert w.read_run_inputs("run").work_ids == (work_id,)
