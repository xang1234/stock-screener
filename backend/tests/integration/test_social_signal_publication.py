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
from tests.unit.services.test_social_theme_market_service import basket


@pytest.fixture(autouse=True)
def local_calendar_cache(monkeypatch):
    from app.services.market_calendar_adapters import RawMarketCalendarAdapter
    monkeypatch.setattr(RawMarketCalendarAdapter, "_session_range_cache_client", lambda self: None)


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
        b = empty_batch(source)
        w.persist_observations(replace(b, request=replace(b.request, observed_at=now,
            target_published_after=now-timedelta(days=14))), run_id=run)
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
            from app.services.social_signal_backlog_service import ProcessSocialBacklog
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
    from app.services.social_signal_backlog_service import ProcessSocialBacklog
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
    w.clock = lambda: NOW + timedelta(hours=1)
    w.persist_observations(batch(likes=900, age=1))
    assert saved.batches[0].posts[0].likes == 10
    assert saved.work_ids == (work_id,)
    assert w.read_run_inputs("run") == saved


def test_current_replay_preserves_terminal_history_and_collection_freshness(store, monkeypatch):
    from app.infra.db.models.social_signals import SocialSourceConfiguration
    w = writer(store)
    w.create_run("old", NOW)
    work_id = save_success(store, w, "old")
    w.prepare_run("old", (), NOW)
    old = w.read_run_inputs("old")
    with store() as db:
        timestamps = tuple(s.last_successful_collection_at for s in db.query(SocialSourceConfiguration).order_by(SocialSourceConfiguration.content_source_id))
    def no_collection(*args, **kwargs):
        raise AssertionError("replay must not collect observations")
    monkeypatch.setattr(w, "persist_observations", no_collection)
    current = NOW + timedelta(hours=8)
    w.create_replay_run("current", "old", current)
    saved = w.read_run_inputs("current")
    assert saved.as_of == current
    assert saved.batches == old.batches
    assert saved.work_ids == (work_id,)
    assert saved.replay_manifest.saved_run_id == "old"
    assert saved.replay_manifest.required_inputs[0].disposition == "current"
    assert w.read_run_inputs("old") == old
    w.prepare_run("current", (), current)
    with store() as db:
        assert tuple(s.last_successful_collection_at for s in db.query(SocialSourceConfiguration).order_by(SocialSourceConfiguration.content_source_id)) == timestamps


@pytest.mark.parametrize("change", ["provider", "source_version", "added", "disabled"])
def test_replay_keeps_historical_batches_but_pins_complete_current_sources(store, change):
    from app.infra.db.models.social_signals import SocialSourceRegistry, SocialSourceConfiguration
    from app.models.theme import ContentSource
    w = writer(store)
    collected(w, "old")
    old = w.read_run_inputs("old")
    with store.begin() as db:
        registry = db.get(SocialSourceRegistry, 1)
        registry.version += 1
        if change == "provider":
            registry.provider = "xui"
        elif change == "source_version":
            db.get(SocialSourceConfiguration, 1).version += 1
        else:
            source = ContentSource(name="Extra", source_type="twitter", url="https://x.com/i/lists/3")
            db.add(source)
            db.flush()
            db.add(SocialSourceConfiguration(content_source_id=source.id, x_list_id="3", lifecycle_state="enabled", provenance="admin", version=1))
            if change == "disabled":
                db.get(SocialSourceConfiguration, 1).lifecycle_state = "disabled"
    w.create_replay_run("new", "old", NOW + timedelta(hours=1))
    new = w.read_run_inputs("new")
    assert new.batches == old.batches
    assert new.replay_manifest.missing_sources
    with pytest.raises(ValueError, match="participation_incomplete"):
        w.prepare_run("new", (), new.as_of)
    assert w.read_run_inputs("old") == old


@pytest.mark.parametrize("succeeded", [False, True])
def test_replay_aged_work_is_audit_or_saved_carry_in_never_fake_success(store, succeeded):
    from app.infra.db.models.social_analysis import SocialExtractionWork
    from app.models.theme import ContentItem
    from app.services.social_signal_backlog_service import ProcessSocialBacklog
    w = writer(store)
    w.create_run("old", NOW)
    if succeeded:
        work_id = save_success(store, w, "old")
    else:
        w.persist_observations(batch(), run_id="old")
        w.persist_observations(empty_batch(2), run_id="old")
        with store() as db:
            content_id = db.query(ContentItem).one().id
        work_id = ProcessSocialBacklog(store).enqueue(content_id, batch().posts[0], selected_model="synthetic/model", now=NOW, run_id="old")
    old = w.read_run_inputs("old")
    w.create_replay_run("new", "old", NOW + timedelta(days=14))
    new = w.read_run_inputs("new")
    assert new.batches == old.batches
    assert new.replay_manifest.required_inputs == ()
    assert new.replay_manifest.historical_work_ids == (work_id,)
    assert new.replay_manifest.carry_in_work_ids == ((work_id,) if succeeded else ())
    w.prepare_run("new", (), new.as_of)
    with store() as db:
        assert db.get(SocialExtractionWork, work_id).state == ("succeeded" if succeeded else "pending")
    assert w.read_run_inputs("old") == old


def publication_context(store, now=NOW):
    from app.services.social_confirmation_reader import SocialConfirmationReader
    from app.domain.social_signals import records
    assert hasattr(records, "SocialPublicationContext"), "typed publication context missing"
    with store() as db:
        market = SocialConfirmationReader(db).read_market("US", ("AAA",), now, theme_keys=())
    return records.SocialPublicationContext((market,))


def candidate_context(context, candidate="AAA"):
    from app.domain.social_signals import records as r
    from app.domain.social_signals.states import classify_signal_state
    from app.domain.social_signals.scoring import score_confirmation
    assert hasattr(r, "CandidatePublicationContext"), "typed candidate context missing"
    market = context.market_batches[0]
    fact = dict(market.facts)[candidate]
    def fresh(value):
        if value.required_session is None or value.actual_session is None or value.reason in {"calendar_unavailable", "listing_mic_unknown"}:
            return None
        return value.fresh
    state = r.SignalStateInput(True, True, "US", feature_fresh=fresh(fact.feature_freshness),
        market_fresh=fresh(fact.market_freshness), liquidity_eligible=fact.liquidity_eligible,
        setup_ready=fact.setup_ready, setup_score=fact.setup_score, market_exposure=fact.market_exposure)
    decision = classify_signal_state(state)
    social = r.SocialScoreResult(Decimal("80"), (), decision, candidate_key=candidate,
        canonical_symbol=candidate, market="US", acceleration=Decimal("1.5"),
        post_memberships=(("100", ("1", "2")),), exclusions=(("99", "outside_window"),))
    return r.CandidatePublicationContext(candidate, 14, state,
        decision, social, score_confirmation(next(v for v in market.inputs if v.candidate_key == f"US:{candidate}")))


def test_market_publication_context_roundtrip_is_frozen_and_run_scoped(store):
    from dataclasses import FrozenInstanceError
    from tests.unit.services.test_social_confirmation_reader import seed_batch
    with store.begin() as db:
        seed_batch(db)
    w = writer(store)
    collected(w)
    context = publication_context(store)
    context = replace(context, candidates=(candidate_context(context),))
    w.prepare_run("run", (row("run"),), NOW, context=context)
    saved = w.read_publication_context("run")
    assert saved.market_batches == context.market_batches
    assert saved.candidates == context.candidates
    assert saved.candidates[0].social_result.acceleration == Decimal("1.5")
    assert saved.source_progress == (("1", "cursor"), ("2", "cursor"))
    assert saved.market_batches[0].facts[0][1].avg_dollar_volume == Decimal("100000000")
    with pytest.raises(FrozenInstanceError):
        saved.market_batches[0].market_context.exposure_id = 99
    with pytest.raises(TypeError, match="immutable"):
        replace(saved, market_batches=list(saved.market_batches))
    with store() as db:
        run = db.get(SocialSignalRun, "run")
        assert run.feature_run_ids_json == {"US": context.market_batches[0].pinned_run.run_id}
        assert run.exposure_dates_json == {"US": "2026-07-02"}
        assert "market_batches" not in db.query(SocialSignalSnapshot).one().explanation_json


@pytest.mark.parametrize("change", ["candidate", "state", "social", "confirmation", "window", "missing"])
def test_candidate_publication_context_rejects_snapshot_mismatch(store, change):
    w = writer(store)
    collected(w)
    context = publication_context(store)
    context = replace(context, candidates=(candidate_context(context),))
    record = row("run")
    if change == "candidate": record = replace(record, candidate_key="other")
    elif change == "state": record = replace(record, candidate_state="risk_off")
    elif change == "social": record = replace(record, social_score=Decimal("10"))
    elif change == "confirmation": record = replace(record, confirmation_score=Decimal("50"), queue_score=Decimal("68"))
    elif change == "window": record = replace(record, window_days=7)
    else: context = replace(context, candidates=())
    with pytest.raises(ValueError, match="candidate_context_mismatch"):
        w.prepare_run("run", (record,), NOW, context=context)


@pytest.mark.parametrize("change", ["pointer", "exposure_update", "exposure_insert", "feature", "group", "grace"])
@pytest.mark.parametrize("stage", ["prepare", "publish"])
def test_changed_market_context_requires_new_evaluation(store, change, stage):
    from tests.unit.services.test_social_confirmation_reader import seed_batch, utc
    from app.infra.db.models.feature_store import FeatureRun, FeatureRunPointer, StockFeatureDaily
    from app.models.market_exposure import MarketExposure
    from app.models.industry import IBDGroupRank
    from app.infra.db.repositories.social_signal_writer import SocialSignalWriter
    at = utc("2026-07-02T23:00:00")
    clock = [at]
    with store.begin() as db:
        feature_run, _, exposure = seed_batch(db)
        feature_id, exposure_id = feature_run.id, exposure.id
        if change == "exposure_insert":
            db.delete(exposure)
    w = SocialSignalWriter(store, clock=lambda: clock[0])
    w.create_run("run", at)
    for source in (1, 2):
        b = empty_batch(source)
        w.persist_observations(replace(b, request=replace(b.request, observed_at=at,
            target_published_after=at-timedelta(days=14))), run_id="run")
    context = publication_context(store, at)
    if stage == "publish":
        w.prepare_run("run", (), at, context=context)
    with store.begin() as db:
        if change == "pointer":
            other = FeatureRun(as_of_date=at.date(), run_type="daily_snapshot", status="published",
                config_json={"universe": {"market": "US"}}, completed_at=at, published_at=at)
            db.add(other)
            db.flush()
            db.get(FeatureRunPointer, "latest_published_market:US").run_id = other.id
        elif change == "exposure_update":
            db.get(MarketExposure, exposure_id).exposure_score = 20
        elif change == "exposure_insert":
            db.add(MarketExposure(market="US", date=at.date(), exposure_score=60, stance="uptrend", benchmark_symbol="SPY",
                created_at=at+timedelta(minutes=1), updated_at=at+timedelta(minutes=1)))
            clock[0] = at + timedelta(minutes=2)
        elif change == "feature":
            feature = db.get(StockFeatureDaily, (feature_id, "AAA"))
            feature.details_json = {**feature.details_json, "setup_engine": {"setup_score": 10, "setup_ready": False}}
        elif change == "group":
            db.query(IBDGroupRank).filter_by(market="US", date=at.date(), industry_group="Chips").one().rank = 4
        else:
            clock[0] = utc("2026-07-06T22:00:00")
    with pytest.raises(ValueError, match="market_context_changed"):
        if stage == "prepare":
            w.prepare_run("run", (), at, context=context)
        else:
            w.publish("run", 1)
    with store() as db:
        assert db.query(SocialSignalRunPointer).count() == 0
    if stage == "publish":
        assert w.read_publication_context("run").market_batches == context.market_batches


def test_publication_validator_never_acquires_shared_calendar_cache(store, monkeypatch):
    from app.services.market_calendar_adapters import RawMarketCalendarAdapter
    w = writer(store)
    collected(w)
    context = publication_context(store)
    def forbidden(*args, **kwargs):
        raise AssertionError("publication validation must use local calendar only")
    monkeypatch.setattr(RawMarketCalendarAdapter, "_read_session_range_cache", forbidden)
    monkeypatch.setattr(RawMarketCalendarAdapter, "_write_session_range_cache", forbidden)
    monkeypatch.setattr(RawMarketCalendarAdapter, "_session_range_cache_client", forbidden)
    prepared = w.prepare_run("run", (), NOW, context=context)
    assert w.publish("run", prepared.registry_version).published


def test_replay_deferred_current_analysis_blocks_until_real_saved_success(store):
    from dataclasses import asdict
    from app.domain.social_signals.records import ExtractionResult, ExtractionPostJudgment
    from app.infra.db.models.social_analysis import SocialExtractionWork
    from app.models.theme import ContentItem
    from app.services.social_signal_backlog_service import ProcessSocialBacklog
    w = writer(store)
    w.create_run("old", NOW)
    w.persist_observations(batch(), run_id="old")
    w.persist_observations(empty_batch(2), run_id="old")
    with store() as db:
        content_id = db.query(ContentItem).one().id
    backlog = ProcessSocialBacklog(store)
    work_id = backlog.enqueue(content_id, batch().posts[0], selected_model="synthetic/model", now=NOW, run_id="old")
    with store.begin() as db:
        db.get(SocialExtractionWork, work_id).state = "waiting_budget"
    old = w.read_run_inputs("old")
    w.create_replay_run("new", "old", NOW + timedelta(hours=1))
    assert w.read_run_inputs("new").work_ids == ()
    with pytest.raises(ValueError, match="incomplete"):
        w.prepare_run("new", (), NOW+timedelta(hours=1))
    assert backlog.enqueue(content_id, batch().posts[0], selected_model="synthetic/model", now=NOW+timedelta(hours=1), run_id="new") == work_id
    with pytest.raises(ValueError, match="incomplete"):
        w.prepare_run("new", (), NOW+timedelta(hours=1))
    with store.begin() as db:
        work = db.get(SocialExtractionWork, work_id)
        result = ExtractionResult(work.input_hash, "synthetic", "model", work.prompt_version, work.schema_version,
            (), 1, 1, (ExtractionPostJudgment("100", True, "claim-100"),))
        work.state, work.actual_provider, work.actual_model, work.result_json = "succeeded", "synthetic", "model", asdict(result)
    w.prepare_run("new", (), NOW+timedelta(hours=1))
    assert w.read_run_inputs("old") == old


def test_validation_to_live_replay_uses_new_generation_and_same_raw_read(store):
    from app.infra.db.models.social_signals import SocialSourceRegistry
    w = writer(store)
    with store.begin() as db:
        db.get(SocialSourceRegistry, 1).mode = "validation"
    collected(w, "validation")
    prior = w.prepare_run("validation", (), NOW)
    assert not w.publish("validation", prior.registry_version).published
    with store.begin() as db:
        registry = db.get(SocialSourceRegistry, 1)
        registry.mode, registry.version = "live", registry.version + 1
    w.create_replay_run("live", "validation", NOW + timedelta(hours=1))
    live = w.prepare_run("live", (), NOW + timedelta(hours=1))
    assert w.publish("live", live.registry_version).published
    assert w.read_run_inputs("live").batches == w.read_run_inputs("validation").batches


def test_replaying_incomplete_replay_retains_entire_original_audit(store):
    from app.infra.db.models.social_signals import SocialSourceRegistry
    w = writer(store)
    w.create_run("old", NOW)
    work_id = save_success(store, w, "old")
    with store.begin() as db:
        registry = db.get(SocialSourceRegistry, 1)
        registry.provider, registry.version = "xui", registry.version + 1
    w.create_replay_run("first", "old", NOW + timedelta(hours=1))
    w.create_replay_run("second", "first", NOW + timedelta(hours=2))
    second = w.read_run_inputs("second")
    assert second.batches == w.read_run_inputs("old").batches
    assert second.replay_manifest.historical_work_ids == (work_id,)
    assert second.replay_manifest.missing_sources


def test_replay_succeeded_carry_in_preserves_first_day_author_cap(store):
    from dataclasses import asdict
    from app.domain.social_signals.records import ExtractionResult, ExtractionPostJudgment, SocialEvidenceInput
    from app.domain.social_signals.scoring import score_social_candidates
    from app.infra.db.models.social_analysis import SocialExtractionWork
    from app.models.theme import ContentItem
    from app.services.social_signal_backlog_service import ProcessSocialBacklog
    w = writer(store)
    w.create_run("old", NOW)
    posts = tuple(replace(batch(post_id=str(100+i)).posts[0], created_at=NOW-timedelta(days=14)+timedelta(hours=hour),
        has_new_thesis=True, canonical_claim_key=f"claim-{i}") for i, hour in enumerate((1, 2, 3, 5)))
    b = batch()
    w.persist_observations(replace(b, posts=posts, outcome=replace(b.outcome, received_count=4,
        observed_oldest_at=posts[0].created_at, observed_newest_at=posts[-1].created_at)), run_id="old")
    w.persist_observations(empty_batch(2), run_id="old")
    with store() as db:
        ids = {v.external_id: v.id for v in db.query(ContentItem)}
    for post in posts:
        work_id = ProcessSocialBacklog(store).enqueue(ids[post.provider_post_id], post,
            selected_model="synthetic/model", now=NOW, run_id="old")
        with store.begin() as db:
            work = db.get(SocialExtractionWork, work_id)
            result = ExtractionResult(work.input_hash, "synthetic", "model", work.prompt_version, work.schema_version,
                (), 1, 1, (ExtractionPostJudgment(post.provider_post_id, True, post.canonical_claim_key),))
            work.state, work.actual_provider, work.actual_model, work.result_json = "succeeded", "synthetic", "model", asdict(result)
    current = NOW + timedelta(hours=4)
    w.create_replay_run("new", "old", current)
    saved = w.read_run_inputs("new")
    assert len(saved.replay_manifest.required_inputs) == 1
    assert len(saved.replay_manifest.carry_in_work_ids) == 3
    evidence = SocialEvidenceInput("US:AAA", "AAA", "US", saved.batches[0].posts, ("1", "2"))
    scored = score_social_candidates((evidence,), 14, current)[0]
    assert scored.mention_count == 0
    assert ("official:103", "author_24h_cap") in scored.exclusions
    w.prepare_run("new", (), current)


def test_grace_boundary_during_final_publication_lock_is_rejected(store):
    from tests.unit.services.test_social_confirmation_reader import utc
    from app.infra.db.repositories.social_signal_writer import SocialSignalWriter
    at = utc("2026-09-08T21:59:59")
    clock = [at]
    w = SocialSignalWriter(store, clock=lambda: clock[0])
    collected(w, "run", at)
    context = publication_context(store, at)
    prepared = w.prepare_run("run", (), at, context=context)
    ticks = iter((at, at+timedelta(seconds=1)))
    w.clock = lambda: next(ticks)
    with pytest.raises(ValueError, match="market_context_changed"):
        w.publish("run", prepared.registry_version)
    with store() as db:
        assert db.query(SocialSignalRunPointer).count() == 0


def test_replay_stores_raw_corpus_once_per_generation(store):
    import json
    w = writer(store)
    w.create_run("old", NOW)
    save_success(store, w, "old")
    w.create_replay_run("new", "old", NOW+timedelta(hours=1))
    with store() as db:
        encoded = json.dumps(db.get(SocialSignalRun, "new").application_progress_json)
        assert encoded.count('"text": "$AAA supplies cooling"') == 1


@pytest.mark.parametrize("corrupt", [False, True])
def test_frozen_theme_context_uses_exact_market_pin_and_roundtrips(db_session, basket, corrupt):
    from sqlalchemy.orm import sessionmaker
    from tests.unit.services.test_social_theme_market_service import NOW as at
    from app.services.social_confirmation_reader import SocialConfirmationReader
    from app.services.social_source_admin_service import SocialSourceAdminService
    from app.infra.db.models.social_signals import SocialSourceRegistry
    from app.models.market_exposure import MarketExposure
    from app.infra.db.repositories.social_signal_writer import SocialSignalWriter
    from app.domain.social_signals.records import SocialPublicationContext
    _, symbols, calendar = basket(3, 3)
    db_session.commit()
    SocialSourceAdminService(db_session).ensure_seed_sources()
    registry = db_session.get(SocialSourceRegistry, 1)
    registry.mode, registry.provider = "live", "official"
    db_session.add(MarketExposure(market="US", date=at.date(), exposure_score=60, stance="uptrend", benchmark_symbol="SPY",
        created_at=at, updated_at=at))
    db_session.commit()
    factory = sessionmaker(db_session.bind)
    w = SocialSignalWriter(factory, clock=lambda: at)
    collected(w, "run", at)
    with factory() as db:
        market = SocialConfirmationReader(db, calendar=calendar).read_market("US", symbols, at)
    assert len(market.theme_evidence) == 1
    if corrupt:
        bad = replace(market.theme_evidence[0], feature_run_ids=tuple((symbol, 999) for symbol in symbols))
        market = replace(market, theme_evidence=(bad,), inputs=tuple(replace(v, theme_confirmations=(bad,)) for v in market.inputs))
    context = SocialPublicationContext((market,))
    if corrupt:
        with pytest.raises(ValueError, match="market_context_changed:theme"):
            w.prepare_run("run", (), at, context=context)
    else:
        prepared = w.prepare_run("run", (), at, context=context)
        assert w.publish("run", prepared.registry_version).published
        assert w.read_publication_context("run").market_batches == (market,)


@pytest.mark.parametrize("mixed", [False, True])
def test_normal_bounded_read_selects_current_inputs_and_audits_linked_aged_work(store, mixed):
    from app.models.theme import ContentItem
    from app.infra.db.models.social_analysis import SocialExtractionWork
    from app.services.social_signal_backlog_service import ProcessSocialBacklog
    w = writer(store)
    w.create_run("run", NOW)
    old = replace(batch(post_id="old").posts[0], created_at=NOW-timedelta(days=16))
    posts = (old, batch().posts[0]) if mixed else (old,)
    b = batch()
    w.persist_observations(replace(b, posts=posts, outcome=replace(b.outcome, received_count=len(posts),
        observed_oldest_at=old.created_at, observed_newest_at=posts[-1].created_at)), run_id="run")
    w.persist_observations(empty_batch(2), run_id="run")
    with store() as db:
        ids = {v.external_id: v.id for v in db.query(ContentItem)}
    aged_id = ProcessSocialBacklog(store).enqueue(ids["old"], old, selected_model="synthetic/model", now=NOW, run_id="run")
    with store.begin() as db:
        db.get(SocialExtractionWork, aged_id).state = "outside_window"
    saved = w.read_run_inputs("run")
    selected = w.select_current_inputs("run")
    assert selected.batches == saved.batches
    assert len(selected.current_manifest.required_inputs) == int(mixed)
    assert selected.current_manifest.audit_work_ids == (aged_id,)
    assert selected.work_ids == ()
    assert w.select_current_inputs("run") == selected
    if mixed:
        with pytest.raises(ValueError, match="incomplete"):
            w.prepare_run("run", (), NOW)
    else:
        w.prepare_run("run", (), NOW)
        w.create_replay_run("replay", "run", NOW+timedelta(hours=1))
        assert w.read_run_inputs("replay").replay_manifest.historical_work_ids == (aged_id,)
    with store() as db:
        assert db.get(SocialExtractionWork, aged_id).state == "outside_window"


@pytest.mark.parametrize("published_after_cutoff", [False, True])
def test_later_observation_keeps_true_timing_and_publication_cutoff(store, published_after_cutoff):
    from dataclasses import asdict
    from app.domain.social_signals.records import ExtractionResult, ExtractionPostJudgment
    from app.infra.db.models.social_analysis import SocialExtractionWork
    from app.infra.db.repositories.social_signal_writer import SocialSignalWriter
    from app.models.theme import ContentItem
    from app.services.social_signal_backlog_service import ProcessSocialBacklog
    observed = NOW + timedelta(minutes=1)
    w = SocialSignalWriter(store, clock=lambda: observed+timedelta(seconds=1))
    w.create_run("run", NOW)
    b = batch()
    post = replace(b.posts[0], observed_at=observed,
        created_at=NOW+timedelta(seconds=30) if published_after_cutoff else b.posts[0].created_at)
    b = replace(b, request=replace(b.request, observed_at=observed), posts=(post,),
        outcome=replace(b.outcome, observed_oldest_at=post.created_at, observed_newest_at=post.created_at))
    w.persist_observations(b, run_id="run")
    w.persist_observations(empty_batch(2), run_id="run")
    with store() as db:
        content_id = db.query(ContentItem).one().id
    work_id = ProcessSocialBacklog(store).enqueue(content_id, post, selected_model="synthetic/model", now=observed, run_id="run")
    with store.begin() as db:
        work = db.get(SocialExtractionWork, work_id)
        result = ExtractionResult(work.input_hash, "synthetic", "model", work.prompt_version, work.schema_version,
            (), 1, 1, (ExtractionPostJudgment("100", True, "claim-100"),))
        work.state, work.actual_provider, work.actual_model, work.result_json = "succeeded", "synthetic", "model", asdict(result)
    w.prepare_run("run", (), NOW)
    saved = w.read_run_inputs("run")
    assert saved.as_of == NOW
    assert saved.batches[0].request.observed_at == observed
    assert saved.batches[0].posts == (post,)
    assert len(saved.current_manifest.required_inputs) == int(not published_after_cutoff)
    assert saved.current_manifest.carry_in_work_ids == ()
    assert saved.work_ids == (() if published_after_cutoff else (work_id,))
    if published_after_cutoff:
        assert saved.current_manifest.audit_work_ids == (work_id,)
        assert "publication_after_evaluation" in saved.current_manifest.coverage_reasons


@pytest.mark.parametrize("field", ["request", "post", "publication"])
def test_ingress_rejects_observation_beyond_trusted_clock_tolerance(store, field):
    from app.models.theme import ContentItem
    w = writer(store)
    w.create_run("run", NOW)
    b = batch()
    if field == "request": b = replace(b, request=replace(b.request, observed_at=NOW+timedelta(minutes=6)))
    elif field == "post": b = replace(b, posts=(replace(b.posts[0], observed_at=NOW+timedelta(minutes=6)),))
    else: b = replace(b, posts=(replace(b.posts[0], observed_at=NOW+timedelta(minutes=5), created_at=NOW+timedelta(minutes=6)),))
    with pytest.raises(ValueError, match="future_timestamp"):
        w.persist_observations(b, run_id="run")
    with store() as db:
        assert db.query(ContentItem).count() == 0


@pytest.mark.parametrize("contradiction", ["missing_actionable", "decision", "social_state", "other_confirmation", "component_reasons", "coherent_actionable"])
def test_candidate_evidence_must_agree_with_frozen_market_input(store, contradiction):
    from app.domain.social_signals.records import SignalStateDecision, SocialPublicationContext
    from app.domain.social_signals.states import classify_signal_state
    from app.domain.social_signals.scoring import score_confirmation, queue_score
    from app.services.social_confirmation_reader import SocialConfirmationReader
    from tests.unit.services.test_social_confirmation_reader import seed_batch, utc
    at = NOW
    if contradiction in {"other_confirmation", "coherent_actionable"}:
        at = utc("2026-07-02T23:00:00")
        with store.begin() as db:
            seed_batch(db)
    w = writer(store)
    w.clock = lambda: at
    collected(w, now=at)
    with store() as db:
        market = SocialConfirmationReader(db).read_market("US", ("AAA", "BBB"), at, theme_keys=())
    context = SocialPublicationContext((market,))
    candidate = candidate_context(context)
    if contradiction == "missing_actionable":
        fake = replace(candidate.state_input, feature_fresh=True, market_fresh=True, liquidity_eligible=True,
            setup_ready=True, setup_score=80, market_exposure=60)
        decision = classify_signal_state(fake)
        candidate = replace(candidate, state_input=fake, state_decision=decision,
            social_result=replace(candidate.social_result, state=decision))
    elif contradiction == "decision":
        decision = SignalStateDecision("actionable")
        candidate = replace(candidate, state_decision=decision, social_result=replace(candidate.social_result, state=decision))
    elif contradiction == "social_state":
        candidate = replace(candidate, social_result=replace(candidate.social_result, state=SignalStateDecision("actionable")))
    elif contradiction == "other_confirmation":
        other = next(v for v in market.inputs if v.candidate_key == "US:BBB")
        candidate = replace(candidate, confirmation=score_confirmation(other))
    elif contradiction == "component_reasons":
        candidate = replace(candidate, confirmation=replace(candidate.confirmation, reasons=("invented",)))
    context = replace(context, candidates=(candidate,))
    record = replace(row("run"), candidate_state=candidate.state_decision.state,
        confirmation_score=candidate.confirmation.value,
        queue_score=queue_score(social=80, confirmation=candidate.confirmation.value))
    if contradiction == "coherent_actionable":
        assert candidate.state_decision.state == "actionable"
        prepared = w.prepare_run("run", (record,), at, context=context)
        assert w.publish("run", prepared.registry_version).published
    else:
        with pytest.raises(ValueError, match="candidate_context_mismatch"):
            w.prepare_run("run", (record,), at, context=context)


@pytest.mark.parametrize("required,actual,fresh,reason,want", [
    (None, None, False, "calendar_unavailable", None),
    (NOW.date(), None, False, "missing_session", None),
    (NOW.date(), NOW.date(), False, "listing_mic_unknown", None),
    (NOW.date(), NOW.date()-timedelta(days=1), False, "stale_session", False),
    (NOW.date(), NOW.date()+timedelta(days=1), False, "future_session", False),
    (NOW.date(), NOW.date(), True, None, True),
])
def test_frozen_freshness_maps_unknown_and_known_failure_to_classifier(required, actual, fresh, reason, want):
    from app.domain.social_signals.records import DailyFreshness
    value = DailyFreshness(required, actual, fresh, reason)
    assert value.signal_state_value is want
    assert value.reason == reason


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
    from app.services.social_signal_backlog_service import ProcessSocialBacklog
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
