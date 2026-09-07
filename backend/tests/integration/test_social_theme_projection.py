"""Shared catalog projection against isolated real repositories, no provider I/O."""
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.domain.social_signals.records import ExtractionClaim, ExtractionPostJudgment, ExtractionResult, SocialPostRecord
from app.infra.db.models.social_analysis import SocialExtractionWork, SocialRunWork
from app.infra.db.models.social_signals import SocialSignalRun, SocialSourceRegistry
from app.models.theme import ContentItem, ThemeCluster, ThemeConstituent, ThemeMention
from app.models.stock_universe import StockUniverse
from app.services.social_extraction_service import SocialExtractionService

NOW = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)


@pytest.fixture
def social_fixture(tmp_path):
    from app.services.social_theme_projection_service import SocialThemeProjectionService
    from app.services.social_company_identity_service import SocialCompanyIdentityService
    engine = create_engine(f"sqlite:///{tmp_path / 'themes.sqlite'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory.begin() as db:
        db.add(SocialSourceRegistry(id=1, mode="validation", provider="official"))
        for symbol, market in [("AAA", "US"), ("BBB", "US"), ("CCC", "US"), ("0005.HK", "HK")]:
            db.add(StockUniverse(symbol=symbol, market=market, is_active=True))
    with factory() as db:
        SocialCompanyIdentityService(db, admin_authorized=True).replace([
            {"symbol": s, "company_id": c, "verification_reference": "synthetic-review:1", "verified_at": NOW.isoformat()}
            for s, c in [("AAA", "a"), ("BBB", "b"), ("CCC", "c"), ("0005.HK", "a")]], expected_version=1, actor="admin")
    with factory() as db:
        yield Fixture(db, SocialThemeProjectionService)
    engine.dispose()


class Fixture:
    def __init__(self, db, cls):
        self.db, self.service = db, cls(db, admin_authorized=True)
        self.serial = 0

    def save(self, symbols=("AAA",), *, author=None, age=1, support="supported", thesis=True, key=None, repost=False):
        self.serial += 1
        number = str(self.serial)
        text = " ".join(symbols) + " supply cooling equipment"
        post = SocialPostRecord("official", number, "fixture-source", text, f"https://x.com/a/status/{number}",
                                author or f"author-{number}", NOW - timedelta(days=age), NOW, is_repost=repost)
        item = ContentItem(source_type="twitter", external_id=number, url=post.url, content=text, published_at=post.created_at)
        self.db.add(item)
        self.db.flush()
        input_hash = SocialExtractionService.input_hash((post,))
        claims = tuple(ExtractionClaim(number, "cooling", "Cooling", symbol, "supplies", text, support, ()) for symbol in symbols)
        result = ExtractionResult(input_hash, "synthetic", "model", "social-extraction-v1", "social-extraction-v1", claims, 10, 10,
                                  (ExtractionPostJudgment(number, thesis, key or f"claim-{number}"),))
        snapshot = asdict(post)
        snapshot.update(created_at=post.created_at.isoformat(), observed_at=NOW.isoformat())
        work = SocialExtractionWork(content_item_id=item.id, input_hash=input_hash, prompt_version=result.prompt_version,
            schema_version=result.schema_version, selected_model="synthetic/model", input_snapshot_json=snapshot,
            actual_provider="synthetic", actual_model="model", state="succeeded", result_json=asdict(result))
        self.db.add(work)
        self.db.commit()
        return work.id

    def prepare(self, works, mode="live"):
        registry = self.db.get(SocialSourceRegistry, 1)
        if registry.mode != mode:
            registry.mode, registry.version = mode, registry.version + 1
        run = SocialSignalRun(id=f"run-{self.serial}-{mode}-{self.db.query(SocialSignalRun).count()}", registry_id=1,
            registry_version=registry.version, mode=mode, provider="official", status="running", source_outcomes_json={},
            application_progress_json={}, feature_run_ids_json={}, exposure_dates_json={}, coverage_json={})
        self.db.add(run)
        self.db.flush()
        for work_id in works:
            self.db.add(SocialRunWork(run_id=run.id, work_id=work_id, input_hash=self.db.get(SocialExtractionWork, work_id).input_hash, included_at=NOW))
        self.db.flush()
        run.status = "staged"
        self.db.commit()
        return self.service.prepare(run.id, NOW)

    def apply(self, projection):
        self.service.apply_live(projection, projection.registry_version)
        self.db.commit()

    def associations(self):
        from app.infra.db.models.social_analysis import SocialThemeAssociation
        return self.db.query(SocialThemeAssociation).order_by(SocialThemeAssociation.canonical_symbol).all()


def test_validation_does_not_mutate_live_catalog(social_fixture):
    from app.infra.db.models.social_signals import SocialSignalRunPointer
    f = social_fixture
    projection = f.prepare([f.save(("AAA", "BBB", "CCC"))], "validation")
    assert len(projection.proposals) == 3
    assert f.db.query(ThemeCluster).count() == f.db.query(ThemeMention).count() == 0
    assert f.associations() == []
    assert f.db.query(SocialSignalRunPointer).count() == 0
    with pytest.raises(ValueError, match="live"):
        f.apply(projection)
    f.db.rollback()
    assert f.db.query(ThemeCluster).count() == 0


def test_empty_catalog_proposes_accepts_promotes_and_is_idempotent(social_fixture):
    f = social_fixture
    first = f.save(("AAA", "BBB", "CCC"), age=3)
    f.apply(f.prepare([first]))
    theme = f.db.query(ThemeCluster).one()
    assert theme.lifecycle_state == "candidate"
    assert [a.state for a in f.associations()] == ["proposed"] * 3
    assert f.service.effective_live_membership(theme.id) == ()
    second = f.save(("AAA", "BBB", "CCC"), age=2)
    f.apply(f.prepare([second]))
    assert [a.state for a in f.associations()] == ["accepted"] * 3
    assert len(f.service.effective_live_membership(theme.id)) == 3
    assert theme.lifecycle_state == "candidate"
    third = f.save(("AAA",), age=1)
    p = f.prepare([third])
    f.apply(p)
    assert theme.lifecycle_state == "active"
    assert f.db.query(ThemeCluster).count() == 1
    assert f.db.query(ThemeConstituent).count() == 0
    count = f.db.query(ThemeMention).count()
    f.apply(p)
    assert f.db.query(ThemeMention).count() == count
    from app.services.theme_discovery_service import ThemeDiscoveryService
    legacy = ThemeDiscoveryService(f.db)
    assert legacy.calculate_mention_metrics(theme.id, NOW)["mentions_7d"] == 0
    assert legacy._count_active_ingestion_days(NOW - timedelta(days=14), NOW) == 0
    legacy.apply_dormancy_and_reactivation_policies(now=NOW)
    assert theme.lifecycle_state == "active"


@pytest.mark.parametrize("change", [{"author": "same"}, {"support": "uncertain"}, {"thesis": False}, {"repost": True}, {"key": "copied"}, {"age": 15}])
def test_non_corroborating_evidence_does_not_accept(social_fixture, change):
    f = social_fixture
    first = f.save(author="same", key="copied")
    second = f.save(**change)
    f.apply(f.prepare([first, second]))
    assert f.associations()[0].state == "proposed"


def test_unknown_identity_and_alternate_listings_do_not_inflate_gate(social_fixture):
    f = social_fixture
    f.db.add(StockUniverse(symbol="DDD", market="US", is_active=True))
    f.db.commit()
    works = [f.save(("AAA", "0005.HK", "BBB", "DDD"), age=n) for n in (1, 2, 3)]
    projection = f.prepare(works)
    assert any("company_identity_unknown" in r.reason_codes for r in projection.resolutions)
    f.apply(projection)
    assert f.db.query(ThemeCluster).one().lifecycle_state == "candidate"
    assert next(a for a in f.associations() if a.canonical_symbol == "DDD").state == "proposed"


def test_alternate_listing_acceptance_snapshots_only_qualifying_company_work(social_fixture):
    from app.infra.db.models.social_analysis import SocialThemeDecision
    f = social_fixture
    primary = f.save(("AAA",), age=3, key="original")
    alternate = f.save(("0005.HK",), age=2)
    copied = f.save(("AAA",), age=1, key="original")
    uncertain = f.save(("AAA",), age=1, support="uncertain")
    unrelated = f.save(("BBB",), age=1)
    f.apply(f.prepare([primary, alternate, copied, uncertain, unrelated]))
    associations = {a.canonical_symbol: a for a in f.associations()}
    assert associations["AAA"].state == associations["0005.HK"].state == "accepted"
    assert associations["BBB"].state == "proposed"
    assert associations["AAA"].evidence_work_ids == [primary, copied, uncertain]
    assert associations["0005.HK"].evidence_work_ids == [alternate]
    decisions = f.db.query(SocialThemeDecision).order_by(SocialThemeDecision.id).all()
    assert len(decisions) == 2
    assert {d.association_id for d in decisions} == {associations["AAA"].id, associations["0005.HK"].id}
    assert [d.evidence_work_ids for d in decisions] == [[primary, alternate], [primary, alternate]]
    original_snapshots = [(d.id, d.run_id, list(d.evidence_work_ids)) for d in decisions]

    later = f.save(("AAA",), age=0)
    f.apply(f.prepare([later]))
    assert associations["AAA"].evidence_work_ids == [primary, copied, uncertain, later]
    f.db.expire_all()
    decisions = f.db.query(SocialThemeDecision).order_by(SocialThemeDecision.id).all()
    assert [(d.id, d.run_id, d.evidence_work_ids) for d in decisions] == original_snapshots


@pytest.mark.parametrize("legacy_first", [True, False])
def test_admin_override_and_independent_legacy_membership(social_fixture, legacy_first):
    f = social_fixture
    first = f.save()
    p = f.prepare([first])
    if legacy_first:
        f.db.add(ThemeCluster(name="Cooling", display_name="Cooling", canonical_key="cooling", pipeline="technical"))
        f.db.flush()
        f.db.add(ThemeConstituent(theme_cluster_id=f.db.query(ThemeCluster).one().id, symbol="AAA", source="manual", is_active=True))
    f.apply(p)
    association = f.associations()[0]
    if not legacy_first:
        f.db.add(ThemeConstituent(theme_cluster_id=association.theme_cluster_id, symbol="AAA", source="manual", is_active=True))
        f.db.commit()
    with pytest.raises(ValueError):
        f.service.decide(association.id, "rejected", "", "admin", association.version)
    if not f.db.in_transaction():
        f.db.begin()
    f.service.decide(association.id, "rejected", "reviewed evidence", "admin", association.version)
    f.db.commit()
    assert f.service.effective_live_membership(association.theme_cluster_id)[0].origins == ("legacy",)
    f.apply(f.prepare([f.save(), f.save()]))
    assert association.state == "rejected" and association.decision_owner == "admin"
    f.db.begin()
    with pytest.raises(ValueError, match="version"):
        f.service.decide(association.id, "accepted", "new review", "admin", 0)


def test_registry_change_cancels_projection_and_caller_rollback_is_atomic(social_fixture):
    f = social_fixture
    p = f.prepare([f.save()])
    f.service.apply_live(p, p.registry_version)
    f.db.rollback()
    assert f.db.query(ThemeCluster).count() == 0
    registry = f.db.get(SocialSourceRegistry, 1)
    registry.mode, registry.version = "off", registry.version + 1
    f.db.commit()
    with pytest.raises(ValueError):
        f.apply(p)
    f.db.rollback()
    assert f.db.query(ThemeCluster).count() == 0


@pytest.mark.parametrize("corruption", ["hash", "excerpt", "post", "state"])
def test_saved_result_attribution_must_match_pinned_input(social_fixture, corruption):
    f = social_fixture
    work_id = f.save()
    p = f.prepare([work_id])
    work = f.db.get(SocialExtractionWork, work_id)
    result = dict(work.result_json)
    if corruption == "hash":
        result["input_hash"] = "other"
    elif corruption == "state":
        work.state = "waiting_budget"
    else:
        result["claims"] = [{**result["claims"][0], "excerpt" if corruption == "excerpt" else "post_id": "invented"}]
    work.result_json = result
    f.db.commit()
    with pytest.raises(ValueError):
        f.service.prepare(p.run_id, NOW)


def test_admin_history_is_immutable_and_non_admin_cannot_decide(social_fixture):
    from app.infra.db.models.social_analysis import SocialThemeDecision
    from app.services.social_theme_projection_service import SocialThemeProjectionService
    f = social_fixture
    f.apply(f.prepare([f.save()]))
    a = f.associations()[0]
    with pytest.raises(PermissionError):
        SocialThemeProjectionService(f.db).decide(a.id, "accepted", "review", "spoof", a.version)
    f.service.decide(a.id, "accepted", "review", "admin", a.version)
    f.db.commit()
    decision = f.db.query(SocialThemeDecision).one()
    decision.reason = "rewritten"
    with pytest.raises(ValueError, match="append_only"):
        f.db.flush()
    f.db.rollback()


def test_identity_change_invalidates_inflight_projection(social_fixture):
    from app.services.social_company_identity_service import SocialCompanyIdentityService
    f = social_fixture
    p = f.prepare([f.save()])
    f.db.rollback()
    SocialCompanyIdentityService(f.db, admin_authorized=True).replace([], expected_version=p.registry_version, actor="admin")
    f.db.begin()
    with pytest.raises(ValueError, match="version"):
        f.apply(p)
    f.db.rollback()
    assert f.db.query(ThemeCluster).count() == 0


def test_backfilled_legacy_row_records_social_decision_independently(social_fixture):
    from app.infra.db.models.social_analysis import SocialThemeAssociation
    f = social_fixture
    first = f.save()
    f.db.add(ThemeCluster(name="Cooling", display_name="Cooling", canonical_key="cooling", pipeline="technical"))
    f.db.flush()
    theme = f.db.query(ThemeCluster).one()
    f.db.add(ThemeConstituent(theme_cluster_id=theme.id, symbol="AAA", source="manual", is_active=True))
    f.db.add(SocialThemeAssociation(theme_cluster_id=theme.id, market="US", canonical_symbol="AAA", state="accepted",
        origin="legacy", decision_owner="system", evidence_work_ids=[], policy_version="legacy-preserved-v1", version=1,
        first_seen_at=NOW, accepted_at=NOW, updated_at=NOW))
    f.db.commit()
    f.apply(f.prepare([first]))
    association = f.associations()[0]
    assert (association.origin, association.state) == ("social", "proposed")
    assert f.service.effective_live_membership(theme.id)[0].origins == ("legacy",)


def test_expired_social_lifecycle_evidence_does_not_expire_membership(social_fixture):
    from app.services.theme_discovery_service import ThemeDiscoveryService
    f = social_fixture
    works = [f.save(("AAA", "BBB", "CCC"), age=age) for age in (13, 12, 11)]
    f.apply(f.prepare(works))
    theme = f.db.query(ThemeCluster).one()
    assert theme.lifecycle_state == "active"
    ThemeDiscoveryService(f.db).apply_dormancy_and_reactivation_policies(now=NOW + timedelta(days=4))
    assert theme.lifecycle_state == "dormant"
    assert len(f.service.effective_live_membership(theme.id)) == 3


def test_candidate_survives_legacy_passes_and_aging_without_invented_retirement(social_fixture):
    from app.services.theme_discovery_service import ThemeDiscoveryService
    f = social_fixture
    f.apply(f.prepare([f.save()]))
    theme = f.db.query(ThemeCluster).one()
    for now in (NOW, NOW + timedelta(days=30)):
        legacy = ThemeDiscoveryService(f.db)
        legacy.promote_candidate_themes(now=now)
        legacy.apply_dormancy_and_reactivation_policies(now=now)
        assert theme.lifecycle_state == "candidate"
        assert f.service.effective_live_membership(theme.id) == ()


def test_revised_work_cannot_recount_canonical_post_or_rewrite_first_run(social_fixture):
    f = social_fixture
    original_id = f.save()
    original_projection = f.prepare([original_id])
    f.apply(original_projection)
    original = f.db.get(SocialExtractionWork, original_id)
    revised = SocialExtractionWork(content_item_id=original.content_item_id, input_hash=original.input_hash,
        prompt_version=original.prompt_version, schema_version=original.schema_version, selected_model="synthetic/revised",
        input_snapshot_json=original.input_snapshot_json, actual_provider="synthetic", actual_model="model",
        state="succeeded", result_json=original.result_json)
    f.db.add(revised)
    f.db.commit()
    f.apply(f.prepare([revised.id]))
    assert f.associations()[0].state == "proposed"
    assert f.db.query(ThemeMention).filter_by(social_work_id=original_id).one().social_run_id == original_projection.run_id
    assert f.db.query(ThemeCluster).one().lifecycle_state == "candidate"


def test_cached_taxonomy_classification_is_live_only_and_rolls_back(social_fixture):
    from app.models.theme import ThemeEmbedding
    from app.services.theme_taxonomy_service import CENTROID_EMBEDDING_MODEL
    f = social_fixture
    parent = ThemeCluster(name="Technology", display_name="Technology", canonical_key="technology_l1", pipeline="technical", is_l1=True, lifecycle_state="active", is_active=True)
    child = ThemeCluster(name="Cooling", display_name="Cooling", canonical_key="cooling", pipeline="technical", is_active=True)
    f.db.add_all([parent, child])
    f.db.flush()
    f.db.add_all([ThemeEmbedding(theme_cluster_id=parent.id, embedding="[1,0]", embedding_model=CENTROID_EMBEDDING_MODEL),
                  ThemeEmbedding(theme_cluster_id=child.id, embedding="[1,0]")])
    f.db.commit()
    work = f.save()
    f.prepare([work], "validation")
    assert child.parent_cluster_id is None
    p = f.prepare([work])
    f.service.apply_live(p, p.registry_version)
    assert child.parent_cluster_id == parent.id
    f.db.rollback()
    assert child.parent_cluster_id is None


@pytest.mark.parametrize("social_first", [True, False])
def test_identical_legacy_social_post_preserves_legacy_attention(social_fixture, social_first):
    from app.models.theme import ContentSource
    from app.services.theme_evidence_eligibility_service import grant_eligibility
    from app.services.theme_discovery_service import ThemeDiscoveryService
    f = social_fixture
    work_id = f.save()
    item_id = f.db.get(SocialExtractionWork, work_id).content_item_id
    source = ContentSource(name="Independent legacy", source_type="twitter", is_active=True)
    theme = ThemeCluster(name="Cooling", display_name="Cooling", canonical_key="cooling", pipeline="technical", is_active=True)
    f.db.add_all([source, theme])
    f.db.commit()
    if social_first:
        f.apply(f.prepare([work_id]))
    grant_eligibility(f.db, item_id, "technical", "legacy", source.id, NOW - timedelta(days=1))
    f.db.add(ThemeMention(content_item_id=item_id, source_type="twitter", raw_theme="Cooling", theme_cluster_id=theme.id,
        pipeline="technical", mentioned_at=NOW - timedelta(days=1), confidence=1.0, sentiment="bullish"))
    f.db.commit()
    if not social_first:
        f.apply(f.prepare([work_id]))
    discovery = ThemeDiscoveryService(f.db)
    metrics = discovery.calculate_mention_metrics(theme.id, NOW)
    assert metrics["mentions_7d"] == 1
    assert metrics["sentiment_score"] == 1.0
    assert discovery._count_active_ingestion_days(NOW - timedelta(days=14), NOW) == 1


def test_social_mention_cannot_hide_failed_legacy_extraction(social_fixture):
    from app.models.theme import ContentSource, ContentItemPipelineState
    from app.services.theme_evidence_eligibility_service import grant_eligibility
    from app.services.theme_extraction_service import ThemeExtractionService
    f = social_fixture
    work_id = f.save()
    f.apply(f.prepare([work_id]))
    item_id = f.db.get(SocialExtractionWork, work_id).content_item_id
    source = ContentSource(name="Independent legacy", source_type="twitter", url="https://x.com/independent", is_active=True, pipelines=["technical"])
    f.db.add(source)
    f.db.flush()
    grant_eligibility(f.db, item_id, "technical", "legacy", source.id, NOW)
    f.db.add(ContentItemPipelineState(content_item_id=item_id, pipeline="technical", status="processed", attempt_count=1))
    f.db.commit()
    result = ThemeExtractionService(f.db).identify_silent_failures(max_age_days=30)
    assert result["reset_count"] == 1


@pytest.mark.parametrize("path", ["live", "extraction"])
def test_merge_refuses_social_evidence_without_mutation(social_fixture, path, monkeypatch):
    from app.services.theme_merging_service import ThemeMergingService, ThemeMergeConflictError
    from app.services.theme_extraction_service import ThemeNormalizationService
    monkeypatch.setattr(ThemeMergingService, "update_theme_embedding", lambda *args: None)
    f = social_fixture
    f.apply(f.prepare([f.save()]))
    source = f.db.query(ThemeCluster).one()
    target = ThemeCluster(name="Other", display_name="Other", canonical_key="other", pipeline="technical", is_active=True)
    f.db.add(target)
    f.db.commit()
    service = ThemeMergingService(f.db) if path == "live" else ThemeNormalizationService(f.db)
    if path == "live":
        result = service.execute_merge(source.id, target.id)
        assert not result["success"]
        assert result["error_code"] == "theme_merge_social_evidence_unsupported"
    else:
        with pytest.raises(ThemeMergeConflictError) as error:
            service.merge_clusters(source.id, target.id)
        assert error.value.error_code == "theme_merge_social_evidence_unsupported"
    f.db.rollback()
    assert source.is_active
    assert f.db.query(ThemeMention).one().theme_cluster_id == source.id
    assert f.associations()[0].theme_cluster_id == source.id


def test_legacy_backfill_merge_still_works_and_registry_locks_first(social_fixture, monkeypatch):
    from sqlalchemy import event
    from app.infra.db.models.social_analysis import SocialThemeAssociation
    from app.services.theme_merging_service import ThemeMergingService
    monkeypatch.setattr(ThemeMergingService, "update_theme_embedding", lambda *args: None)
    f = social_fixture
    source = ThemeCluster(name="Source", display_name="Source", canonical_key="source", pipeline="technical", is_active=True)
    target = ThemeCluster(name="Target", display_name="Target", canonical_key="target", pipeline="technical", is_active=True)
    f.db.add_all([source, target])
    f.db.flush()
    f.db.add(ThemeConstituent(theme_cluster_id=source.id, symbol="AAA", source="manual", is_active=True))
    f.db.add(SocialThemeAssociation(theme_cluster_id=source.id, market="US", canonical_symbol="AAA", state="accepted", origin="legacy",
        decision_owner="system", evidence_work_ids=[], policy_version="legacy-preserved-v1", version=1, first_seen_at=NOW, updated_at=NOW))
    f.db.commit()
    statements = []
    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement.lower())
    event.listen(f.db.bind, "before_cursor_execute", capture)
    try:
        result = ThemeMergingService(f.db).execute_merge(source.id, target.id)
    finally:
        event.remove(f.db.bind, "before_cursor_execute", capture)
    assert result["success"]
    assert f.db.query(ThemeConstituent).one().theme_cluster_id == target.id
    first_theme_read = next(i for i, sql in enumerate(statements) if sql.startswith("select") and "from theme_clusters" in sql)
    assert any("update social_source_registry" in sql for sql in statements[:first_theme_read])


def test_migration_preserves_legacy_membership_and_roundtrips(tmp_path):
    import importlib.util
    from pathlib import Path
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import inspect, text
    path = Path(__file__).resolve().parents[2] / "alembic/versions/20260907_0037_add_social_theme_associations.py"
    assert path.exists(), "association migration missing"
    spec = importlib.util.spec_from_file_location("theme_projection_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = create_engine(f"sqlite:///{tmp_path / 'migration.sqlite'}")
    Base.metadata.create_all(engine)
    from app.infra.db.models.social_analysis import SocialThemeAssociation, SocialThemeDecision
    with engine.begin() as conn:
        SocialThemeDecision.__table__.drop(conn)
        SocialThemeAssociation.__table__.drop(conn)
        with Operations.context(MigrationContext.configure(conn)):
            with Operations(MigrationContext.configure(conn)).batch_alter_table("theme_mentions") as batch:
                batch.drop_constraint("uq_social_work_theme_mention", type_="unique")
                batch.drop_column("social_work_id")
                batch.drop_column("social_run_id")
        conn.execute(text("INSERT INTO theme_clusters (id,name,display_name,canonical_key,pipeline,is_l1,taxonomy_level,lifecycle_state) VALUES (1,'Cooling','Cooling','cooling','technical',0,2,'active')"))
        conn.execute(text("INSERT INTO theme_constituents (theme_cluster_id,symbol,is_active) VALUES (1,'AAA',1)"))
        conn.execute(StockUniverse.__table__.insert().values(symbol="AAA", market="US", is_active=True))
        with Operations.context(MigrationContext.configure(conn)):
            migration.upgrade()
            row = conn.execute(text("SELECT origin,state,canonical_symbol FROM social_theme_associations")).one()
            assert tuple(row) == ("legacy", "accepted", "AAA")
            assert {c["name"] for c in inspect(conn).get_columns("social_theme_associations")} == set(SocialThemeAssociation.__table__.c.keys())
            migration.downgrade()
        assert conn.execute(text("SELECT count(*) FROM theme_constituents")).scalar() == 1
        assert "social_work_id" not in {c["name"] for c in inspect(conn).get_columns("theme_mentions")}
    engine.dispose()
