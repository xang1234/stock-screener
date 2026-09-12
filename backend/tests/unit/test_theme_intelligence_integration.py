"""Integration of grouped reads, revision work, and additive schema."""

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from app.api.v1.themes_intelligence import theme_developments
from app.database import Base
from app.infra.db.models.social_signals import ContentPipelineEligibility
from app.models.theme import (
    ContentItem,
    ContentSource,
    ThemeCluster,
    ThemeConstituent,
    ThemeMention,
    ThemeRelationship,
)
from app.models.theme_intelligence import (
    ThemeDevelopmentObservation,
    ThemeDevelopmentWork,
)
from app.services.theme_development_worker import enqueue, process_one
from app.services.theme_discovery_service import ThemeDiscoveryService
from app.services.theme_equivalence_service import (
    ThemeEquivalenceService,
    guard_grouped_merge,
)
from app.services.theme_group_reads import grouped_constituents
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

NOW = datetime.now(timezone.utc)


@pytest.fixture
def sessions():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine)
    engine.dispose()


def seed(db):
    source = ContentSource(
        name="News", source_type="news", url="https://example.com", is_active=True
    )
    a = ThemeCluster(
        name="CPO",
        display_name="CPO",
        canonical_key="cpo",
        pipeline="technical",
        is_active=True,
    )
    b = ThemeCluster(
        name="Co-Packaged Optics",
        display_name="Co-Packaged Optics",
        canonical_key="co_packaged_optics",
        pipeline="technical",
        is_active=True,
    )
    db.add_all([source, a, b])
    db.flush()
    item = ContentItem(
        source_id=source.id,
        source_type="news",
        content="Nebius Project A order 2026-09 rumored",
        published_at=NOW,
        fetched_at=NOW,
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
    for theme in (a, b):
        db.add(
            ThemeMention(
                content_item_id=item.id,
                theme_cluster_id=theme.id,
                raw_theme=theme.name,
                pipeline="technical",
                source_type="news",
                tickers=["NBIS"],
                mentioned_at=NOW,
            )
        )
        db.add(
            ThemeConstituent(
                theme_cluster_id=theme.id,
                symbol="NBIS",
                is_active=True,
                mention_count=1,
            )
        )
    db.flush()
    return item, a, b


def facts(bundle):
    return [
        {
            "theme_ids": bundle["theme_ids"][:1],
            "actor": "Nebius",
            "action": "order",
            "object": "Project A",
            "event_time": "2026-09",
            "status": "rumored",
            "summary": "Nebius reportedly ordered Project A",
            "citations": [
                {
                    "source_id": "primary",
                    "quote": "Nebius Project A order 2026-09 rumored",
                }
            ],
        }
    ]


def test_group_metrics_and_constituents_count_each_parent_once(sessions):
    with sessions.begin() as db:
        item, a, b = seed(db)
        group = ThemeEquivalenceService(db)
        operation = group.apply(
            a.id, b.id, actor="test", reason="Equivalent", key="one"
        )
        metrics = ThemeDiscoveryService(db)._calculate_mention_metrics_batch(
            [a.id, b.id], as_of_date=NOW
        )
        assert metrics[a.id]["mentions_7d"] == metrics[b.id]["mentions_7d"] == 1
        rows = grouped_constituents(db, a.id)
        assert len(rows) == 1 and rows[0].mention_count == 1
        assert db.query(ThemeMention).filter_by(content_item_id=item.id).count() == 2
        with pytest.raises(Exception, match="reversible"):
            guard_grouped_merge(db, a.id, b.id)
        group.undo(operation["id"], actor="test", reason="Undo")
        assert group.members(a.id) == [a.id]


def test_worker_idempotency_failure_and_correction(sessions):
    with sessions.begin() as db:
        item, a, _b = seed(db)
        item_id, theme_id = item.id, a.id
        work = enqueue(db, item.id, "technical")
        assert enqueue(db, item.id, "technical").id == work.id
    assert process_one(sessions, generate=lambda p, db, bundle: facts(bundle))
    assert not process_one(
        sessions, generate=lambda *args: pytest.fail("duplicate model call")
    )
    with sessions.begin() as db:
        row = db.query(ThemeDevelopmentObservation).one()
        assert not row.superseded
        assert theme_developments(theme_id, limit=1, db=db)["event_count"] == 1
        item = db.get(ContentItem, item_id)
        item.content += " corrected"
        enqueue(db, item.id, "technical")

    def fail(*args):
        raise RuntimeError("provider unavailable")

    assert process_one(sessions, generate=fail)
    with sessions.begin() as db:
        assert not db.query(ThemeDevelopmentObservation).one().superseded
        retry = db.query(ThemeDevelopmentWork).filter_by(status="retry").one()
        retry.next_attempt_at = NOW - timedelta(minutes=1)
    assert process_one(sessions, generate=lambda *args: [])
    with sessions() as db:
        assert db.query(ThemeDevelopmentObservation).one().superseded
        assert theme_developments(theme_id, limit=1, db=db)["event_count"] == 0


def test_stale_model_result_is_discarded(sessions):
    with sessions.begin() as db:
        item, _a, _b = seed(db)
        enqueue(db, item.id, "technical")

    def changed(pipeline, db, bundle):
        values = facts(bundle)
        bundle["item"].content += " revised during model call"
        db.commit()
        return values

    process_one(sessions, generate=changed)
    with sessions() as db:
        assert db.query(ThemeDevelopmentObservation).count() == 0
        assert (
            db.query(ThemeDevelopmentWork).filter_by(status="superseded").count() == 1
        )


def test_expired_final_attempt_becomes_visible_failure(sessions):
    with sessions.begin() as db:
        item, _a, _b = seed(db)
        row = enqueue(db, item.id, "technical")
        row.status = "processing"
        row.attempts = 3
        row.lease_until = NOW - timedelta(minutes=1)
    assert not process_one(sessions)
    with sessions() as db:
        assert db.query(ThemeDevelopmentWork).one().status == "failed"


def test_target_migration_round_trip_preserves_existing_tables():
    engine = create_engine("sqlite:///:memory:")
    path = (
        Path(__file__).parents[2]
        / "alembic/versions/20260912_0043_theme_intelligence.py"
    )
    spec = importlib.util.spec_from_file_location("intelligence_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with engine.begin() as connection:
        for table in ["theme_metrics", "theme_clusters", "content_items"]:
            connection.execute(text(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY)"))
        module.op = Operations(MigrationContext.configure(connection))
        module.upgrade()
        assert "theme_development_themes" in inspect(connection).get_table_names()
        module.downgrade()
        assert inspect(connection).get_table_names() == [
            "content_items",
            "theme_clusters",
            "theme_metrics",
        ]
        module.upgrade()


def test_group_api_preview_apply_search_undo_and_conflict(sessions, monkeypatch):
    from app.api.v1 import themes_intelligence as api
    from fastapi import HTTPException

    with sessions() as db:
        _, a, b = seed(db)
        preview = api.preview_equivalence(a.id, b.id, db)
        request = api.GroupRequest(
            source_id=a.id,
            target_id=b.id,
            reason="Equivalent exposure",
            operation_key="api-test",
            expected_version=preview["version"],
        )
        result = api.apply_equivalence(request, db, x_admin_actor="Reviewer")
        assert result["refresh_status"] == "pending"
        choices = api.search_equivalent_themes(db, q="CPO", pipeline="technical")[
            "themes"
        ]
        assert [row["id"] for row in choices] == [b.id]
        assert api.apply_equivalence(request, db, x_admin_actor="Reviewer")["id"] == result["id"]
        assert api.equivalence_history(db, pipeline="technical")["operations"][0][
            "active"
        ]
        with pytest.raises(HTTPException) as error:
            api.apply_equivalence(
                request.model_copy(update={"operation_key": "stale"}), db, x_admin_actor="Reviewer"
            )
        assert error.value.status_code == 409
        api.undo_equivalence(
            result["id"], api.UndoRequest(reason="Separate again"), db, x_admin_actor="Undo Reviewer"
        )
        operation = api.equivalence_history(db, pipeline="technical")["operations"][0]
        assert operation["undone_by"] == "Undo Reviewer"
        assert ThemeEquivalenceService(db).members(a.id) == [a.id]


def test_disabled_backfill_is_read_only_and_rejects_model_work(sessions, monkeypatch):
    from app.api.v1.themes_intelligence import BackfillRequest, backfill_developments
    from fastapi import HTTPException

    monkeypatch.delenv("THEME_DEVELOPMENT_TRACKING_ENABLED", raising=False)
    with sessions() as db:
        item, _, _ = seed(db)
        assert (
            backfill_developments(BackfillRequest(item_ids=[item.id]), db)[
                "model_calls_enabled"
            ]
            is False
        )
        assert db.query(ThemeDevelopmentWork).count() == 0
        with pytest.raises(HTTPException) as error:
            backfill_developments(BackfillRequest(item_ids=[item.id], apply=True), db)
        assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_equivalence_mutations_require_admin_key(sessions, monkeypatch):
    import httpx
    from app.api.v1.config import settings as config_settings
    from app.database import get_db
    from app.main import app
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    monkeypatch.setattr(config_settings, "admin_api_key", "review-secret")
    with sessions() as db:
        app.dependency_overrides[get_db] = lambda: db
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/api/v1/themes/equivalence",
                    json={
                        "source_id": 1,
                        "target_id": 2,
                        "reason": "Equivalent",
                        "operation_key": "auth-test",
                        "expected_version": "a" * 64,
                    },
                )
                assert response.status_code == 401
        finally:
            app.dependency_overrides.pop(get_db, None)


def test_grouped_timeline_deduplicates_events_and_retains_alias_provenance(sessions):
    with sessions.begin() as db:
        item, a, b = seed(db)
        first_id, second_id = a.id, b.id
        operation = ThemeEquivalenceService(db).apply(
            a.id, b.id, actor="review", reason="Equivalent", key="timeline"
        )
        operation_id = operation["id"]
        enqueue(db, item.id, "technical")
    process_one(sessions, generate=lambda p, db, bundle: facts(bundle))
    with sessions.begin() as db:
        assert theme_developments(second_id, db, limit=1)["event_count"] == 1
        assert db.query(ThemeDevelopmentObservation).one().theme_ids == [first_id]
        ThemeEquivalenceService(db).undo(
            operation_id, actor="review", reason="Separate"
        )
        assert theme_developments(second_id, db, limit=1)["event_count"] == 0
        assert theme_developments(first_id, db, limit=1)["event_count"] == 1
        db.query(ContentPipelineEligibility).delete()
        assert theme_developments(first_id, db, limit=1)["observations"] == []


def test_category_view_hides_alias_children_and_clears_emptied_parent(sessions):
    from app.models.theme import ThemeMetrics
    from app.services.theme_taxonomy_service import ThemeTaxonomyService

    with sessions.begin() as db:
        _, a, b = seed(db)
        parents = [
            ThemeCluster(
                name=name,
                display_name=name,
                canonical_key=name,
                pipeline="technical",
                is_active=True,
                is_l1=True,
            )
            for name in ["parent-a", "parent-b", "unrelated-parent"]
        ]
        db.add_all(parents)
        db.flush()
        a.parent_cluster_id, b.parent_cluster_id = [row.id for row in parents[:2]]
        for theme in [a, b, *parents]:
            db.add(
                ThemeMetrics(
                    theme_cluster_id=theme.id,
                    pipeline="technical",
                    date=NOW.date(),
                    mentions_7d=1,
                    mentions_30d=1,
                )
            )
        db.flush()
        ThemeEquivalenceService(db).apply(
            a.id, b.id, actor="reviewer", reason="Same exposure", key="category"
        )
        taxonomy = ThemeTaxonomyService(db)
        assert taxonomy.get_l1_with_children(parents[0].id)["total_children"] == 0
        assert taxonomy.get_l1_with_children(parents[1].id)["total_children"] == 1
        taxonomy.compute_all_l1_metrics(as_of_date=NOW)
        assert (
            db.query(ThemeMetrics)
            .filter_by(theme_cluster_id=parents[0].id)
            .one()
            .mentions_7d
            == 0
        )
        target = db.query(ThemeMetrics).filter_by(theme_cluster_id=parents[1].id).one()
        assert target.mentions_7d == 1 and target.num_constituents == 1
        unrelated = (
            db.query(ThemeMetrics).filter_by(theme_cluster_id=parents[2].id).one()
        )
        assert unrelated.mentions_7d == 1


def test_model_preparation_reuses_route_and_existing_event_hints(sessions, monkeypatch):
    import json

    from app.services.theme_development_preparation import generate_facts, input_bundle
    from app.services.theme_extraction_service import ThemeExtractionService

    with sessions.begin() as db:
        item, _, _ = seed(db)
        item.translated_content = "Prepared English evidence"
        item_id = item.id
        enqueue(db, item.id, "technical")
    process_one(sessions, generate=lambda p, db, bundle: facts(bundle))
    prompts = []

    def model(self, prompt, *, system_prompt):
        prompts.append(json.loads(prompt))
        return "[]"

    rate_limits = []
    monkeypatch.setattr(ThemeExtractionService, "_rate_limit", lambda self: rate_limits.append(True))
    monkeypatch.setattr(ThemeExtractionService, "_try_generate_litellm", model)
    with sessions() as db:
        bundle = input_bundle(db, item_id, "technical")
        assert generate_facts("technical", db, bundle) == []
        assert (
            "Prepared English evidence" in prompts[0]["sources"]["translated_primary"]
        )
        assert len(prompts[0]["known_event_identities"]) == 1
        assert rate_limits == [True]


def test_grouped_detail_aggregates_alias_relationships(sessions):
    from app.api.v1.themes_queries import get_theme_detail

    with sessions.begin() as db:
        _, alias, representative = seed(db)
        alias.first_seen_at = representative.first_seen_at = NOW
        peer = ThemeCluster(
            name="Datacenter Networking",
            display_name="Datacenter Networking",
            canonical_key="datacenter_networking",
            pipeline="technical",
            is_active=True,
        )
        db.add(peer)
        db.flush()
        db.add(
            ThemeRelationship(
                source_cluster_id=alias.id,
                target_cluster_id=peer.id,
                relationship_type="related",
                pipeline="technical",
                confidence=0.8,
                is_active=True,
            )
        )
        ThemeEquivalenceService(db).apply(
            alias.id,
            representative.id,
            actor="reviewer",
            reason="Equivalent",
            key="relationship-group",
        )

        detail = get_theme_detail(alias.id, db)
        assert [row.peer_theme_id for row in detail.relationships] == [peer.id]


def test_emerging_group_includes_alias_constituents(sessions, monkeypatch):
    with sessions.begin() as db:
        _, alias, representative = seed(db)
        alias.first_seen_at = representative.first_seen_at = NOW
        db.query(ThemeConstituent).filter_by(theme_cluster_id=alias.id).update(
            {"symbol": "MRVL"}
        )
        ThemeEquivalenceService(db).apply(
            alias.id,
            representative.id,
            actor="reviewer",
            reason="Equivalent",
            key="emerging-group",
        )
        service = ThemeDiscoveryService(db)
        monkeypatch.setattr(
            service,
            "calculate_mention_metrics",
            lambda _theme_id: {
                "mentions_7d": 3,
                "mention_velocity": 2.0,
                "sentiment_score": 0.5,
            },
        )
        monkeypatch.setattr(service, "_passes_emerging_lifecycle_gate", lambda **_kw: True)

        rows = service.discover_emerging_themes(min_velocity=1, min_mentions=1)
        assert set(rows[0]["tickers"]) == {"MRVL", "NBIS"}


def test_source_and_history_queries_use_the_same_representative(sessions):
    from app.api.v1.themes_queries import get_theme_history, get_theme_mentions
    from app.models.theme import ThemeMetrics

    with sessions.begin() as db:
        _, a, b = seed(db)
        ThemeEquivalenceService(db).apply(
            a.id, b.id, actor="review", reason="Same exposure", key="reads"
        )
        db.add(
            ThemeMetrics(
                theme_cluster_id=b.id,
                pipeline="technical",
                date=NOW.date(),
                mentions_7d=1,
                grouping_version=ThemeEquivalenceService(db).version("technical"),
            )
        )
        db.flush()
        mentions = get_theme_mentions(a.id, limit=50, db=db)
        assert mentions.theme_id == b.id and len(mentions.mentions) == 1
        history = get_theme_history(a.id, days=30, db=db)
        assert history["theme"] == b.display_name
        assert (
            len(history["history"]) == 1 and history["history"][0]["grouping_version"]
        )


def test_changed_evidence_is_requeued_without_new_mentions(sessions):
    with sessions.begin() as db:
        item, _, _ = seed(db)
        enqueue(db, item.id, "technical")

    def changed(pipeline, db, bundle):
        values = facts(bundle)
        bundle["item"].translated_content = "Updated English evidence"
        db.commit()
        return values

    process_one(sessions, generate=changed)
    with sessions() as db:
        assert (
            db.query(ThemeDevelopmentWork).filter_by(status="superseded").count() == 1
        )
        assert db.query(ThemeDevelopmentWork).filter_by(status="pending").count() == 1


def test_development_enqueue_failure_preserves_extracted_mentions(
    sessions, monkeypatch
):
    from types import SimpleNamespace

    from app.services import theme_development_worker
    from app.services.theme_extraction_service import ThemeExtractionService
    from app.services.theme_taxonomy_service import ThemeTaxonomyService
    from app.tasks import theme_intelligence_tasks
    from sqlalchemy.exc import SQLAlchemyError

    with sessions.begin() as db:
        item, cluster, _ = seed(db)
        service = ThemeExtractionService.__new__(ThemeExtractionService)
        service.db = db
        service.pipeline = "technical"
        service.last_grounding_context = None
        service.extract_from_content = lambda *_args, **_kwargs: [
            {
                "theme": "CPO",
                "tickers": ["MRVL"],
                "sentiment": "bullish",
                "confidence": 0.9,
                "excerpt": "CPO demand",
                "development": "Demand increased",
                "claim_support": None,
            }
        ]
        decision = SimpleNamespace(
            method="exact",
            score=1.0,
            threshold=0.9,
            threshold_version="test",
            score_model=None,
            score_model_version=None,
            fallback_reason=None,
            best_alternative_cluster_id=None,
            best_alternative_score=None,
            score_margin=None,
        )
        service._resolve_cluster_match = lambda *_args, **_kwargs: (cluster, decision)
        service._update_theme_constituents = lambda *_args: None
        monkeypatch.setattr(ThemeTaxonomyService, "classify_new_l2_to_l1", lambda *_args: None)
        monkeypatch.setattr(theme_intelligence_tasks, "tracking_enabled", lambda: True)
        monkeypatch.setattr(
            theme_development_worker,
            "enqueue",
            lambda *_args: (_ for _ in ()).throw(SQLAlchemyError("queue unavailable")),
        )

        assert service._extract_and_store_mentions(item) == 1
        assert db.query(ThemeMention).filter_by(content_item_id=item.id).count() == 1
        assert db.execute(text("SELECT 1")).scalar_one() == 1


def test_bounded_discovery_rotates_completed_work_and_notices_translation(sessions):
    from app.services.theme_development_worker import discover

    with sessions.begin() as db:
        item, a, _ = seed(db)
        first_id = item.id
        # A second eligible source reuses the same theme.
        second = ContentItem(
            source_id=item.source_id,
            source_type="news",
            content="Second report",
            fetched_at=NOW,
        )
        db.add(second)
        db.flush()
        second_id = second.id
        db.add(
            ContentPipelineEligibility(
                content_item_id=second.id,
                pipeline="technical",
                channel="legacy",
                originating_source_id=item.source_id,
                observed_at=NOW,
            )
        )
        db.add(
            ThemeMention(
                content_item_id=second.id,
                theme_cluster_id=a.id,
                pipeline="technical",
                source_type="news",
                raw_theme="CPO",
            )
        )
        assert discover(db, limit=1) == 1
        db.query(ThemeDevelopmentWork).one().status = "complete"
    with sessions.begin() as db:
        assert discover(db, limit=1) == 1
        assert (
            db.query(ThemeDevelopmentWork)
            .filter_by(content_item_id=second_id)
            .one()
            .status
            == "pending"
        )
        db.get(ContentItem, first_id).translated_content = "Updated English evidence"
    with sessions.begin() as db:
        assert discover(db, limit=1) == 1
        assert (
            db.query(ThemeDevelopmentWork)
            .filter_by(content_item_id=first_id, status="pending")
            .count()
            == 1
        )
        assert (
            db.query(ThemeDevelopmentWork)
            .filter_by(content_item_id=first_id, status="superseded")
            .count()
            == 1
        )
