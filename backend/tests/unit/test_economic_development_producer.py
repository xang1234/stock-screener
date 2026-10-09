"""Developments are produced from economic evidence under economic authority (#513)."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import sessionmaker

from app.infra.db.repositories.economic_taxonomy_repo import EconomicTaxonomyRepository
from app.infra.db.repositories.economic_taxonomy_work_repo import (
    EconomicTaxonomyWorkRepository,
)
from app.models.economic_taxonomy_runtime import (
    ClaimAssignment,
    ClaimReviewArtifact,
    ClassificationAttempt,
    ClassificationAttemptEvent,
    ExtractionArtifact,
    TaxonomyAuthority,
)
from app.models.theme import ContentItem
from app.models.theme_intelligence import (
    EconomicThemeDevelopment,
    ThemeDevelopmentObservation,
    ThemeDevelopmentTheme,
    ThemeDevelopmentWork,
)
from app.services import theme_development_worker as worker
from app.services.economic_source_admission import (
    CONTENT_INGESTION_ROUTE,
    EconomicSourceAdmissionService,
    EvidenceAdmission,
    content_family_key,
    content_route_record_id,
)
from app.services.economic_taxonomy_seed import seed_initial_dimensions
from app.services.theme_development_preparation import development_bundle

TEXT = "Freeport-McMoRan said it won a 10-year copper supply contract with Acme Wire."


def _taxonomy(db, *, mode="economic"):
    repo = EconomicTaxonomyRepository(db)
    draft = repo.create_draft(actor="test:author", reason="development fixture")
    seed_initial_dimensions(repo, draft.id, actor="test:author")
    theme = repo.create_theme(
        draft.id,
        display_name="Copper Miners",
        definition="Copper mining exposure.",
        mechanism="Mine economics",
        lifecycle="established",
        lifecycle_policy_version="lifecycle-v1",
        actor="test:author",
    )
    taxonomy = repo.seal_draft(draft.id)
    db.merge(
        TaxonomyAuthority(
            id=1,
            mode=mode,
            processing_taxonomy_version_id=taxonomy.id,
            processing_head_revision=1,
            authority_epoch=1,
            writes_fenced=False,
            rollback_state="ready",
        )
    )
    db.commit()
    return taxonomy, theme


def _classify(db, admitted, taxonomy, theme_ids, *, resolver="resolver-v1"):
    now = datetime.now(timezone.utc)
    request = EconomicTaxonomyWorkRepository(db).enqueue_request(
        source_lineage_id=admitted.source_lineage_id,
        evidence_packet_id=admitted.packet_id,
        policy_bundle_version=f"bundle-{resolver}",
        available_at=now,
    )
    extraction = ExtractionArtifact(
        evidence_packet_id=admitted.packet_id,
        extraction_policy_version=f"extract-{resolver}",
        result_status="accepted_candidates",
        result_payload={},
        provider_response_hash=f"extract-{uuid4().hex}",
    )
    db.add(extraction)
    db.flush()
    review = ClaimReviewArtifact(
        extraction_artifact_id=extraction.id,
        claim_review_policy_version="review-v1",
        facet_catalog_semantic_hash="facet-v1",
        result_status="accepted_candidates",
        result_payload={},
        provider_response_hash=f"review-{uuid4().hex}",
    )
    db.add(review)
    db.flush()
    attempt = ClassificationAttempt(
        processing_request_id=request.id,
        claim_review_artifact_id=review.id,
        input_taxonomy_version_id=taxonomy.id,
        resolver_policy_version=resolver,
        naming_policy_version="naming-v1",
        derivation_policy_version="derive-v1",
        result_status="completed",
        result_payload={},
    )
    db.add(attempt)
    db.flush()
    db.add(
        ClassificationAttemptEvent(
            classification_attempt_id=attempt.id,
            sequence_number=1,
            event_type="completed",
            event_payload={},
        )
    )
    for theme_id in theme_ids:
        db.add(
            ClaimAssignment(
                classification_attempt_id=attempt.id,
                claim_fingerprint="copper-claim",
                economic_theme_id=theme_id,
                exposure_support="direct",
                claim_payload={},
                provenance={},
            )
        )
    db.commit()
    return attempt


def _admit_item(db, *, channels=("fundamental",)):
    now = datetime.now(timezone.utc)
    external_id = f"news-{uuid4().hex[:8]}"
    item = ContentItem(
        source_type="news",
        external_id=external_id,
        url=f"https://example.com/{external_id}",
        title="Copper contract",
        content=TEXT,
        published_at=now,
        fetched_at=now,
    )
    db.add(item)
    db.flush()
    admitted = EconomicSourceAdmissionService(db).admit_content(
        EvidenceAdmission(
            provider="news",
            canonical_source_family=content_family_key("news", external_id, item.url),
            capture_route=CONTENT_INGESTION_ROUTE,
            route_record_id=content_route_record_id(item.id, None),
            original_text=TEXT,
            preparation_version="content-ingestion-v1",
            source_metadata={"content_item_id": item.id},
            captured_at=now,
            available_at=now,
            evidence_channels=channels,
        )
    )
    db.commit()
    return item, admitted


def _economic_item(db, *, mode="economic"):
    """A fetched item admitted as economic evidence, classified into 'Copper Miners'."""
    taxonomy, theme = _taxonomy(db, mode=mode)
    item, admitted = _admit_item(db)
    _classify(db, admitted, taxonomy, [theme.id])
    return item, theme


def _fact(theme_ref):
    return {
        "theme_ids": [theme_ref],
        "actor": "Freeport-McMoRan",
        "action": "won",
        "object": "copper supply contract",
        "event_time": None,
        "reference": None,
        "status": "announced",
        "summary": "Freeport-McMoRan won a copper supply contract.",
        "quantities": [],
        "citations": [{"source_id": "primary", "quote": TEXT}],
    }


def _work(db, item_id):
    return db.scalars(
        select(ThemeDevelopmentWork).where(ThemeDevelopmentWork.content_item_id == item_id)
    ).all()


def test_economic_bundle_offers_the_classified_economic_themes(db_session):
    item, theme = _economic_item(db_session)

    bundle = development_bundle(db_session, item.id, "fundamental")

    assert bundle["kind"] == "economic"
    assert bundle["themes"] == {1: "Copper Miners"}
    assert bundle["economic_refs"] == {1: theme.id}
    assert bundle["theme_ids"] == [1]


def test_economic_bundle_ignores_a_channel_the_evidence_is_not_eligible_for(db_session):
    item, _theme = _economic_item(db_session)

    bundle = development_bundle(db_session, item.id, "technical")

    assert bundle["theme_ids"] == []


def test_reclassification_into_the_same_themes_keeps_the_revision(db_session):
    # A new attempt (e.g. a resolver bump) with the same themes must not re-run
    # the model and supersede the item's observations.
    taxonomy, theme = _taxonomy(db_session)
    item, admitted = _admit_item(db_session)
    _classify(db_session, admitted, taxonomy, [theme.id])
    before = development_bundle(db_session, item.id, "fundamental")["revision"]

    _classify(db_session, admitted, taxonomy, [theme.id], resolver="resolver-v2")

    assert development_bundle(db_session, item.id, "fundamental")["revision"] == before


def test_discovery_under_economic_authority_queues_only_eligible_channels(db_session):
    item, _theme = _economic_item(db_session)

    worker.discover(db_session, item_ids=[item.id])
    db_session.commit()

    assert [(row.pipeline, row.status) for row in _work(db_session, item.id)] == [
        ("fundamental", "pending")
    ]


def test_automatic_discovery_reaches_every_classified_item(db_session):
    # Repeated runs must rotate through all recent items, not re-pick the first.
    taxonomy, theme = _taxonomy(db_session)
    items = []
    for _ in range(12):
        item, admitted = _admit_item(db_session)
        _classify(db_session, admitted, taxonomy, [theme.id])
        items.append(item.id)

    for _ in range(4):
        worker.discover(db_session, limit=4)
        db_session.commit()

    queued = {
        row.content_item_id
        for row in db_session.scalars(
            select(ThemeDevelopmentWork).where(ThemeDevelopmentWork.content_item_id.in_(items))
        )
    }
    assert queued == set(items)


def test_candidate_query_does_not_compare_json_columns():
    # PostgreSQL cannot DISTINCT or compare its json type (source_metadata).
    sql = str(
        worker._economic_candidate_query(item_ids=None).compile(dialect=postgresql.dialect())  # noqa: SLF001
    )
    assert "DISTINCT" not in sql.upper()


def test_recording_under_economic_authority_writes_per_event_economic_links_only(db_session):
    item, theme = _economic_item(db_session)
    worker.discover(db_session, item_ids=[item.id])
    db_session.commit()
    sessions = sessionmaker(bind=db_session.get_bind())

    assert worker.process_one(sessions, generate=lambda *_args: [_fact(1)]) is True

    observation = db_session.scalar(
        select(ThemeDevelopmentObservation).where(
            ThemeDevelopmentObservation.content_item_id == item.id
        )
    )
    assert observation is not None
    links = db_session.scalars(
        select(EconomicThemeDevelopment).where(
            EconomicThemeDevelopment.observation_id == observation.id
        )
    ).all()
    assert [(link.economic_theme_id, link.link_origin) for link in links] == [
        (theme.id, "economic_native")
    ]
    assert db_session.scalars(
        select(ThemeDevelopmentTheme).where(
            ThemeDevelopmentTheme.observation_id == observation.id
        )
    ).all() == []


def test_unclassified_item_at_cutover_records_nothing(db_session, monkeypatch):
    # Legacy work pending at cutover, for an item economic evidence has not
    # classified yet. "Not ready" must not record an (empty) economic revision:
    # recording supersedes every other revision of the item and channel, which
    # would wipe its legacy development history.
    _taxonomy(db_session)
    item, _admitted = _admit_item(db_session)
    db_session.add(
        ThemeDevelopmentWork(
            content_item_id=item.id,
            pipeline="fundamental",
            revision="legacy-revision",
            checked_at=datetime.now(timezone.utc),
            status="pending",
            attempts=0,
        )
    )
    db_session.commit()
    recorded = []
    monkeypatch.setattr(worker, "record_developments", lambda *args, **kwargs: recorded.append(kwargs))
    sessions = sessionmaker(bind=db_session.get_bind())

    for _ in range(3):
        worker.process_one(sessions, generate=lambda *_args: [])

    assert recorded == []
    db_session.expire_all()
    assert "superseded" in {row.status for row in _work(db_session, item.id)}


def test_legacy_authority_keeps_the_legacy_bundle(db_session):
    item, _theme = _economic_item(db_session, mode="legacy")

    bundle = development_bundle(db_session, item.id, "fundamental")

    assert bundle["kind"] == "legacy"


def test_scheduled_task_runs_the_economic_producer_under_economic_authority(db_session, monkeypatch):
    # The real entry point: it used to skip on entry under economic authority.
    from app.tasks import theme_discovery_tasks, theme_intelligence_tasks

    item, _theme = _economic_item(db_session)
    monkeypatch.setenv("THEME_DEVELOPMENT_TRACKING_ENABLED", "true")
    monkeypatch.setattr(theme_discovery_tasks, "_theme_automation_gate_result", lambda _db: None)
    monkeypatch.setattr(worker, "process_one", lambda _sessions: False)

    result = theme_intelligence_tasks.prepare_developments()

    assert result["status"] == "processed"
    db_session.expire_all()
    assert [row.pipeline for row in _work(db_session, item.id)] == ["fundamental"]


def test_backfill_endpoint_admits_the_economic_producer(db_session):
    # It used to return 409 under economic authority (legacy writer guard).
    import pytest
    from fastapi import HTTPException

    from app.api.v1 import themes_intelligence

    _taxonomy(db_session)
    themes_intelligence.guard_development_backfill(db_session)  # economic: allowed

    db_session.get(TaxonomyAuthority, 1).writes_fenced = True
    db_session.commit()
    with pytest.raises(HTTPException):
        themes_intelligence.guard_development_backfill(db_session)

    route = next(
        r for r in themes_intelligence.router.routes if r.path.endswith("/developments/backfill")
    )
    guards = {dependency.call for dependency in route.dependant.dependencies}
    assert themes_intelligence.guard_development_backfill in guards


def test_discovery_finds_an_item_whose_effective_evidence_is_a_later_capture(db_session):
    # Social's capture of the same post supersedes the content-ingestion packet
    # (#500); classification lands on Social's packet, which has no content item.
    taxonomy, theme = _taxonomy(db_session)
    item, admitted = _admit_item(db_session)
    now = datetime.now(timezone.utc)
    social = EconomicSourceAdmissionService(db_session).admit_content(
        EvidenceAdmission(
            provider="news",
            canonical_source_family=content_family_key("news", item.external_id, item.url),
            capture_route="social",
            original_text=TEXT,
            preparation_version="social-v1",
            source_metadata={"social_work_id": 7},
            captured_at=now,
            available_at=now,
            evidence_channels=("fundamental",),
            supersedes_packet_id=admitted.packet_id,
        )
    )
    db_session.commit()
    _classify(db_session, social, taxonomy, [theme.id])

    worker.discover(db_session)
    db_session.commit()

    assert [row.pipeline for row in _work(db_session, item.id)] == ["fundamental"]
