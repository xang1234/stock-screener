"""Developments are produced from economic evidence under economic authority (#513)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.infra.db.repositories.economic_taxonomy_repo import EconomicTaxonomyRepository
from app.infra.db.repositories.economic_taxonomy_work_repo import (
    EconomicTaxonomyWorkRepository,
)
from app.models.economic_taxonomy_runtime import (
    ClaimAssignment,
    ClaimReviewArtifact,
    ClassificationAttempt,
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


def _economic_item(db, *, mode="economic"):
    """A fetched content item admitted as economic evidence and classified into
    the theme 'Copper Miners', with economic authority serving."""
    now = datetime.now(timezone.utc)
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
            evidence_channels=("fundamental",),
        )
    )
    request = EconomicTaxonomyWorkRepository(db).enqueue_request(
        source_lineage_id=admitted.source_lineage_id,
        evidence_packet_id=admitted.packet_id,
        policy_bundle_version="bundle-v1",
        available_at=now,
    )
    extraction = ExtractionArtifact(
        evidence_packet_id=admitted.packet_id,
        extraction_policy_version="extract-v1",
        result_status="accepted_candidates",
        result_payload={},
        provider_response_hash="extract",
    )
    db.add(extraction)
    db.flush()
    review = ClaimReviewArtifact(
        extraction_artifact_id=extraction.id,
        claim_review_policy_version="review-v1",
        facet_catalog_semantic_hash="facet-v1",
        result_status="accepted_candidates",
        result_payload={},
        provider_response_hash="review",
    )
    db.add(review)
    db.flush()
    attempt = ClassificationAttempt(
        processing_request_id=request.id,
        claim_review_artifact_id=review.id,
        input_taxonomy_version_id=taxonomy.id,
        resolver_policy_version="resolver-v1",
        naming_policy_version="naming-v1",
        derivation_policy_version="derive-v1",
        result_status="completed",
        result_payload={},
    )
    db.add(attempt)
    db.flush()
    db.add(
        ClaimAssignment(
            classification_attempt_id=attempt.id,
            claim_fingerprint="copper-claim",
            economic_theme_id=theme.id,
            exposure_support="direct",
            claim_payload={},
            provenance={},
        )
    )
    db.commit()
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


def test_discovery_under_economic_authority_queues_classified_items(db_session):
    item, _theme = _economic_item(db_session)

    worker.discover(db_session, item_ids=[item.id])
    db_session.commit()

    queued = db_session.scalars(
        select(ThemeDevelopmentWork).where(ThemeDevelopmentWork.content_item_id == item.id)
    ).all()
    assert [(row.pipeline, row.status) for row in queued] == [("fundamental", "pending")]


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


def test_legacy_authority_keeps_the_legacy_bundle(db_session):
    item, _theme = _economic_item(db_session, mode="legacy")

    bundle = development_bundle(db_session, item.id, "fundamental")

    assert bundle["kind"] == "legacy"
