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
from app.models.economic_taxonomy_runtime import EvidencePacket as EvidencePacketLineage
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
    taxonomy, themes = _taxonomy_themes(db, ("Copper Miners",), mode=mode)
    return taxonomy, themes[0]


def _taxonomy_themes(db, names, *, mode="economic"):
    repo = EconomicTaxonomyRepository(db)
    draft = repo.create_draft(actor="test:author", reason="development fixture")
    seed_initial_dimensions(repo, draft.id, actor="test:author")
    themes = [
        repo.create_theme(
            draft.id,
            display_name=name,
            definition=f"{name} exposure.",
            mechanism="Mine economics",
            lifecycle="established",
            lifecycle_policy_version="lifecycle-v1",
            actor="test:author",
        )
        for name in names
    ]
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
    return taxonomy, themes


def _classify(db, admitted, taxonomy, theme_ids, *, resolver="resolver-v1", created_at=None):
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
        **({"created_at": created_at} if created_at is not None else {}),
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
    monkeypatch.setattr(theme_discovery_tasks, "_theme_automation_gate_result", lambda _db, **_kw: None)
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


def _serving_generation(db, taxonomy, admitted, *, override_id=None, selected_attempt_id=None):
    """A minimal serving generation whose manifest selection carries an override."""
    from app.models.economic_taxonomy_runtime import (
        GenerationInputManifest,
        InterpretationSet,
        MetricsRevision,
        ReaderCapabilityManifest,
        ReaderSnapshotBundle,
        ServingGeneration,
    )

    manifest = GenerationInputManifest(
        status="unsealed",
        semantic_invalidation_revision=0,
        committed_revision_tuples=[],
        selections=[
            {
                "lineage": str(admitted.source_lineage_id),
                "evidence_packet_id": str(admitted.packet_id),
                "selected_attempt_id": str(selected_attempt_id) if selected_attempt_id else None,
                "interpretation_override_revision_id": str(override_id) if override_id else None,
            }
        ],
        created_by="test:publisher",
    )
    db.add(manifest)
    db.flush()
    manifest.seal(semantic_hash="manifest", artifact_integrity_hash="manifest-artifact")
    interpretation = InterpretationSet(
        status="unsealed", generation_input_manifest_id=manifest.id, created_by="test:publisher"
    )
    db.add(interpretation)
    db.flush()
    interpretation.seal(semantic_hash="interpretation", artifact_integrity_hash="artifact")
    metrics = MetricsRevision(
        status="unsealed",
        interpretation_set_id=interpretation.id,
        generation_input_manifest_id=manifest.id,
        formula_version="metrics-v1",
        as_of=datetime.now(timezone.utc),
        created_by="test:publisher",
    )
    snapshots = ReaderSnapshotBundle(
        status="unsealed", generation_input_manifest_id=manifest.id, payload={}, created_by="test:publisher"
    )
    capability = ReaderCapabilityManifest(
        backend_contract=1,
        frontend_contract=1,
        migration_version="0049",
        consumer_test_hash="tests",
        verified_by="test:publisher",
    )
    db.add_all([metrics, snapshots, capability])
    db.flush()
    metrics.seal(semantic_hash="metrics", artifact_integrity_hash="metrics-artifact")
    snapshots.seal(semantic_hash="snapshots", artifact_integrity_hash="snapshot-artifact")
    generation = ServingGeneration(
        taxonomy_version_id=taxonomy.id,
        interpretation_set_id=interpretation.id,
        metrics_revision_id=metrics.id,
        generation_input_manifest_id=manifest.id,
        reader_snapshot_bundle_id=snapshots.id,
        reader_capability_manifest_id=capability.id,
        semantic_hash="generation",
        artifact_integrity_hash="generation-artifact",
        created_by="test:publisher",
    )
    db.add(generation)
    db.flush()
    from app.models.economic_taxonomy_runtime import ServingGenerationEvent

    db.add(
        ServingGenerationEvent(
            serving_generation_id=generation.id,
            sequence_number=1,
            event_type="published",
            actor="test:publisher",
            details={},
        )
    )
    db.get(TaxonomyAuthority, 1).serving_generation_id = generation.id
    db.commit()


def test_bundle_follows_the_reviewer_override_the_serving_generation_carries(db_session):
    from app.domain.economic_taxonomy.contracts import AdminPrincipal
    from app.services.economic_taxonomy_interpretations import (
        EconomicTaxonomyInterpretationService,
        create_interpretation_override,
    )

    taxonomy, (miners, smelters) = _taxonomy_themes(db_session, ("Copper Miners", "Copper Smelters"))
    item, admitted = _admit_item(db_session)
    first = _classify(db_session, admitted, taxonomy, [miners.id])
    second = _classify(db_session, admitted, taxonomy, [smelters.id], resolver="resolver-v2")
    default = EconomicTaxonomyInterpretationService(None).default_attempt(
        db_session, admitted.source_lineage_id
    )
    chosen = second if default.id == first.id else first
    override = create_interpretation_override(
        db_session,
        source_lineage_id=admitted.source_lineage_id,
        selected_attempt_id=chosen.id,
        reason="Reviewer prefers this reading",
        principal=AdminPrincipal(
            subject="reviewer", auth_method="api_key", roles=frozenset({"taxonomy:review"})
        ),
    )
    _serving_generation(
        db_session, taxonomy, admitted, override_id=override.id, selected_attempt_id=chosen.id
    )

    bundle = development_bundle(db_session, item.id, "fundamental")

    expected = miners if chosen.id == first.id else smelters
    assert list(bundle["economic_refs"].values()) == [expected.id]


def test_completed_empty_classification_is_ready_but_unclassified_is_not(db_session):
    taxonomy, _theme = _taxonomy(db_session)
    pending, _ = _admit_item(db_session)
    empty, admitted = _admit_item(db_session)
    _classify(db_session, admitted, taxonomy, [])

    assert development_bundle(db_session, pending.id, "fundamental")["ready"] is False
    bundle = development_bundle(db_session, empty.id, "fundamental")
    assert bundle["ready"] is True
    assert bundle["theme_ids"] == []


def test_a_reclassification_to_no_themes_records_the_empty_revision(db_session, monkeypatch):
    # It supersedes the observations still linked to the removed themes.
    taxonomy, _theme = _taxonomy(db_session)
    item, admitted = _admit_item(db_session)
    _classify(db_session, admitted, taxonomy, [])
    worker.enqueue(db_session, item.id, "fundamental")
    db_session.commit()
    recorded = []
    monkeypatch.setattr(worker, "record_developments", lambda *args, **kwargs: recorded.append(kwargs))

    worker.process_one(sessionmaker(bind=db_session.get_bind()), generate=lambda *_args: [])

    assert len(recorded) == 1


def test_discovery_admits_a_completed_empty_first_classification(db_session):
    # Its empty revision must be recorded to supersede legacy observations an
    # item may carry from before cutover.
    taxonomy, _theme = _taxonomy(db_session)
    item, admitted = _admit_item(db_session)
    _classify(db_session, admitted, taxonomy, [])

    worker.discover(db_session)
    db_session.commit()

    assert [row.pipeline for row in _work(db_session, item.id)] == ["fundamental"]


def test_economic_observations_belong_to_the_lineage_source_family(db_session):
    # Publication pins development observations by the lineage's source family;
    # a content-item-keyed observation would never reach a generation.
    from app.models.economic_taxonomy_runtime import SourceLineage

    item, _theme = _economic_item(db_session)
    worker.discover(db_session, item_ids=[item.id])
    db_session.commit()

    worker.process_one(sessionmaker(bind=db_session.get_bind()), generate=lambda *_args: [_fact(1)])

    observation = db_session.scalar(
        select(ThemeDevelopmentObservation).where(
            ThemeDevelopmentObservation.content_item_id == item.id
        )
    )
    family = db_session.scalar(
        select(SourceLineage.source_family_id).where(
            SourceLineage.id
            == select(EvidencePacketLineage.source_lineage_id)
            .where(EvidencePacketLineage.capture_route == CONTENT_INGESTION_ROUTE)
            .order_by(EvidencePacketLineage.created_at.desc())
            .limit(1)
            .scalar_subquery()
        )
    )
    assert observation.source_family_id == family


def test_economic_gate_does_not_require_legacy_content_sources(db_session, monkeypatch):
    # A deployment fed only by Social or other directly admitted evidence has
    # no active legacy content source; the economic producer must still run.
    from app.tasks import theme_discovery_tasks, theme_intelligence_tasks

    item, _theme = _economic_item(db_session)
    monkeypatch.setenv("THEME_DEVELOPMENT_TRACKING_ENABLED", "true")
    monkeypatch.setattr(theme_discovery_tasks.settings, "feature_themes", True)
    monkeypatch.setattr(
        theme_discovery_tasks,
        "get_runtime_bootstrap_status",
        lambda _db: type("Status", (), {"bootstrap_required": False, "bootstrap_state": "ready"})(),
    )
    monkeypatch.setattr(worker, "process_one", lambda _sessions: False)

    result = theme_intelligence_tasks.prepare_developments()

    assert result["status"] == "processed"
    db_session.expire_all()
    assert [row.pipeline for row in _work(db_session, item.id)] == ["fundamental"]


def test_channels_come_from_the_classified_packet_not_a_newer_unclassified_one(db_session):
    # A newer capture (technical-only, not yet classified) supersedes the
    # classified fundamental packet. As publication does, the themes keep the
    # eligibility of the packet they were classified from.
    taxonomy, theme = _taxonomy(db_session)
    item, admitted = _admit_item(db_session, channels=("fundamental",))
    _classify(db_session, admitted, taxonomy, [theme.id])
    now = datetime.now(timezone.utc)
    EconomicSourceAdmissionService(db_session).admit_content(
        EvidenceAdmission(
            provider="news",
            canonical_source_family=content_family_key("news", item.external_id, item.url),
            capture_route="social",
            original_text=TEXT + " Updated.",
            preparation_version="social-v1",
            source_metadata={"social_work_id": 8},
            captured_at=now,
            available_at=now,
            evidence_channels=("technical",),
            supersedes_packet_id=admitted.packet_id,
        )
    )
    db_session.commit()

    fundamental = development_bundle(db_session, item.id, "fundamental")
    technical = development_bundle(db_session, item.id, "technical")
    assert fundamental["economic_refs"] == {1: theme.id}
    # Classified, but the classified packet is not technical-eligible.
    assert technical["ready"] is True
    assert technical["theme_ids"] == []

    worker.discover(db_session)
    db_session.commit()
    assert "fundamental" in {row.pipeline for row in _work(db_session, item.id)}


def test_work_finished_as_not_ready_is_redone_once_classification_completes(db_session):
    # Unclassified and completed-empty both offer no themes; the revision must
    # still differ, or the completed not-ready row hides the ready bundle.
    taxonomy, _theme = _taxonomy(db_session)
    item, admitted = _admit_item(db_session)
    worker.enqueue(db_session, item.id, "fundamental")
    db_session.commit()
    worker.process_one(sessionmaker(bind=db_session.get_bind()), generate=lambda *_args: [])
    db_session.expire_all()
    assert [row.status for row in _work(db_session, item.id)] == ["complete"]

    _classify(db_session, admitted, taxonomy, [])
    worker.enqueue(db_session, item.id, "fundamental")
    db_session.commit()

    assert "pending" in {row.status for row in _work(db_session, item.id)}


def test_a_recent_reviewer_override_brings_an_old_item_back_into_discovery(db_session):
    from datetime import timedelta

    taxonomy, theme = _taxonomy(db_session)
    item, admitted = _admit_item(db_session)
    old = datetime.now(timezone.utc) - timedelta(days=5)
    attempt = _classify(db_session, admitted, taxonomy, [theme.id], created_at=old)
    item.fetched_at = old
    db_session.commit()
    worker.discover(db_session)
    db_session.commit()
    assert _work(db_session, item.id) == []  # too old for automatic discovery

    # Created long ago (as create_interpretation_override writes it), but only
    # now published in the serving generation: rediscovery follows publication.
    from app.models.economic_taxonomy_runtime import InterpretationOverrideRevision

    override = InterpretationOverrideRevision(
        source_lineage_id=admitted.source_lineage_id,
        revision_number=1,
        override_kind="select_attempt",
        payload={
            "selected_attempt_id": str(attempt.id),
            "authenticated": True,
            "auth_method": "api_key",
            "roles": ["taxonomy:review"],
        },
        reason="Reviewer confirms this reading",
        created_by="reviewer",
        created_at=old,
    )
    db_session.add(override)
    db_session.commit()
    worker.discover(db_session)
    db_session.commit()
    assert _work(db_session, item.id) == []  # not serving yet: nothing changed for users

    _serving_generation(
        db_session, taxonomy, admitted, override_id=override.id, selected_attempt_id=attempt.id
    )
    worker.discover(db_session)
    db_session.commit()

    assert [row.pipeline for row in _work(db_session, item.id)] == ["fundamental"]


def test_prompt_uses_the_classified_packet_text_not_the_stale_item_text(db_session):
    # A later capture in force (here Social's, superseding the ingestion packet)
    # carries new text while the ContentItem keeps the old one. Classification
    # ran on the new text, so the prompt must too, and it changes the revision.
    taxonomy, theme = _taxonomy(db_session)
    item, admitted = _admit_item(db_session)
    _classify(db_session, admitted, taxonomy, [theme.id])
    before = development_bundle(db_session, item.id, "fundamental")
    edited = TEXT + " The contract was later extended to 12 years."
    now = datetime.now(timezone.utc)
    corrected = EconomicSourceAdmissionService(db_session).admit_content(
        EvidenceAdmission(
            provider="news",
            canonical_source_family=content_family_key("news", item.external_id, item.url),
            capture_route="social",
            original_text=edited,
            preparation_version="social-v1",
            source_metadata={"social_work_id": 9},
            captured_at=now,
            available_at=now,
            evidence_channels=("fundamental",),
            supersedes_packet_id=admitted.packet_id,
        )
    )
    db_session.commit()
    _classify(db_session, corrected, taxonomy, [theme.id], resolver="resolver-v2")

    after = development_bundle(db_session, item.id, "fundamental")

    assert "extended to 12 years" in after["sources"]["primary"]
    assert item.content == TEXT  # the item itself was not updated
    assert after["revision"] != before["revision"]


def test_a_channel_the_selected_evidence_lost_is_ready_with_no_themes(db_session):
    # The older packet was technical-eligible; the selected (newer, classified)
    # packet is fundamental-only. The technical channel must record an empty
    # revision to supersede its earlier observations, not wait as "not ready".
    taxonomy, theme = _taxonomy(db_session)
    item, admitted = _admit_item(db_session, channels=("technical",))
    _classify(db_session, admitted, taxonomy, [theme.id])
    now = datetime.now(timezone.utc)
    newer = EconomicSourceAdmissionService(db_session).admit_content(
        EvidenceAdmission(
            provider="news",
            canonical_source_family=content_family_key("news", item.external_id, item.url),
            capture_route="social",
            original_text=TEXT,
            preparation_version="social-v1",
            source_metadata={"social_work_id": 10},
            captured_at=now,
            available_at=now,
            evidence_channels=("fundamental",),
            supersedes_packet_id=admitted.packet_id,
        )
    )
    db_session.commit()
    _classify(db_session, newer, taxonomy, [theme.id], resolver="resolver-v2")

    technical = development_bundle(db_session, item.id, "technical")

    assert technical["ready"] is True
    assert technical["theme_ids"] == []


def test_discovery_offers_a_channel_that_still_has_active_observations(db_session):
    # Eligibility for "technical" was removed from the packet, but technical
    # observations recorded earlier are still active: the channel must be
    # offered so its empty revision supersedes them.
    taxonomy, theme = _taxonomy(db_session)
    item, admitted = _admit_item(db_session, channels=("technical",))
    _classify(db_session, admitted, taxonomy, [theme.id])
    worker.discover(db_session, item_ids=[item.id])
    db_session.commit()
    worker.process_one(sessionmaker(bind=db_session.get_bind()), generate=lambda *_args: [_fact(1)])
    EconomicSourceAdmissionService(db_session).revise_lens_eligibility(
        admitted.packet_id, evidence_channels=("fundamental",), reason="technical withdrawn"
    )
    db_session.commit()

    pairs = worker._economic_candidates(db_session, [item.id])  # noqa: SLF001

    assert (item.id, "technical") in pairs


def test_explicit_backfill_only_queries_the_requested_items_lineages():
    from uuid import UUID

    lineage = UUID(int=7)
    sql = str(
        worker._economic_candidate_query(item_ids=[1], lineage_ids=[lineage]).compile(  # noqa: SLF001
            dialect=postgresql.dialect()
        )
    )
    assert "source_lineage_id IN" in sql
