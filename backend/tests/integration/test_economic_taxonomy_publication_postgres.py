from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Event
from uuid import UUID, uuid4

import pytest
from app.database import SessionLocal, engine
from app.domain.economic_taxonomy.contracts import AdminPrincipal
from app.infra.db.repositories.economic_taxonomy_publication_repo import (
    EconomicTaxonomyPublicationRepository,
)
from app.infra.db.repositories.economic_taxonomy_repo import EconomicTaxonomyRepository
from app.models.economic_taxonomy import TaxonomyVersion
from app.models.economic_taxonomy_runtime import (
    GenerationInputManifest,
    ReaderCapabilityManifest,
    ServingGenerationEvent,
    TaxonomyAuthority,
    TaxonomyBenchmarkResult,
    TaxonomyProjectionDeliveryEvent,
    TaxonomyProjectionEvent,
)
from app.services.economic_taxonomy_benchmark_store import (
    benchmark_anchor_version,
    register_verified_benchmark,
)
from app.services.economic_taxonomy_fence import (
    StaleAuthorityEpoch,
    producer_write,
)
from app.services.economic_taxonomy_publication import (
    CompatibilityProjection,
    EconomicTaxonomyPublicationCoordinator,
    InjectedPublicationCrash,
    ManifestChanged,
)
from app.services.economic_taxonomy_processor import SYSTEM_ACTOR
from app.services.economic_taxonomy_runtime import EconomicTaxonomyRuntimeService
from sqlalchemy import select, update
from sqlalchemy.exc import DBAPIError

pytestmark = pytest.mark.skipif(
    engine.dialect.name != "postgresql",
    reason="requires PostgreSQL transaction advisory locks",
)

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
ADMIN = AdminPrincipal(
    subject="admin:publication",
    auth_method="admin_api_key",
    roles=frozenset({"taxonomy:review"}),
)


@pytest.fixture
def seeded():
    with SessionLocal() as session:
        repository = EconomicTaxonomyRepository(session)
        draft = repository.create_draft(actor=ADMIN.subject, reason="pg publication")
        taxonomy = repository.seal_draft(draft.id)
        capability = ReaderCapabilityManifest(
            backend_contract=1,
            frontend_contract=1,
            migration_version="0055",
            consumer_test_hash=f"pg-publication-{uuid4()}",
            verified_by=ADMIN.subject,
        )
        session.add_all(
            [
                capability,
                TaxonomyAuthority(
                    id=1,
                    mode="dual",
                    processing_taxonomy_version_id=taxonomy.id,
                    processing_head_revision=1,
                    authority_epoch=7,
                    writes_fenced=False,
                    semantic_invalidation_revision=0,
                    cutover_catch_up_cursor=[],
                    rollback_state="ready",
                ),
            ]
        )
        register_verified_benchmark(
            session,
            report={
                "passed": True,
                "fixture_version": 1,
                "taxonomy_hash": taxonomy.semantic_hash,
                "policy_bundle": "economic-taxonomy-v1",
                "errors": [],
                "cases": [],
            },
            verified_by=ADMIN.subject,
        )
        session.commit()
        return taxonomy.id, capability.id


def _coordinator(**kwargs):
    return EconomicTaxonomyPublicationCoordinator(
        SessionLocal, clock=lambda: NOW, **kwargs
    )


def _compatibility(_session, context):
    return [
        CompatibilityProjection(
            source_lineage="post:1",
            projection_kind="legacy_theme",
            projection_version=1,
            target="legacy",
            payload={"themes": ["memory"]},
            origin_representation="economic",
            selected_interpretation_version=str(context.interpretation_set_id),
            mapping_version=str(context.taxonomy_version_id),
        )
    ]


def _prepare(coordinator, capability_id, *, target_mode="economic"):
    cutoff = coordinator.capture_cutoff(principal=ADMIN, selections=[])
    return cutoff, coordinator.prepare_generation(
        cutoff,
        principal=ADMIN,
        reader_capability_manifest_id=capability_id,
        target_mode=target_mode,
    )


def test_registered_benchmark_is_database_append_only(seeded):
    with SessionLocal() as session:
        result = session.scalar(select(TaxonomyBenchmarkResult))
        assert result is not None
        with pytest.raises(DBAPIError, match="runtime_payload_immutable"):
            session.execute(
                update(TaxonomyBenchmarkResult)
                .where(TaxonomyBenchmarkResult.id == result.id)
                .values(verified_by="admin:replacement")
            )
            session.commit()
        session.rollback()


def test_late_lower_id_is_in_cutoff_and_post_cutoff_work_is_backlog(seeded):
    _taxonomy_id, capability_id = seeded
    writer_entered = Event()
    release_writer = Event()
    cutoff_done = Event()
    captured = {}

    def late_lower_id_writer():
        with SessionLocal() as session:
            with producer_write(
                session, expected_epoch=7, allowed_modes={"dual"}
            ) as authority:
                EconomicTaxonomyPublicationRepository(session).append_source_revision(
                    revision_id=UUID("00000000-0000-0000-0000-000000000001"),
                    producer_kind="content",
                    logical_source_key="post:late-low-id",
                    revision_kind="evidence",
                    revision_number=1,
                    content_hash="late-low-id",
                    authority_epoch=authority.authority_epoch,
                )
                writer_entered.set()
                release_writer.wait(timeout=5)
            session.commit()

    def capture():
        captured["value"] = _coordinator().capture_cutoff(
            principal=ADMIN, selections=[]
        )
        cutoff_done.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(late_lower_id_writer)
        assert writer_entered.wait(timeout=5)
        capturer = pool.submit(capture)
        assert not cutoff_done.wait(timeout=0.2)
        release_writer.set()
        writer.result(timeout=5)
        capturer.result(timeout=5)

    cutoff = captured["value"]
    assert any(row[1] == "post:late-low-id" for row in cutoff.committed_revision_tuples)
    prepared = _coordinator().prepare_generation(
        cutoff,
        principal=ADMIN,
        reader_capability_manifest_id=capability_id,
        target_mode="economic",
    )
    with SessionLocal() as session:
        authority = EconomicTaxonomyPublicationRepository(session).lock_authority()
        repository = EconomicTaxonomyPublicationRepository(session)
        repository.append_source_revision(
            producer_kind="content",
            logical_source_key="post:late-low-id",
            revision_kind="evidence",
            revision_number=2,
            content_hash="changed-after-snapshot",
            authority_epoch=authority.authority_epoch,
        )
        for ordinal in range(5):
            repository.append_source_revision(
                producer_kind="social",
                logical_source_key=f"post:continuous:{ordinal}",
                revision_kind="decision",
                revision_number=1,
                content_hash=f"continuous-{ordinal}",
                authority_epoch=authority.authority_epoch,
            )
        session.commit()

    _coordinator().publish_generation(prepared.id, principal=ADMIN)

    with SessionLocal() as session:
        authority = session.get(TaxonomyAuthority, 1)
        manifest = session.get(GenerationInputManifest, cutoff.manifest_id)
        assert authority.serving_generation_id == prepared.id
        assert authority.cutover_catch_up_cursor == manifest.committed_revision_tuples
        assert all(
            not (row[1] == "post:late-low-id" and row[3] == 2)
            for row in manifest.committed_revision_tuples
        )


def test_stale_parent_candidate_is_abandoned(seeded):
    _taxonomy_id, capability_id = seeded
    coordinator = _coordinator()
    cutoff_a, first = _prepare(coordinator, capability_id)
    _cutoff_b, second = _prepare(coordinator, capability_id)

    coordinator.publish_generation(first.id, principal=ADMIN)
    with pytest.raises(ManifestChanged, match="expected_parent_changed"):
        coordinator.publish_generation(second.id, principal=ADMIN)

    with SessionLocal() as session:
        authority = session.get(TaxonomyAuthority, 1)
        assert authority.serving_generation_id == first.id
        assert (
            session.scalar(
                select(ServingGenerationEvent.id).where(
                    ServingGenerationEvent.serving_generation_id == second.id,
                    ServingGenerationEvent.event_type == "abandoned",
                )
            )
            is not None
        )
        assert cutoff_a.expected_parent_generation_id is None


def test_structural_invalidation_abandons_but_ordinary_arrivals_do_not(seeded):
    _taxonomy_id, capability_id = seeded
    coordinator = _coordinator()
    _cutoff, prepared = _prepare(coordinator, capability_id)
    with SessionLocal() as session:
        EconomicTaxonomyPublicationRepository(session).append_semantic_invalidation(
            reason="reviewed merge", actor=ADMIN.subject
        )
        session.commit()

    with pytest.raises(ManifestChanged, match="semantic_invalidation_changed"):
        coordinator.publish_generation(prepared.id, principal=ADMIN)

    with SessionLocal() as session:
        assert session.get(TaxonomyAuthority, 1).serving_generation_id is None


def test_publish_never_waits_for_outbox_and_polling_recovers_notification_crash(
    seeded,
):
    _taxonomy_id, capability_id = seeded
    coordinator = _coordinator(compatibility_builder=_compatibility)
    _cutoff, prepared = _prepare(coordinator, capability_id)

    def crash():
        raise InjectedPublicationCrash("after-commit-before-notify")

    with pytest.raises(InjectedPublicationCrash):
        coordinator.publish_generation(
            prepared.id, principal=ADMIN, notify_delivery_workers=crash
        )

    with SessionLocal() as session:
        authority = session.get(TaxonomyAuthority, 1)
        assert authority.serving_generation_id == prepared.id
        assert authority.rollback_state == "temporarily_unavailable"
        epoch = authority.authority_epoch
        runtime = EconomicTaxonomyRuntimeService(session)
        claims = runtime.claim_deliveries_from_published_generations(
            worker_id="test:poller", expected_epoch=epoch, now=NOW, limit=10
        )
        session.commit()
    assert len(claims) == 1

    with SessionLocal() as session:
        runtime = EconomicTaxonomyRuntimeService(session)
        first = runtime.apply_delivery(claims[0], expected_epoch=epoch, now=NOW)
        session.commit()
        assert first.applied is True

    with SessionLocal() as session:
        retry = EconomicTaxonomyRuntimeService(session).apply_delivery(
            claims[0], expected_epoch=epoch, now=NOW
        )
        session.commit()
        assert retry.outcome == "success"

    assert coordinator.refresh_rollback_availability() == "ready"
    with SessionLocal() as session:
        assert session.query(TaxonomyProjectionDeliveryEvent).count() == 1


def test_old_writer_finishes_before_epoch_switch_then_becomes_stale(seeded):
    _taxonomy_id, capability_id = seeded
    coordinator = _coordinator()
    _cutoff, prepared = _prepare(coordinator, capability_id)
    writer_entered = Event()
    release_writer = Event()
    publisher_done = Event()

    def old_writer():
        with SessionLocal() as session:
            with producer_write(
                session, expected_epoch=7, allowed_modes={"dual"}
            ) as authority:
                EconomicTaxonomyPublicationRepository(session).append_source_revision(
                    producer_kind="content",
                    logical_source_key="old-writer",
                    revision_kind="evidence",
                    revision_number=1,
                    content_hash="old-writer",
                    authority_epoch=authority.authority_epoch,
                )
                writer_entered.set()
                release_writer.wait(timeout=5)
            session.commit()

    def publish():
        coordinator.publish_generation(prepared.id, principal=ADMIN)
        publisher_done.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(old_writer)
        assert writer_entered.wait(timeout=5)
        publisher = pool.submit(publish)
        assert not publisher_done.wait(timeout=0.2)
        release_writer.set()
        writer.result(timeout=5)
        publisher.result(timeout=5)

    with SessionLocal() as session:  # noqa: SIM117
        with pytest.raises(StaleAuthorityEpoch):
            with producer_write(session, expected_epoch=7, allowed_modes={"economic"}):
                pass


def test_precommit_crash_and_unhealthy_rollback_leave_coherent_states(seeded):
    _taxonomy_id, capability_id = seeded
    coordinator = _coordinator(compatibility_builder=_compatibility)
    _cutoff, prepared = _prepare(coordinator, capability_id)

    def fail_before_commit():
        raise InjectedPublicationCrash("precommit")

    with pytest.raises(InjectedPublicationCrash):
        coordinator.publish_generation(
            prepared.id, principal=ADMIN, before_commit=fail_before_commit
        )
    with SessionLocal() as session:
        assert session.get(TaxonomyAuthority, 1).serving_generation_id is None

    coordinator.publish_generation(prepared.id, principal=ADMIN)
    rolled_back = coordinator.rollback(principal=ADMIN, reason="outbox blocked")

    with SessionLocal() as session:
        authority = session.get(TaxonomyAuthority, 1)
        assert authority.serving_generation_id == rolled_back.id
        assert authority.mode == "legacy"
        assert authority.writes_fenced is False
        assert authority.rollback_state == "ready"
        published = session.scalars(
            select(ServingGenerationEvent).where(
                ServingGenerationEvent.event_type == "published"
            )
        ).all()
        assert len(published) == 2
        abandoned_events = session.scalars(
            select(TaxonomyProjectionEvent)
            .join(
                ServingGenerationEvent,
                ServingGenerationEvent.serving_generation_id
                == TaxonomyProjectionEvent.serving_generation_id,
            )
            .where(ServingGenerationEvent.event_type == "abandoned")
        ).all()
        assert abandoned_events == []


def test_abandoned_candidate_projection_is_never_claimed(seeded):
    _taxonomy_id, capability_id = seeded
    coordinator = _coordinator(compatibility_builder=_compatibility)
    _cutoff, prepared = _prepare(coordinator, capability_id)
    coordinator.abandon(prepared.id, principal=ADMIN, reason="stale fixture")

    with SessionLocal() as session:
        claims = EconomicTaxonomyRuntimeService(
            session
        ).claim_deliveries_from_published_generations(
            worker_id="test:poller",
            expected_epoch=7,
            now=NOW,
            limit=10,
        )
        assert claims == []


def test_benchmark_anchor_walk_on_postgres(seeded):
    taxonomy_id, _capability_id = seeded
    with SessionLocal() as session:
        head_id = taxonomy_id
        for index in range(12):
            version = TaxonomyVersion(
                parent_version_id=head_id,
                status="sealed",
                semantic_hash=f"pg-processor-{index}",
                created_by=SYSTEM_ACTOR,
                reason="pg anchor",
            )
            session.add(version)
            session.flush()
            head_id = version.id
        loop = TaxonomyVersion(status="draft", created_by=SYSTEM_ACTOR, reason="pg loop")
        session.add(loop)
        session.flush()
        loop.parent_version_id = loop.id
        session.commit()

        assert benchmark_anchor_version(session, head_id).id == taxonomy_id
        with pytest.raises(LookupError, match="benchmark_anchor_missing"):
            benchmark_anchor_version(session, loop.id)
