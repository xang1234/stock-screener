from __future__ import annotations

from datetime import datetime, timezone

import pytest
from app.domain.economic_taxonomy.contracts import AdminPrincipal
from app.infra.db.repositories.economic_taxonomy_publication_repo import (
    EconomicTaxonomyPublicationRepository,
)
from app.infra.db.repositories.economic_taxonomy_repo import EconomicTaxonomyRepository
from app.models.economic_taxonomy_runtime import (
    GenerationInputManifest,
    InterpretationSet,
    MetricsRevision,
    ReaderCapabilityManifest,
    ReaderSnapshotBundle,
    ReaderSnapshotPointer,
    ServingGenerationEvent,
    TaxonomyAuthority,
    TaxonomyProjectionEvent,
    TaxonomySourceRevisionLog,
)
from app.services.economic_taxonomy_benchmark_store import (
    register_verified_benchmark,
)
from app.services.economic_taxonomy_publication import (
    BenchmarkRejected,
    CompatibilityProjection,
    EconomicTaxonomyPublicationCoordinator,
    InjectedPublicationCrash,
    ManifestChanged,
)
from app.services.economic_taxonomy_publication_compatibility import (
    CompatibilityProjection as SplitCompatibilityProjection,
)
from app.services.economic_taxonomy_runtime import EconomicTaxonomyRuntimeService
from sqlalchemy import select

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
ADMIN = AdminPrincipal(
    subject="admin:alice",
    auth_method="admin_api_key",
    roles=frozenset({"taxonomy:review"}),
)


def test_publication_facade_preserves_split_contract_identity():
    assert SplitCompatibilityProjection is CompatibilityProjection


def _seed(db_session, *, include_benchmark=True):
    repository = EconomicTaxonomyRepository(db_session)
    draft = repository.create_draft(actor=ADMIN.subject, reason="publication fixture")
    taxonomy = repository.seal_draft(draft.id)
    capability = ReaderCapabilityManifest(
        backend_contract=1,
        frontend_contract=1,
        migration_version="0055",
        consumer_test_hash="publication-tests-v1",
        verified_by=ADMIN.subject,
    )
    authority = TaxonomyAuthority(
        id=1,
        mode="dual",
        processing_taxonomy_version_id=taxonomy.id,
        processing_head_revision=4,
        authority_epoch=7,
        writes_fenced=False,
        semantic_invalidation_revision=0,
        cutover_catch_up_cursor=[],
        rollback_state="ready",
    )
    db_session.add_all([capability, authority])
    if include_benchmark:
        register_verified_benchmark(
            db_session,
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
    db_session.commit()
    return taxonomy, capability


def _coordinator(db_session, **overrides):
    factory = lambda: db_session.__class__(bind=db_session.get_bind())
    return EconomicTaxonomyPublicationCoordinator(
        factory,
        clock=lambda: NOW,
        **overrides,
    )


def test_prepare_builds_all_artifacts_from_one_manifest(db_session):
    _taxonomy, capability = _seed(db_session)
    coordinator = _coordinator(db_session)

    cutoff = coordinator.capture_cutoff(principal=ADMIN, selections=[])
    prepared = coordinator.prepare_generation(
        cutoff,
        principal=ADMIN,
        reader_capability_manifest_id=capability.id,
    )

    with db_session.__class__(bind=db_session.get_bind()) as check:
        generation = prepared.load(check)
        interpretation = check.get(InterpretationSet, generation.interpretation_set_id)
        metrics = check.get(MetricsRevision, generation.metrics_revision_id)
        snapshots = check.get(
            ReaderSnapshotBundle, generation.reader_snapshot_bundle_id
        )
        assert interpretation.generation_input_manifest_id == cutoff.manifest_id
        assert metrics.generation_input_manifest_id == cutoff.manifest_id
        assert snapshots.generation_input_manifest_id == cutoff.manifest_id
        assert [event.event_type for event in generation.events] == ["prepared"]
        benchmark = generation.events[0].details["benchmark"]
        assert benchmark["passed"] is True
        assert benchmark["benchmark_result_id"]
        assert benchmark["report_hash"]
        assert check.get(TaxonomyAuthority, 1).serving_generation_id is None


def test_lock_authority_returns_the_row_as_locked_not_as_first_loaded(db_session):
    # #556: a caller that loaded the authority earlier in the session must see
    # a cutover or epoch change committed before its lock, not the stale copy.
    from sqlalchemy import update

    repo = EconomicTaxonomyPublicationRepository(db_session)
    loaded = repo.lock_authority()
    db_session.commit()
    assert (loaded.authority_epoch, loaded.writes_fenced) == (1, False)
    assert db_session.get(TaxonomyAuthority, 1) is loaded  # held in the identity map

    # Another transaction's commit: the row changes, the session's copy does not.
    db_session.execute(
        update(TaxonomyAuthority)
        .where(TaxonomyAuthority.id == 1)
        .values(authority_epoch=2, writes_fenced=True)
        .execution_options(synchronize_session=False)
    )
    assert loaded.authority_epoch == 1

    locked = repo.lock_authority()

    assert (locked.authority_epoch, locked.writes_fenced) == (2, True)


def test_ordinary_post_cutoff_revision_does_not_abort(db_session):
    _taxonomy, capability = _seed(db_session)
    coordinator = _coordinator(db_session)
    cutoff = coordinator.capture_cutoff(principal=ADMIN, selections=[])
    prepared = coordinator.prepare_generation(
        cutoff,
        principal=ADMIN,
        reader_capability_manifest_id=capability.id,
    )
    with db_session.__class__(bind=db_session.get_bind()) as later:
        authority = EconomicTaxonomyPublicationRepository(later).lock_authority()
        EconomicTaxonomyPublicationRepository(later).append_source_revision(
            producer_kind="content",
            logical_source_key="article:after-cutoff",
            revision_kind="evidence",
            revision_number=1,
            content_hash="later",
            authority_epoch=authority.authority_epoch,
        )
        later.commit()

    coordinator.publish_generation(prepared.id, principal=ADMIN)

    with db_session.__class__(bind=db_session.get_bind()) as check:
        authority = check.get(TaxonomyAuthority, 1)
        manifest = check.get(GenerationInputManifest, cutoff.manifest_id)
        assert authority.serving_generation_id == prepared.id
        assert authority.cutover_catch_up_cursor == manifest.committed_revision_tuples
        assert authority.authority_epoch == 8
        assert {
            row.reader_key for row in check.scalars(select(ReaderSnapshotPointer))
        } == {"economic_taxonomy", "economic_themes"}


def test_structural_invalidator_abandons_without_switch(db_session):
    _taxonomy, capability = _seed(db_session)
    coordinator = _coordinator(db_session)
    prepared = coordinator.prepare_generation(
        coordinator.capture_cutoff(principal=ADMIN, selections=[]),
        principal=ADMIN,
        reader_capability_manifest_id=capability.id,
    )
    with db_session.__class__(bind=db_session.get_bind()) as changed:
        EconomicTaxonomyPublicationRepository(changed).append_semantic_invalidation(
            reason="reviewed split", actor=ADMIN.subject
        )
        changed.commit()

    with pytest.raises(ManifestChanged, match="semantic_invalidation_changed"):
        coordinator.publish_generation(prepared.id, principal=ADMIN)

    with db_session.__class__(bind=db_session.get_bind()) as check:
        assert check.get(TaxonomyAuthority, 1).serving_generation_id is None
        events = check.scalars(
            select(ServingGenerationEvent)
            .where(ServingGenerationEvent.serving_generation_id == prepared.id)
            .order_by(ServingGenerationEvent.sequence_number)
        ).all()
        assert [event.event_type for event in events] == ["prepared", "abandoned"]


def test_crash_after_commit_leaves_published_events_pollable(db_session):
    _taxonomy, capability = _seed(db_session)

    def compatibility(_session, context):
        return [
            CompatibilityProjection(
                source_lineage="post:1",
                projection_kind="legacy_theme",
                projection_version=1,
                target="legacy",
                payload={"themes": []},
                origin_representation="economic",
                selected_interpretation_version=str(context.interpretation_set_id),
                mapping_version=str(context.taxonomy_version_id),
            )
        ]

    coordinator = _coordinator(db_session, compatibility_builder=compatibility)
    prepared = coordinator.prepare_generation(
        coordinator.capture_cutoff(principal=ADMIN, selections=[]),
        principal=ADMIN,
        reader_capability_manifest_id=capability.id,
    )

    def crash():
        raise InjectedPublicationCrash("publication_commit_before_notification")

    with pytest.raises(InjectedPublicationCrash):
        coordinator.publish_generation(
            prepared.id, principal=ADMIN, notify_delivery_workers=crash
        )

    with db_session.__class__(bind=db_session.get_bind()) as check:
        assert check.get(TaxonomyAuthority, 1).serving_generation_id == prepared.id
        assert (
            check.scalar(
                select(ServingGenerationEvent.id).where(
                    ServingGenerationEvent.serving_generation_id == prepared.id,
                    ServingGenerationEvent.event_type == "published",
                )
            )
            is not None
        )
        assert (
            check.scalar(
                select(TaxonomyProjectionEvent.id).where(
                    TaxonomyProjectionEvent.serving_generation_id == prepared.id
                )
            )
            is not None
        )


def test_abandoned_generation_events_are_never_delivery_eligible(db_session):
    _taxonomy, capability = _seed(db_session)

    def compatibility(_session, context):
        return [
            CompatibilityProjection(
                source_lineage="post:abandoned",
                projection_kind="legacy_theme",
                projection_version=1,
                target="legacy",
                payload={"themes": []},
                origin_representation="economic",
                selected_interpretation_version=str(context.interpretation_set_id),
                mapping_version=str(context.taxonomy_version_id),
            )
        ]

    coordinator = _coordinator(db_session, compatibility_builder=compatibility)
    prepared = coordinator.prepare_generation(
        coordinator.capture_cutoff(principal=ADMIN, selections=[]),
        principal=ADMIN,
        reader_capability_manifest_id=capability.id,
    )
    coordinator.abandon(prepared.id, principal=ADMIN, reason="stale_parent")

    with db_session.__class__(bind=db_session.get_bind()) as check:
        eligible = check.scalars(
            select(TaxonomyProjectionEvent)
            .join(
                ServingGenerationEvent,
                ServingGenerationEvent.serving_generation_id
                == TaxonomyProjectionEvent.serving_generation_id,
            )
            .where(ServingGenerationEvent.event_type == "published")
        ).all()
        assert eligible == []


def test_mode_change_requires_trusted_principal(db_session):
    _seed(db_session)
    coordinator = _coordinator(db_session)
    untrusted = AdminPrincipal(
        subject="viewer:bob", auth_method="session", roles=frozenset()
    )

    with pytest.raises(PermissionError, match="taxonomy_publication_forbidden"):
        coordinator.capture_cutoff(principal=untrusted, selections=[])


def test_precommit_crash_keeps_old_authority_and_no_published_event(db_session):
    _taxonomy, capability = _seed(db_session)
    coordinator = _coordinator(db_session)
    prepared = coordinator.prepare_generation(
        coordinator.capture_cutoff(principal=ADMIN, selections=[]),
        principal=ADMIN,
        reader_capability_manifest_id=capability.id,
    )

    def crash():
        raise InjectedPublicationCrash("before_commit")

    with pytest.raises(InjectedPublicationCrash, match="before_commit"):
        coordinator.publish_generation(
            prepared.id, principal=ADMIN, before_commit=crash
        )

    with db_session.__class__(bind=db_session.get_bind()) as check:
        assert check.get(TaxonomyAuthority, 1).serving_generation_id is None
        assert (
            check.scalar(
                select(ServingGenerationEvent.id).where(
                    ServingGenerationEvent.serving_generation_id == prepared.id,
                    ServingGenerationEvent.event_type == "published",
                )
            )
            is None
        )


def test_unhealthy_rollback_rebuilds_legacy_before_switch(db_session):
    _taxonomy, capability = _seed(db_session)

    def compatibility(_session, context):
        return [
            CompatibilityProjection(
                source_lineage="post:rollback",
                projection_kind="legacy_theme",
                projection_version=1,
                target="legacy",
                payload={"themes": ["memory"]},
                origin_representation="economic",
                selected_interpretation_version=str(context.interpretation_set_id),
                mapping_version=str(context.taxonomy_version_id),
            )
        ]

    coordinator = _coordinator(db_session, compatibility_builder=compatibility)
    first = coordinator.prepare_generation(
        coordinator.capture_cutoff(principal=ADMIN, selections=[]),
        principal=ADMIN,
        reader_capability_manifest_id=capability.id,
        target_mode="economic",
    )
    coordinator.publish_generation(first.id, principal=ADMIN)

    rolled_back = coordinator.rollback(principal=ADMIN, reason="delivery unhealthy")

    with db_session.__class__(bind=db_session.get_bind()) as check:
        authority = check.get(TaxonomyAuthority, 1)
        assert rolled_back.id == authority.serving_generation_id
        assert authority.mode == "legacy"
        assert authority.writes_fenced is False
        assert authority.rollback_state == "ready"
        assert EconomicTaxonomyRuntimeService(check).generation_acknowledged(first.id)
        recovery = check.scalar(
            select(TaxonomySourceRevisionLog).where(
                TaxonomySourceRevisionLog.revision_kind == "rollback_recovery"
            )
        )
        assert recovery is not None


def test_healthy_rollback_projects_economic_developments(db_session, monkeypatch):
    # Developments recorded under economic authority need legacy links on every
    # rollback, not only on the recovery path, and in the transaction that
    # switches to legacy: none may commit after it and before the links (#513).
    import app.services.economic_taxonomy_publication as publication_module

    taxonomy, capability = _seed(db_session)
    coordinator = _coordinator(db_session)
    first = coordinator.prepare_generation(
        coordinator.capture_cutoff(principal=ADMIN, selections=[]),
        principal=ADMIN,
        reader_capability_manifest_id=capability.id,
        target_mode="economic",
    )
    coordinator.publish_generation(first.id, principal=ADMIN)
    projected = []
    monkeypatch.setattr(
        publication_module,
        "project_economic_developments",
        lambda session, version_id: projected.append(
            (version_id, session.get(TaxonomyAuthority, 1).mode)
        ),
    )

    coordinator.rollback(principal=ADMIN)

    assert projected == [(taxonomy.id, "economic")]


def test_failed_benchmark_does_not_prepare_candidate(db_session):
    _taxonomy, capability = _seed(db_session)
    coordinator = _coordinator(
        db_session,
        benchmark_verifier=lambda _session, _context: {
            "passed": False,
            "failed_cases": ["contrast:memory-vs-storage"],
        },
    )
    cutoff = coordinator.capture_cutoff(principal=ADMIN, selections=[])

    with pytest.raises(BenchmarkRejected, match="benchmark_failed"):
        coordinator.prepare_generation(
            cutoff,
            principal=ADMIN,
            reader_capability_manifest_id=capability.id,
        )

    with db_session.__class__(bind=db_session.get_bind()) as check:
        assert check.query(ServingGenerationEvent).count() == 0


def test_passing_callback_cannot_bypass_durable_benchmark_artifact(db_session):
    _taxonomy, capability = _seed(db_session)
    coordinator = _coordinator(
        db_session,
        benchmark_verifier=lambda _session, _context: {"passed": True},
    )
    cutoff = coordinator.capture_cutoff(principal=ADMIN, selections=[])

    with pytest.raises(BenchmarkRejected, match="benchmark_result_missing"):
        coordinator.prepare_generation(
            cutoff,
            principal=ADMIN,
            reader_capability_manifest_id=capability.id,
        )


def test_default_benchmark_verification_fails_closed_without_registered_result(
    db_session,
):
    _taxonomy, capability = _seed(db_session, include_benchmark=False)
    coordinator = _coordinator(db_session)
    cutoff = coordinator.capture_cutoff(principal=ADMIN, selections=[])

    with pytest.raises(BenchmarkRejected, match="benchmark_result_missing"):
        coordinator.prepare_generation(
            cutoff,
            principal=ADMIN,
            reader_capability_manifest_id=capability.id,
        )
