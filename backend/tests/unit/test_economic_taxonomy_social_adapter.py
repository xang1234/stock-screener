from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from app.domain.economic_taxonomy.contracts import AdminPrincipal
from app.infra.db.models.social_analysis import (
    EconomicSocialAssociation,
    EconomicSocialAssociationRevision,
    EconomicSocialAssociationSource,
    SocialExtractionWork,
    SocialRunWork,
    SocialThemeAssociation,
    SocialThemeDecision,
)
from app.infra.db.models.social_signals import SocialSignalRun, SocialSourceRegistry
from app.infra.db.repositories.economic_taxonomy_repo import EconomicTaxonomyRepository
from app.models.economic_taxonomy import EconomicTheme
from app.models.economic_taxonomy_runtime import (
    EvidencePacket,
    TaxonomyAuthority,
    TaxonomySourceRevisionLog,
)
from app.models.stock_universe import StockUniverse
from app.models.theme import ContentItem, ThemeCluster
from app.services.economic_social_taxonomy_adapter import EconomicSocialTaxonomyAdapter
from app.services.economic_source_admission import (
    CONTENT_INGESTION_ROUTE,
    EconomicSourceAdmissionService,
    EvidenceAdmission,
)
from app.services.economic_taxonomy_fence import AuthorityWritesFenced
from app.services.economic_taxonomy_publication_compatibility import (
    build_default_compatibility_projections,
)
from app.services.economic_taxonomy_publication_contracts import PreparationContext
from app.services.economic_taxonomy_rollback_recovery import RollbackRecovery
from app.services.economic_taxonomy_runtime import EconomicTaxonomyRuntimeService
from app.services.social_theme_market_service import EconomicAcceptedBasketReader
from app.services.social_theme_projection_service import (
    SocialThemeProjectionService,
)

from .economic_taxonomy_reader_helpers import seed_generation

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
ADMIN = AdminPrincipal(
    subject="admin:test",
    auth_method="admin_api_key",
    roles=frozenset({"taxonomy:review"}),
)


def _global_pair(db_session):
    theme = EconomicTheme(created_by="test:social")
    security = StockUniverse(symbol="MU", market="US", is_active=True)
    db_session.add_all([theme, security])
    db_session.flush()
    return theme, security


def _legacy_association(db_session, *, name: str, state: str):
    cluster = ThemeCluster(
        name=name,
        display_name=name,
        canonical_key=name.casefold().replace(" ", "_"),
        pipeline="technical",
    )
    db_session.add(cluster)
    db_session.flush()
    association = SocialThemeAssociation(
        theme_cluster_id=cluster.id,
        market="US",
        canonical_symbol="MU",
        state=state,
        origin="social",
        decision_owner="admin",
        evidence_work_ids=[],
        policy_version="social-theme-v1",
        version=1,
        first_seen_at=NOW,
        accepted_at=NOW if state == "accepted" else None,
        updated_at=NOW,
    )
    db_session.add(association)
    db_session.flush()
    return association


def test_two_legacy_associations_bridge_one_global_membership(db_session):
    theme, security = _global_pair(db_session)
    accepted_a = _legacy_association(db_session, name="AI Memory A", state="accepted")
    accepted_b = _legacy_association(db_session, name="AI Memory B", state="accepted")

    result = EconomicSocialTaxonomyAdapter(db_session).project_legacy_associations(
        economic_theme_id=theme.id,
        security_id=security.id,
        legacy_association_ids=(accepted_a.id, accepted_b.id),
    )

    assert result.global_association_count == 1
    assert result.bridge_count == 2
    assert result.state == "accepted"
    assert result.live is True
    assert db_session.query(EconomicSocialAssociation).count() == 1
    assert db_session.query(EconomicSocialAssociationSource).count() == 2


def test_conflicting_admin_decisions_are_order_independent_and_not_live(db_session):
    theme, security = _global_pair(db_session)
    accepted = _legacy_association(db_session, name="Memory", state="accepted")
    rejected = _legacy_association(db_session, name="Memory Chips", state="rejected")
    adapter = EconomicSocialTaxonomyAdapter(db_session)

    first = adapter.project_legacy_associations(
        economic_theme_id=theme.id,
        security_id=security.id,
        legacy_association_ids=(accepted.id, rejected.id),
    )
    second = adapter.project_legacy_associations(
        economic_theme_id=theme.id,
        security_id=security.id,
        legacy_association_ids=(rejected.id, accepted.id),
    )

    assert first.state == second.state == "conflict_review_required"
    assert first.live is second.live is False
    assert db_session.query(EconomicSocialAssociationRevision).count() == 1
    assert db_session.query(TaxonomySourceRevisionLog).one().revision_kind == (
        "social_conflict"
    )


def test_admin_rejection_prevents_system_acceptance(db_session):
    theme, security = _global_pair(db_session)
    automatic = _legacy_association(db_session, name="Automatic Memory", state="accepted")
    automatic.decision_owner = "system"
    rejected = _legacy_association(db_session, name="Reviewed Memory", state="rejected")

    result = EconomicSocialTaxonomyAdapter(db_session).project_legacy_associations(
        economic_theme_id=theme.id,
        security_id=security.id,
        legacy_association_ids=(automatic.id, rejected.id),
    )

    assert result.state == "rejected"
    assert result.live is False


def test_pair_revision_history_is_pinned_and_decision_retry_is_idempotent(db_session):
    theme, security = _global_pair(db_session)
    adapter = EconomicSocialTaxonomyAdapter(db_session)
    association = adapter.get_or_create_association(theme.id, security.id)

    accepted = adapter.revise(
        association.id,
        state="accepted",
        idempotency_key="decision-1",
        actor="admin:test",
        reason="reviewed",
        mirror_acknowledged=True,
    )
    accepted_ref = adapter.pin_revision(accepted.id)
    rejected = adapter.revise(
        association.id,
        state="rejected",
        idempotency_key="decision-2",
        actor="admin:test",
        reason="corrected",
        mirror_acknowledged=True,
    )
    retry = adapter.revise(
        association.id,
        state="rejected",
        idempotency_key="decision-2",
        actor="admin:test",
        reason="corrected",
        mirror_acknowledged=True,
    )
    rejected_ref = adapter.pin_revision(rejected.id)

    assert [accepted.revision_number, rejected.revision_number] == [1, 2]
    assert retry.id == rejected.id
    assert adapter.membership(accepted_ref.id).state == "accepted"
    assert adapter.membership(accepted_ref.id).live is True
    assert adapter.membership(rejected_ref.id).state == "rejected"
    assert adapter.membership(rejected_ref.id).live is False

    historical = EconomicAcceptedBasketReader(
        db_session,
        economic_theme_id=theme.id,
        association_revision_ref_ids=(accepted_ref.id,),
    ).read("memory", "US")
    current = EconomicAcceptedBasketReader(
        db_session,
        economic_theme_id=theme.id,
    ).read("memory", "US")
    assert historical.company_stock_symbols == ("MU",)
    assert current.company_stock_symbols == ()


def test_native_membership_waits_for_legacy_mirror_before_becoming_live(db_session):
    theme, security = _global_pair(db_session)
    adapter = EconomicSocialTaxonomyAdapter(db_session)
    association = adapter.get_or_create_association(theme.id, security.id)

    pending = adapter.revise(
        association.id,
        state="accepted",
        idempotency_key="native-decision",
        actor="admin:test",
        reason="reviewed native membership",
        mirror_acknowledged=False,
    )

    assert pending.state == "pending_legacy_mirror"
    assert pending.live is False
    assert db_session.query(SocialThemeAssociation).count() == 0

    acknowledged = adapter.apply_legacy_mirror(pending.id, now=NOW)

    assert acknowledged.state == "accepted"
    assert acknowledged.live is True
    assert acknowledged.mirror_state == "acknowledged"
    assert db_session.query(SocialThemeAssociation).count() == 1
    assert db_session.query(EconomicSocialAssociationSource).count() == 1
    revisions = db_session.scalars(
        select(TaxonomySourceRevisionLog)
        .where(TaxonomySourceRevisionLog.producer_kind == "economic_social")
        .order_by(TaxonomySourceRevisionLog.revision_number)
    ).all()
    assert [row.revision_number for row in revisions] == [1, 2]
    assert [row.content_hash for row in revisions] == [
        pending.reconciliation_hash,
        acknowledged.reconciliation_hash,
    ]


def test_social_membership_delivery_applies_legacy_mirror_before_success(db_session):
    seeded = seed_generation(db_session)
    theme, security = _global_pair(db_session)
    adapter = EconomicSocialTaxonomyAdapter(db_session)
    association = adapter.get_or_create_association(theme.id, security.id)
    pending = adapter.revise(
        association.id,
        state="accepted",
        idempotency_key="delivered-native-decision",
        actor="admin:test",
        reason="reviewed native membership",
        mirror_acknowledged=False,
    )
    db_session.commit()
    authority = db_session.get(TaxonomyAuthority, 1)
    runtime = EconomicTaxonomyRuntimeService(db_session)
    claim = runtime.claim_deliveries_from_published_generations(
        worker_id="worker:social-mirror",
        expected_epoch=authority.authority_epoch,
        now=NOW,
        limit=1,
    )[0]
    db_session.commit()

    result = runtime.apply_delivery(
        claim,
        expected_epoch=authority.authority_epoch,
        now=NOW,
    )
    db_session.commit()

    latest = db_session.scalar(
        select(EconomicSocialAssociationRevision)
        .where(EconomicSocialAssociationRevision.association_id == association.id)
        .order_by(EconomicSocialAssociationRevision.revision_number.desc())
        .limit(1)
    )
    assert claim.projection_event_id == pending.projection_event_id
    assert seeded["generation"].id == authority.serving_generation_id
    assert result.outcome == "success"
    assert latest.state == "accepted"
    assert latest.live is True
    assert latest.mirror_state == "acknowledged"
    assert db_session.query(SocialThemeAssociation).count() == 1


def _deliver_next(db_session):
    authority = db_session.get(TaxonomyAuthority, 1)
    runtime = EconomicTaxonomyRuntimeService(db_session)
    claim = runtime.claim_deliveries_from_published_generations(
        worker_id="worker:social-mirror",
        expected_epoch=authority.authority_epoch,
        now=NOW,
        limit=1,
    )[0]
    db_session.commit()
    result = runtime.apply_delivery(
        claim,
        expected_epoch=authority.authority_epoch,
        now=NOW,
    )
    db_session.commit()
    return claim, result


def test_social_rejection_retracts_and_reacceptance_restores_legacy_mirror(
    db_session,
):
    seed_generation(db_session)
    theme, security = _global_pair(db_session)
    adapter = EconomicSocialTaxonomyAdapter(db_session)
    association = adapter.get_or_create_association(theme.id, security.id)
    adapter.revise(
        association.id,
        state="accepted",
        idempotency_key="accepted-before-retraction",
        actor="admin:test",
        reason="reviewed native membership",
        mirror_acknowledged=False,
    )
    db_session.commit()
    _deliver_next(db_session)
    legacy = db_session.scalar(select(SocialThemeAssociation))
    assert legacy.state == "accepted"

    rejected = adapter.revise(
        association.id,
        state="rejected",
        idempotency_key="rejected-after-mirror",
        actor="admin:test",
        reason="membership no longer supported",
        mirror_acknowledged=True,
    )
    db_session.commit()

    assert rejected.state == "rejected"
    assert rejected.live is False
    assert rejected.mirror_state == "pending"
    assert rejected.projection_event_id is not None

    claim, result = _deliver_next(db_session)

    assert claim.projection_event_id == rejected.projection_event_id
    assert result.outcome == "success"
    db_session.refresh(legacy)
    assert legacy.state == "rejected"
    latest = db_session.scalar(
        select(EconomicSocialAssociationRevision)
        .where(EconomicSocialAssociationRevision.association_id == association.id)
        .order_by(EconomicSocialAssociationRevision.revision_number.desc())
        .limit(1)
    )
    assert latest.state == "rejected"
    assert latest.live is False
    assert latest.mirror_state == "acknowledged"
    assert db_session.scalar(
        select(SocialThemeDecision).where(
            SocialThemeDecision.association_id == legacy.id,
            SocialThemeDecision.after_state == "rejected",
        )
    ).actor == "system:economic-taxonomy-mirror"

    adapter.revise(
        association.id,
        state="accepted",
        idempotency_key="reaccepted-after-retraction",
        actor="admin:test",
        reason="membership reinstated",
        mirror_acknowledged=False,
    )
    db_session.commit()
    _deliver_next(db_session)

    db_session.refresh(legacy)
    assert legacy.state == "accepted"
    assert db_session.query(SocialThemeAssociation).count() == 1


def _latest_revision(db, association_id):
    return max(
        db.scalars(
            select(EconomicSocialAssociationRevision.revision_number).where(
                EconomicSocialAssociationRevision.association_id == association_id
            )
        )
    )


def test_admin_decision_after_cutover_revises_economic_association(db_session):
    seed_generation(db_session)
    theme, security = _global_pair(db_session)
    legacy = _legacy_association(db_session, name="AI Memory", state="accepted")
    projected = EconomicSocialTaxonomyAdapter(db_session).project_legacy_associations(
        economic_theme_id=theme.id,
        security_id=security.id,
        legacy_association_ids=(legacy.id,),
    )
    db_session.commit()
    service = SocialThemeProjectionService(db_session, admin_authorized=True)

    rejected = service.decide(
        legacy.id, "rejected", "reviewed evidence", "admin", legacy.version,
        expected_economic_revision=_latest_revision(db_session, projected.association_id),
    )
    db_session.commit()

    assert rejected.association_id == projected.association_id
    assert rejected.state == "rejected"
    assert rejected.live is False
    assert rejected.mirror_state == "pending"
    db_session.refresh(legacy)
    assert legacy.state == "accepted"

    _deliver_next(db_session)

    db_session.refresh(legacy)
    assert legacy.state == "rejected"
    with pytest.raises(ValueError, match="association_version_conflict"):
        service.decide(
            legacy.id, "accepted", "stale", "admin", legacy.version - 1,
            expected_economic_revision=_latest_revision(db_session, projected.association_id),
        )

    accepted = service.decide(
        legacy.id, "accepted", "reinstated", "admin", legacy.version,
        expected_economic_revision=_latest_revision(db_session, projected.association_id),
    )
    db_session.commit()
    assert accepted.state == "pending_legacy_mirror"

    _deliver_next(db_session)

    db_session.refresh(legacy)
    assert legacy.state == "accepted"


def test_admin_decision_after_cutover_rejects_a_stale_economic_revision(db_session):
    seed_generation(db_session)
    theme, security = _global_pair(db_session)
    legacy = _legacy_association(db_session, name="AI Memory", state="accepted")
    projected = EconomicSocialTaxonomyAdapter(db_session).project_legacy_associations(
        economic_theme_id=theme.id,
        security_id=security.id,
        legacy_association_ids=(legacy.id,),
    )
    db_session.commit()
    current = db_session.get(
        EconomicSocialAssociationRevision, projected.association_revision_id
    ).revision_number
    service = SocialThemeProjectionService(db_session, admin_authorized=True)

    # The legacy version still matches, but the economic state has moved on.
    with pytest.raises(ValueError, match="association_version_conflict"):
        service.decide(
            legacy.id, "rejected", "reviewed evidence", "admin", legacy.version,
            expected_economic_revision=current + 1,
        )
    # Without a token the legacy version alone would let a stale decision through.
    with pytest.raises(ValueError, match="economic_revision_required"):
        service.decide(legacy.id, "rejected", "reviewed evidence", "admin", legacy.version)
    rejected = service.decide(
        legacy.id, "rejected", "reviewed evidence", "admin", legacy.version,
        expected_economic_revision=current,
    )

    assert rejected.state == "rejected"
    # The check runs inside revise(), under the producer fence the append takes.
    with pytest.raises(ValueError, match="association_version_conflict"):
        EconomicSocialTaxonomyAdapter(db_session).revise(
            projected.association_id, state="accepted", idempotency_key="later",
            actor="admin", reason="stale", mirror_acknowledged=False,
            expected_revision=current,
        )


def test_admin_decision_after_cutover_requires_economic_bridge(db_session):
    seed_generation(db_session)
    legacy = _legacy_association(db_session, name="Unmapped", state="accepted")
    db_session.commit()

    with pytest.raises(ValueError, match="economic_association_missing"):
        SocialThemeProjectionService(db_session, admin_authorized=True).decide(
            legacy.id, "rejected", "reviewed evidence", "admin", legacy.version
        )


def test_social_rejection_without_legacy_mirror_stages_no_retraction(db_session):
    seed_generation(db_session)
    theme, security = _global_pair(db_session)
    adapter = EconomicSocialTaxonomyAdapter(db_session)
    association = adapter.get_or_create_association(theme.id, security.id)

    rejected = adapter.revise(
        association.id,
        state="rejected",
        idempotency_key="rejected-without-mirror",
        actor="admin:test",
        reason="never accepted",
        mirror_acknowledged=True,
    )

    assert rejected.mirror_state == "not_required"
    assert rejected.projection_event_id is None


def test_completed_social_mirror_is_not_copied_to_the_next_generation(db_session):
    seeded = seed_generation(db_session)
    theme, security = _global_pair(db_session)
    adapter = EconomicSocialTaxonomyAdapter(db_session)
    association = adapter.get_or_create_association(theme.id, security.id)
    adapter.revise(
        association.id,
        state="accepted",
        idempotency_key="completed-parent-mirror",
        actor="admin:test",
        reason="reviewed native membership",
        mirror_acknowledged=False,
    )
    db_session.commit()
    authority = db_session.get(TaxonomyAuthority, 1)
    runtime = EconomicTaxonomyRuntimeService(db_session)
    claim = runtime.claim_deliveries_from_published_generations(
        worker_id="worker:social-mirror",
        expected_epoch=authority.authority_epoch,
        now=NOW,
        limit=1,
    )[0]
    db_session.commit()
    runtime.apply_delivery(
        claim,
        expected_epoch=authority.authority_epoch,
        now=NOW,
    )
    db_session.commit()

    projections = build_default_compatibility_projections(
        db_session,
        PreparationContext(
            manifest_id=seeded["manifest"].id,
            taxonomy_version_id=seeded["taxonomy"].id,
            interpretation_set_id=seeded["interpretation"].id,
            metrics_revision_id=seeded["metrics"].id,
            reader_snapshot_bundle_id=seeded["bundle"].id,
            expected_parent_generation_id=seeded["generation"].id,
            staged_epoch=authority.authority_epoch,
            target_mode="economic",
        ),
    )

    assert {projection.projection_kind for projection in projections} == {
        "legacy_theme"
    }


def test_rollback_recovery_applies_pending_social_mirror_before_success(db_session):
    seeded = seed_generation(db_session)
    theme, security = _global_pair(db_session)
    adapter = EconomicSocialTaxonomyAdapter(db_session)
    association = adapter.get_or_create_association(theme.id, security.id)
    pending = adapter.revise(
        association.id,
        state="accepted",
        idempotency_key="rollback-parent-mirror",
        actor="admin:test",
        reason="reviewed native membership",
        mirror_acknowledged=False,
    )
    db_session.commit()
    factory = lambda: db_session.__class__(bind=db_session.get_bind())
    recovery = RollbackRecovery(factory, clock=lambda: NOW)
    recovery.begin(
        generation_id=seeded["generation"].id,
        principal=ADMIN,
        reason="compatibility delivery unhealthy",
    )

    recovery.rebuild_legacy_projections(
        seeded["generation"].id,
        principal=ADMIN,
    )

    db_session.expire_all()
    latest = db_session.scalar(
        select(EconomicSocialAssociationRevision)
        .where(EconomicSocialAssociationRevision.association_id == association.id)
        .order_by(EconomicSocialAssociationRevision.revision_number.desc())
        .limit(1)
    )
    assert pending.state == "pending_legacy_mirror"
    assert latest.state == "accepted"
    assert latest.mirror_state == "acknowledged"
    assert db_session.query(SocialThemeAssociation).count() == 1
    assert EconomicTaxonomyRuntimeService(db_session).generation_acknowledged(
        seeded["generation"].id
    )
    assert db_session.get(TaxonomyAuthority, 1).rollback_state == "recovered"


def test_native_legacy_mirror_respects_authority_write_fence(db_session):
    theme, security = _global_pair(db_session)
    adapter = EconomicSocialTaxonomyAdapter(db_session)
    association = adapter.get_or_create_association(theme.id, security.id)
    pending = adapter.revise(
        association.id,
        state="accepted",
        idempotency_key="fenced-native-decision",
        actor="admin:test",
        reason="reviewed native membership",
        mirror_acknowledged=False,
    )
    authority = db_session.get(TaxonomyAuthority, 1)
    authority.mode = "economic"
    authority.authority_epoch = 4
    authority.writes_fenced = True
    db_session.flush()

    with pytest.raises(AuthorityWritesFenced, match="authority_writes_fenced"):
        adapter.apply_legacy_mirror(pending.id, now=NOW)

    assert db_session.query(SocialThemeAssociation).count() == 0


def _saved_work(db_session, *, run_status: str):
    registry = db_session.get(SocialSourceRegistry, 1)
    if registry is None:
        registry = SocialSourceRegistry(id=1, mode="live", provider="official")
        db_session.add(registry)
    item = ContentItem(
        source_type="twitter",
        external_id=f"post-{run_status}",
        url=f"https://x.com/a/status/{run_status}",
        content="Memory pricing rose.",
        published_at=NOW,
    )
    db_session.add(item)
    db_session.flush()
    work = SocialExtractionWork(
        content_item_id=item.id,
        input_hash=f"hash-{run_status}",
        prompt_version="social-v1",
        schema_version="social-v1",
        selected_model="synthetic/model",
        input_snapshot_json={"provider_post_id": item.external_id},
        result_json={"status": "accepted_candidates"},
        state="succeeded",
    )
    run = SocialSignalRun(
        id=f"run-{run_status}",
        registry_id=1,
        registry_version=1,
        mode="live",
        provider="official",
        status="running",
        source_outcomes_json={},
        application_progress_json={},
        feature_run_ids_json={},
        exposure_dates_json={},
        coverage_json={},
    )
    db_session.add_all([work, run])
    db_session.flush()
    db_session.add(
        SocialRunWork(
            run_id=run.id,
            work_id=work.id,
            input_hash=work.input_hash,
            included_at=NOW,
        )
    )
    db_session.flush()
    run.status = "staged"
    db_session.flush()
    if run_status == "published":
        run.status = "published"
        run.published_at = NOW
        db_session.flush()
    return work


def _evidence(work, *, capture_route="social", partial_recapture=False):
    return EvidenceAdmission(
        provider="x",
        canonical_item_id=str(work.content_item_id),
        capture_route=capture_route,
        original_text="Memory pricing rose.",
        preparation_version="social-prep-v1",
        captured_at=NOW,
        observed_at=NOW,
        available_at=NOW,
        evidence_channels=("narrative",),
        source_metadata={
            "archived": "archive" in capture_route,
            "partial_recapture": partial_recapture,
        },
    )


def test_only_published_succeeded_effective_work_is_live_admitted(db_session):
    published = _saved_work(db_session, run_status="published")
    staged = _saved_work(db_session, run_status="staged")
    adapter = EconomicSocialTaxonomyAdapter(db_session)

    admitted = adapter.admit_saved_work(published.id, _evidence(published))
    review_only = adapter.admit_saved_work(staged.id, _evidence(staged))

    assert admitted.precedence_state == "effective"
    assert admitted.live is True
    assert review_only.live is False
    assert review_only.admission_state == "review_only"
    assert db_session.get(EvidencePacket, admitted.packet_id).source_metadata[
        "social_work_id"
    ] == published.id


def test_social_work_is_live_when_legacy_x_content_was_admitted_first(db_session):
    # #500: a legacy-only X source admitted the post first under the same
    # family; Social's differently prepared capture must still go live.
    work = _saved_work(db_session, run_status="published")
    content = EconomicSourceAdmissionService(db_session).admit_content(
        EvidenceAdmission(
            provider="x",
            canonical_source_family=f"x:post:{work.content_item_id}",
            capture_route=CONTENT_INGESTION_ROUTE,
            route_record_id="1:1",
            original_text="Memory pricing rose.",
            preparation_version="content-ingestion-v1",
            captured_at=NOW,
            observed_at=NOW,
            available_at=NOW,
            evidence_channels=("technical",),
        )
    )

    admitted = EconomicSocialTaxonomyAdapter(db_session).admit_saved_work(
        work.id, _evidence(work)
    )

    assert admitted.precedence_state == "effective"
    assert admitted.live is True
    assert (
        db_session.get(EvidencePacket, admitted.packet_id).supersedes_evidence_packet_id
        == content.packet_id
    )


def test_review_only_social_work_does_not_displace_legacy_x_content(db_session):
    # Staged (unpublished) work is review-only; it must not supersede and
    # become the packet discovery processes.
    work = _saved_work(db_session, run_status="staged")
    content = EconomicSourceAdmissionService(db_session).admit_content(
        EvidenceAdmission(
            provider="x",
            canonical_source_family=f"x:post:{work.content_item_id}",
            capture_route=CONTENT_INGESTION_ROUTE,
            route_record_id="1:1",
            original_text="Memory pricing rose.",
            preparation_version="content-ingestion-v1",
            captured_at=NOW,
            observed_at=NOW,
            available_at=NOW,
            evidence_channels=("technical",),
        )
    )

    admitted = EconomicSocialTaxonomyAdapter(db_session).admit_saved_work(
        work.id, _evidence(work)
    )

    assert admitted.live is False
    assert admitted.precedence_state == "hold_review"
    assert (
        EconomicSourceAdmissionService(db_session)
        .effective_packet(content.source_lineage_id)
        .id
        == content.packet_id
    )


def test_unordered_late_archive_remains_review_only(db_session):
    work = _saved_work(db_session, run_status="published")

    result = EconomicSocialTaxonomyAdapter(db_session).admit_saved_work(
        work.id,
        _evidence(work, capture_route="social-archive"),
    )

    assert result.precedence_state == "hold_review"
    assert result.live is False
    assert result.admission_state == "review_only"


def test_partial_recapture_remains_review_only(db_session):
    work = _saved_work(db_session, run_status="published")

    result = EconomicSocialTaxonomyAdapter(db_session).admit_saved_work(
        work.id,
        _evidence(work, partial_recapture=True),
    )

    assert result.precedence_state == "hold_review"
    assert result.live is False


def test_legacy_rows_are_not_mutated_by_global_projection(db_session):
    theme, security = _global_pair(db_session)
    legacy = _legacy_association(db_session, name="Memory Legacy", state="accepted")
    before = tuple(
        db_session.execute(
            select(*SocialThemeAssociation.__table__.columns).where(
                SocialThemeAssociation.id == legacy.id
            )
        ).one()
    )

    EconomicSocialTaxonomyAdapter(db_session).project_legacy_associations(
        economic_theme_id=theme.id,
        security_id=security.id,
        legacy_association_ids=(legacy.id,),
    )
    after = tuple(
        db_session.execute(
            select(*SocialThemeAssociation.__table__.columns).where(
                SocialThemeAssociation.id == legacy.id
            )
        ).one()
    )

    assert after == before


def test_legacy_projection_uses_single_destination_mapping_fallback(db_session):
    legacy = _legacy_association(db_session, name="Memory Legacy", state="accepted")
    security = StockUniverse(symbol="MU", market="US", is_active=True)
    db_session.add(security)
    repo = EconomicTaxonomyRepository(db_session)
    taxonomy = repo.create_draft(actor="test:social", reason="test")
    theme = repo.create_theme(
        taxonomy.id,
        display_name="Memory",
        definition="Memory demand",
        mechanism="Memory pricing",
        lifecycle="established",
        lifecycle_policy_version="v1",
        actor="test:social",
    )
    repo.set_legacy_disposition(
        taxonomy.id,
        legacy.theme_cluster_id,
        disposition="mapped",
        actor="test:social",
    )
    repo.add_legacy_destination(
        taxonomy.id,
        legacy.theme_cluster_id,
        theme.id,
        actor="test:social",
    )
    taxonomy = repo.seal_draft(taxonomy.id)
    db_session.add(
        TaxonomyAuthority(
            id=1,
            mode="dual",
            processing_taxonomy_version_id=taxonomy.id,
            processing_head_revision=1,
            authority_epoch=1,
            writes_fenced=False,
            semantic_invalidation_revision=0,
            cutover_catch_up_cursor=[],
            rollback_state="ready",
        )
    )
    db_session.flush()

    SocialThemeProjectionService(db_session)._project_legacy_association_to_economic(
        legacy.id
    )

    association = db_session.scalar(select(EconomicSocialAssociation))
    assert association.economic_theme_id == theme.id
    assert association.security_id == security.id


def test_legacy_projection_uses_reviewed_split_social_allocation(db_session):
    legacy = _legacy_association(db_session, name="Refining", state="accepted")
    security = StockUniverse(symbol="MU", market="US", is_active=True)
    db_session.add(security)
    repo = EconomicTaxonomyRepository(db_session)
    taxonomy = repo.create_draft(actor="test:social", reason="split allocation")
    petroleum = repo.create_theme(
        taxonomy.id,
        display_name="Petroleum Refining",
        definition="Petroleum refining economics",
        mechanism="Petroleum refining margins",
        lifecycle="established",
        lifecycle_policy_version="v1",
        actor="test:social",
    )
    metals = repo.create_theme(
        taxonomy.id,
        display_name="Metals Refining",
        definition="Metals refining economics",
        mechanism="Metals refining margins",
        lifecycle="established",
        lifecycle_policy_version="v1",
        actor="test:social",
    )
    repo.set_legacy_disposition(
        taxonomy.id,
        legacy.theme_cluster_id,
        disposition="split_required",
        actor="test:social",
    )
    for theme in (petroleum, metals):
        repo.add_legacy_destination(
            taxonomy.id,
            legacy.theme_cluster_id,
            theme.id,
            actor="test:social",
        )
    repo.allocate_legacy_claim(
        taxonomy.id,
        legacy.theme_cluster_id,
        allocation_kind="social_association",
        allocation_key=f"social_theme_association:{legacy.id}",
        destination_theme_id=petroleum.id,
        actor="test:social",
    )
    taxonomy = repo.seal_draft(taxonomy.id)
    db_session.add(
        TaxonomyAuthority(
            id=1,
            mode="dual",
            processing_taxonomy_version_id=taxonomy.id,
            processing_head_revision=1,
            authority_epoch=1,
            writes_fenced=False,
            semantic_invalidation_revision=0,
            cutover_catch_up_cursor=[],
            rollback_state="ready",
        )
    )
    db_session.flush()

    SocialThemeProjectionService(db_session)._project_legacy_association_to_economic(
        legacy.id
    )

    association = db_session.scalar(select(EconomicSocialAssociation))
    assert association.economic_theme_id == petroleum.id
    assert association.economic_theme_id != metals.id
