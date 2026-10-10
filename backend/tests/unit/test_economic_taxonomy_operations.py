from __future__ import annotations

import pytest
from sqlalchemy import select

from app.domain.economic_taxonomy.contracts import AdminPrincipal
from app.infra.db.repositories.economic_taxonomy_repo import EconomicTaxonomyRepository
from app.models.economic_taxonomy import EconomicThemeRevision
from app.models.economic_taxonomy_runtime import (
    ImmutableRuntimePayload,
    SemanticInvalidationRevision,
    TaxonomyAuthority,
    TaxonomyOperationEvent,
    TaxonomyOperationRequest,
    TaxonomySourceRevisionLog,
)
from app.services.economic_taxonomy_operations import (
    EconomicTaxonomyOperationService,
    OperationPreviewChanged,
    TaxonomyReviewForbidden,
)


@pytest.fixture
def admin_principal():
    return AdminPrincipal(
        subject="admin:alice",
        auth_method="admin_api_key",
        roles=frozenset({"taxonomy:review"}),
    )


@pytest.fixture
def seeded_head(db_session):
    repo = EconomicTaxonomyRepository(db_session)
    draft = repo.create_draft(actor="test:seed", reason="operation fixture")
    first = repo.create_theme(
        draft.id,
        display_name="Artificial Intelligence",
        definition="AI economic exposure.",
        mechanism="AI investment cycle",
        lifecycle="established",
        lifecycle_policy_version="lifecycle-v1",
    )
    second = repo.create_theme(
        draft.id,
        display_name="AI Memory",
        definition="Memory exposed to AI demand.",
        mechanism="AI server memory demand",
        lifecycle="established",
        lifecycle_policy_version="lifecycle-v1",
    )
    sealed = repo.seal_draft(draft.id)
    authority = TaxonomyAuthority(
        id=1,
        mode="shadow",
        processing_taxonomy_version_id=sealed.id,
        processing_head_revision=4,
        authority_epoch=8,
        writes_fenced=False,
        semantic_invalidation_revision=0,
        rollback_state="ready",
    )
    db_session.add(authority)
    db_session.commit()
    return sealed, first, second


def test_caller_supplied_actor_is_never_persisted(
    db_session, seeded_head, admin_principal
):
    _sealed, first, _second = seeded_head
    service = EconomicTaxonomyOperationService(db_session)

    preview = service.preview_operation(
        {
            "operation_kind": "rename",
            "theme_id": str(first.id),
            "display_name": "AI",
            "actor": "forged",
        },
        principal=admin_principal,
        expected_epoch=8,
    )
    db_session.commit()

    request = db_session.get(TaxonomyOperationRequest, preview.request_id)
    assert preview.actor_subject == admin_principal.subject
    assert request.actor_subject == admin_principal.subject
    assert "forged" not in str(request.request_payload)


def test_apply_uses_unchanged_preview_and_advances_only_processing_head(
    db_session, seeded_head, admin_principal
):
    sealed, first, _second = seeded_head
    service = EconomicTaxonomyOperationService(db_session)
    preview = service.preview_operation(
        {
            "operation_kind": "rename",
            "theme_id": str(first.id),
            "display_name": "AI",
            "compatibility_intents": ["refresh_legacy_projection"],
        },
        principal=admin_principal,
        expected_epoch=8,
    )
    db_session.commit()

    result = service.apply_operation(
        preview.preview_id,
        principal=admin_principal,
        reason="Use the governed short name.",
        expected_epoch=8,
    )
    db_session.commit()

    authority = db_session.get(TaxonomyAuthority, 1)
    revision = db_session.get(
        EconomicThemeRevision, (result.taxonomy_version_id, first.id)
    )
    assert authority.processing_taxonomy_version_id == result.taxonomy_version_id
    assert authority.processing_head_revision == 5
    assert authority.serving_generation_id is None
    assert result.taxonomy_version_id != sealed.id
    assert revision.display_name == "AI"
    assert result.semantic_hash == preview.after_semantic_hash
    assert db_session.scalar(select(TaxonomySourceRevisionLog)) is not None
    assert [
        event.event_type
        for event in db_session.scalars(
            select(TaxonomyOperationEvent)
            .where(TaxonomyOperationEvent.operation_request_id == preview.request_id)
            .order_by(TaxonomyOperationEvent.sequence_number)
        )
    ] == ["previewed", "reviewed", "applied"]


def test_apply_rejects_candidate_changed_after_preview(
    db_session, seeded_head, admin_principal
):
    _sealed, first, _second = seeded_head
    service = EconomicTaxonomyOperationService(db_session)
    preview = service.preview_operation(
        {
            "operation_kind": "rename",
            "theme_id": str(first.id),
            "display_name": "AI",
        },
        principal=admin_principal,
        expected_epoch=8,
    )
    candidate = db_session.get(
        EconomicThemeRevision, (preview.candidate_taxonomy_version_id, first.id)
    )
    candidate.display_name = "Tampered"
    db_session.commit()

    with pytest.raises(OperationPreviewChanged, match="operation_preview_changed"):
        service.apply_operation(
            preview.preview_id,
            principal=admin_principal,
            reason="Apply reviewed rename.",
            expected_epoch=8,
        )


def test_split_preview_reports_two_destinations_and_no_unallocated_claims(
    db_session, seeded_head, admin_principal
):
    _sealed, first, second = seeded_head
    preview = EconomicTaxonomyOperationService(db_session).preview_operation(
        {
            "operation_kind": "legacy_split",
            "legacy_theme_cluster_id": 42,
            "destination_theme_ids": [str(first.id), str(second.id)],
            "allocations": [
                {
                    "allocation_kind": "claim",
                    "allocation_key": "petroleum",
                    "destination_theme_id": str(first.id),
                },
                {
                    "allocation_kind": "claim",
                    "allocation_key": "metals",
                    "destination_theme_id": str(second.id),
                },
            ],
        },
        principal=admin_principal,
        expected_epoch=8,
    )

    assert preview.destination_count == 2
    assert preview.unallocated_claim_ids == ()


def test_reviewer_role_is_required(db_session, seeded_head):
    _sealed, first, _second = seeded_head
    principal = AdminPrincipal(
        subject="admin:viewer",
        auth_method="admin_api_key",
        roles=frozenset(),
    )

    with pytest.raises(TaxonomyReviewForbidden, match="taxonomy_review_forbidden"):
        EconomicTaxonomyOperationService(db_session).preview_operation(
            {
                "operation_kind": "rename",
                "theme_id": str(first.id),
                "display_name": "AI",
            },
            principal=principal,
            expected_epoch=8,
        )


def test_incompatible_apply_invalidates_prepared_generation(
    db_session, seeded_head, admin_principal, monkeypatch
):
    _sealed, first, _second = seeded_head
    service = EconomicTaxonomyOperationService(db_session)
    preview = service.preview_operation(
        {
            "operation_kind": "rename",
            "theme_id": str(first.id),
            "display_name": "AI",
            "incompatible_with_prepared_generation": True,
        },
        principal=admin_principal,
        expected_epoch=8,
    )
    db_session.commit()
    monkeypatch.setattr(service, "_has_prepared_generation", lambda: True)
    before = db_session.get(TaxonomyAuthority, 1)
    head_before, version_before = (
        before.processing_head_revision,
        before.processing_taxonomy_version_id,
    )

    service.apply_operation(
        preview.preview_id,
        principal=admin_principal,
        reason="Invalidate the prepared artifact.",
        expected_epoch=8,
    )
    db_session.commit()

    authority = db_session.get(TaxonomyAuthority, 1)
    invalidation = db_session.scalar(select(SemanticInvalidationRevision))
    assert authority.semantic_invalidation_revision == 1
    assert invalidation.created_by == admin_principal.subject
    # The invalidation re-locks the authority; the apply's own head advance
    # and new processing version must survive that lock (#556).
    assert authority.processing_head_revision == head_before + 1
    assert authority.processing_taxonomy_version_id != version_before


def test_operation_events_are_append_only(db_session, seeded_head, admin_principal):
    _sealed, first, _second = seeded_head
    preview = EconomicTaxonomyOperationService(db_session).preview_operation(
        {
            "operation_kind": "rename",
            "theme_id": str(first.id),
            "display_name": "AI",
        },
        principal=admin_principal,
        expected_epoch=8,
    )
    db_session.commit()
    event = db_session.scalar(
        select(TaxonomyOperationEvent).where(
            TaxonomyOperationEvent.operation_request_id == preview.request_id
        )
    )

    event.reason = "rewritten"
    with pytest.raises(ImmutableRuntimePayload, match="runtime_payload_immutable"):
        db_session.flush()
