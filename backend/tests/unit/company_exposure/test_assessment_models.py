from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from app.domain.company_exposure.contracts import content_hash
from app.models.company_exposure import (
    AssessmentClaimSelection,
    AssessmentRevision,
    ClaimEvidenceLink,
    ExposureUseHoldRevision,
    IssuerThemeAssessment,
    MaterialityMeasure,
)
from app.models.economic_taxonomy_runtime_common import ImmutableRuntimePayload
from tests.fixtures.company_exposure.factory import (
    FIXED_NOW,
    make_claim,
    make_claim_revision,
    make_issuer,
    make_theme,
    seal_row,
)

ROLE_DATE = datetime(2025, 3, 1, tzinfo=timezone.utc)


def _revision(assessment, number, manifest):
    return AssessmentRevision(
        assessment_id=assessment.id,
        issuer_id=assessment.issuer_id,
        economic_theme_id=assessment.economic_theme_id,
        revision_number=number,
        input_manifest_hash=content_hash(manifest),
        input_manifest=manifest,
        coverage={},
        unresolved_questions=[],
        conflicts=[],
        assessed_at=FIXED_NOW,
        status="unsealed",
    )


def _select(revision, claim_revision, *, carried=False):
    return AssessmentClaimSelection(
        assessment_revision_id=revision.id,
        claim_id=claim_revision.claim_id,
        claim_revision_id=claim_revision.id,
        issuer_id=revision.issuer_id,
        economic_theme_id=revision.economic_theme_id,
        carried_forward=carried,
        selection_reason="carried_forward" if carried else "new_support",
    )


@dataclass
class AssessmentRows:
    assessment: IssuerThemeAssessment
    old_role: object
    new_materiality: object
    new_revision_rows: list
    new_selections: list


@pytest.fixture
def assessment_rows(db_session):
    issuer = make_issuer(db_session)
    theme = make_theme(db_session)
    role_claim = make_claim(db_session, issuer, theme, kind="role")
    old_role = make_claim_revision(db_session, role_claim, supported_as_of=ROLE_DATE)
    measure_claim = make_claim(
        db_session, issuer, theme, kind="materiality", product_key="revenue_share"
    )
    new_materiality = make_claim_revision(db_session, measure_claim, period="FY2025")
    assessment = IssuerThemeAssessment(
        issuer_id=issuer.id, economic_theme_id=theme.id
    )
    db_session.add(assessment)
    db_session.flush()
    first = _revision(assessment, 1, {"claims": [str(old_role.id)]})
    db_session.add(first)
    db_session.flush()
    db_session.add(_select(first, old_role))
    db_session.flush()
    seal_row(db_session, first)

    second = _revision(
        assessment, 2, {"claims": [str(old_role.id), str(new_materiality.id)]}
    )
    db_session.add(second)
    db_session.flush()
    selections = [_select(second, old_role, carried=True), _select(second, new_materiality)]
    return AssessmentRows(assessment, old_role, new_materiality, selections, selections)


@pytest.mark.case("E15")
@pytest.mark.exposure_layer("schema")
def test_dossier_revision_can_select_old_role_and_new_measure(db_session, assessment_rows):
    db_session.add_all(assessment_rows.new_revision_rows)
    db_session.flush()
    assert {row.claim_revision_id for row in assessment_rows.new_selections} == {
        assessment_rows.old_role.id,
        assessment_rows.new_materiality.id,
    }
    db_session.expire_all()
    assert assessment_rows.old_role.supported_as_of.replace(
        tzinfo=timezone.utc
    ) == ROLE_DATE


def test_replaying_the_same_input_manifest_cannot_create_a_second_revision(
    db_session, assessment_rows
):
    db_session.add(
        _revision(
            assessment_rows.assessment, 3, {"claims": [str(assessment_rows.old_role.id)]}
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


def test_sealed_revisions_and_their_children_are_immutable(db_session, assessment_rows):
    old_role = assessment_rows.old_role
    old_role.statement = "rewritten"
    with pytest.raises(ImmutableRuntimePayload):
        db_session.flush()
    db_session.rollback()


def test_evidence_cannot_be_added_after_the_claim_revision_is_sealed(db_session):
    issuer = make_issuer(db_session)
    theme = make_theme(db_session)
    revision = make_claim_revision(db_session, make_claim(db_session, issuer, theme))
    db_session.add(
        ClaimEvidenceLink(
            claim_revision_id=revision.id,
            direction="supporting",
            evidence_role="retrieval_aid_only",
            premise_claim_revision_id=make_claim_revision(
                db_session,
                make_claim(db_session, issuer, theme, kind="participation"),
            ).id,
            locator={},
            attribution={},
            join_scope={},
        )
    )
    with pytest.raises(ImmutableRuntimePayload):
        db_session.flush()
    db_session.rollback()


@pytest.mark.case("E10")
@pytest.mark.exposure_layer("schema")
@pytest.mark.parametrize(
    "fields",
    [
        {"evidence_role": "original_primary"},  # primary role without a passage
        {"evidence_role": "retrieval_aid_only", "extra_target": True},
    ],
)
def test_evidence_link_shape_constraints(db_session, fields):
    issuer = make_issuer(db_session)
    theme = make_theme(db_session)
    target = make_claim_revision(
        db_session, make_claim(db_session, issuer, theme, kind="participation")
    )
    revision = make_claim_revision(
        db_session, make_claim(db_session, issuer, theme), seal=False
    )
    link = ClaimEvidenceLink(
        claim_revision_id=revision.id,
        direction="supporting",
        evidence_role=fields["evidence_role"],
        premise_claim_revision_id=target.id,
        locator={},
        attribution={},
        join_scope={},
    )
    if fields.get("extra_target"):
        link.derivative_id = target.id
    db_session.add(link)
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


@pytest.mark.case("E06")
@pytest.mark.exposure_layer("schema")
def test_numeric_materiality_requires_metric_value_and_period(db_session):
    issuer = make_issuer(db_session)
    theme = make_theme(db_session)
    revision = make_claim_revision(
        db_session,
        make_claim(db_session, issuer, theme, kind="materiality", product_key="share"),
        seal=False,
    )
    db_session.add(
        MaterialityMeasure(
            claim_revision_id=revision.id,
            basis="disclosed",
            metric="revenue_share",
            value_low="0.30",
            reporting_scope="segment_or_subsidiary",
            scope_label="Server segment",
            formula={},
            operand_refs=[],
            supporting_passage_ids=[],
            raw_reported={},
            hold_reasons=[],
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


@pytest.mark.case("I05")
@pytest.mark.exposure_layer("schema")
def test_hold_lift_requires_referenced_hold_and_support(db_session):
    apply = ExposureUseHoldRevision(
        subject_kind="claim",
        subject_id="claim-customer",
        hold_kind="disputed",
        stream_key="claim:claim-customer:disputed",
        revision_number=1,
        action="apply",
        reason="conflicting customer statements",
        actor="system:company-exposure-research",
        detail={},
    )
    db_session.add(apply)
    db_session.flush()
    db_session.add(
        ExposureUseHoldRevision(
            subject_kind="claim",
            subject_id="claim-customer",
            hold_kind="disputed",
            stream_key="claim:claim-customer:disputed",
            revision_number=2,
            action="lift",
            reason="resolved",
            actor="test:admin",
            lifted_hold_id=apply.id,
            detail={},
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()
