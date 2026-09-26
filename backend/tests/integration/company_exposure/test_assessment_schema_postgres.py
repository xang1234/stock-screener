from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError

from app.database import engine
from app.domain.company_exposure.contracts import content_hash
from app.models.company_exposure import (
    AssessmentClaimSelection,
    AssessmentRevision,
    IssuerThemeAssessment,
)
from tests.fixtures.company_exposure.factory import (
    FIXED_NOW,
    make_claim,
    make_claim_revision,
    make_issuer,
    make_theme,
)

pytestmark = [
    pytest.mark.skipif(
        engine.dialect.name != "postgresql",
        reason="requires PostgreSQL triggers and composite foreign keys",
    ),
    pytest.mark.exposure_layer("postgres"),
]


def _assessment_revision(db, issuer, theme):
    assessment = IssuerThemeAssessment(issuer_id=issuer.id, economic_theme_id=theme.id)
    db.add(assessment)
    db.flush()
    revision = AssessmentRevision(
        assessment_id=assessment.id,
        issuer_id=issuer.id,
        economic_theme_id=theme.id,
        revision_number=1,
        input_manifest_hash=content_hash({"issuer": issuer.id}),
        input_manifest={},
        coverage={},
        unresolved_questions=[],
        conflicts=[],
        assessed_at=FIXED_NOW,
        status="unsealed",
    )
    db.add(revision)
    db.flush()
    return revision


@pytest.mark.case("I10")
def test_selection_cannot_cross_issuer_or_theme_scope(db_session):
    issuer = make_issuer(db_session, "issuer-a")
    other_theme = make_theme(db_session, "other")
    theme = make_theme(db_session, "memory")
    foreign_claim = make_claim_revision(
        db_session, make_claim(db_session, issuer, other_theme)
    )
    revision = _assessment_revision(db_session, issuer, theme)
    db_session.add(
        AssessmentClaimSelection(
            assessment_revision_id=revision.id,
            claim_id=foreign_claim.claim_id,
            claim_revision_id=foreign_claim.id,
            issuer_id=issuer.id,
            economic_theme_id=theme.id,
            carried_forward=False,
            selection_reason="new_support",
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


@pytest.mark.case("E15")
def test_sealed_claim_revision_rejects_raw_sql_and_late_children(db_session):
    issuer = make_issuer(db_session)
    theme = make_theme(db_session)
    revision = make_claim_revision(db_session, make_claim(db_session, issuer, theme))
    db_session.commit()
    with pytest.raises(DBAPIError, match="sealed_payload_immutable"):
        db_session.execute(
            text(
                "UPDATE company_exposure_claim_revisions SET statement = 'x' "
                "WHERE id = :id"
            ),
            {"id": revision.id},
        )
    db_session.rollback()
    with pytest.raises(DBAPIError, match="sealed_payload_immutable"):
        db_session.execute(
            text(
                "INSERT INTO company_exposure_claim_evidence_links "
                "(id, claim_revision_id, direction, evidence_role, "
                "premise_claim_revision_id, locator, attribution, join_scope) "
                "VALUES (gen_random_uuid(), :rid, 'supporting', 'retrieval_aid_only', "
                ":rid2, '{}', '{}', '{}')"
            ),
            {"rid": revision.id, "rid2": revision.id},
        )
    db_session.rollback()


def test_unsealed_revision_cannot_seal_without_hash(db_session):
    issuer = make_issuer(db_session)
    theme = make_theme(db_session)
    revision = make_claim_revision(
        db_session, make_claim(db_session, issuer, theme), seal=False
    )
    db_session.commit()
    with pytest.raises(DBAPIError, match="payload_immutable"):
        db_session.execute(
            text(
                "UPDATE company_exposure_claim_revisions SET status = 'sealed' "
                "WHERE id = :id"
            ),
            {"id": revision.id},
        )
    db_session.rollback()
