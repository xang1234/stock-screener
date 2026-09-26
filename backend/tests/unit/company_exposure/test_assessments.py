from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import func, select

from app.domain.company_exposure.contracts import CoverageItem, CoverageOutcome
from app.models.company_exposure import (
    AssessmentClaimSelection,
    AssessmentRevision,
    ClaimEvidenceLink,
    ExposureClaimRevision,
    MaterialityMeasure,
)
from app.services.company_exposure.freshness import HoldRegistry
from app.services.company_exposure.materiality import validate_measure
from tests.fixtures.company_exposure.factory import verified_claim

ROLE_DATE = datetime(2025, 8, 1, tzinfo=timezone.utc)
FY2024_DATE = datetime(2025, 2, 14, tzinfo=timezone.utc)
FY2025_DATE = datetime(2026, 2, 13, tzinfo=timezone.utc)


def _materiality(dossier, period, date):
    passage = dossier.passages["materiality"]
    measure = validate_measure(
        metric="revenue_percent",
        value=Decimal(20),
        unit="percent",
        period=period,
        scope="segment_or_subsidiary",
        scope_label="Memory test",
        quote="Memory test was 20% of revenue",
        passage_id=str(passage.id),
    )
    assert not measure.held
    return verified_claim(
        "materiality",
        passage=passage,
        quote="Memory test was 20% of revenue",
        supported_as_of=date,
        period=period,
        status="unknown",
        materiality=measure,
    )


def _claim_revisions(db):
    return db.execute(select(func.count()).select_from(ExposureClaimRevision)).scalar()


@pytest.fixture
def partial_refresh(dossier):
    role = verified_claim(
        "role",
        passage=dossier.passages["role"],
        supported_as_of=ROLE_DATE,
        period="FY2024",
    )
    dossier.persist(dossier.attempt(role, _materiality(dossier, "FY2024", FY2024_DATE)))
    return dossier.attempt(_materiality(dossier, "FY2025", FY2025_DATE))


@pytest.mark.case("E15")
@pytest.mark.exposure_layer("unit")
def test_partial_refresh_preserves_role_date(dossier, partial_refresh):
    result = dossier.service.assess(partial_refresh)
    role = result.claim("role")
    assert (role.action, role.reason) == ("carried_forward", "not_in_attempt")
    assert role.supported_as_of == ROLE_DATE
    materiality = result.claim("materiality")
    assert (materiality.action, materiality.period) == ("replaced", "FY2025")

    ref = dossier.service.persist_assessment(
        result, expected_prior_revision_id=result.prior_revision_id
    )
    assert ref.revision_number == 2
    selections = (
        dossier.db.execute(
            select(AssessmentClaimSelection).where(
                AssessmentClaimSelection.assessment_revision_id == ref.id
            )
        )
        .scalars()
        .all()
    )
    assert sorted(s.carried_forward for s in selections) == [False, True]
    periods = (
        dossier.db.execute(
            select(MaterialityMeasure.period).order_by(MaterialityMeasure.period)
        )
        .scalars()
        .all()
    )
    # The FY2024 fact is retained as history, not erased.
    assert periods == ["FY2024", "FY2025"]


@pytest.mark.case("E12")
@pytest.mark.exposure_layer("unit")
def test_redownload_does_not_reaffirm_business_evidence(dossier):
    original = verified_claim(
        "role", passage=dossier.passages["role"], supported_as_of=FY2024_DATE
    )
    _, first = dossier.persist(dossier.attempt(original))
    first_revision = dossier.db.get(
        ExposureClaimRevision, next(iter(first.claim_revision_ids.values()))
    )
    # Support time comes from the document, not the (later) assessment time.
    assert first_revision.supported_as_of.replace(tzinfo=timezone.utc) == FY2024_DATE
    assert first_revision.freshness_state == "stale"

    dossier.clock.advance(days=3)
    redownload = replace(
        original, statement="role hbm-test-equipment (re-captured copy)"
    )
    result = dossier.service.assess(dossier.attempt(redownload))
    role = result.claim("role")
    assert (role.action, role.reason) == ("carried_forward", "same_substantive_date")
    assert role.supported_as_of == FY2024_DATE
    before = _claim_revisions(dossier.db)
    ref = dossier.service.persist_assessment(
        result, expected_prior_revision_id=result.prior_revision_id
    )
    assert ref.unchanged and ref.id == first.id
    assert _claim_revisions(dossier.db) == before


def test_first_assessment_links_evidence_and_seals(dossier):
    role = verified_claim(
        "role", passage=dossier.passages["role"], supported_as_of=ROLE_DATE
    )
    _, ref = dossier.persist(dossier.attempt(role))
    revision = dossier.db.get(AssessmentRevision, ref.id)
    assert revision.status == "sealed" and revision.prior_revision_id is None
    claim_revision = dossier.db.get(
        ExposureClaimRevision,
        ref.claim_revision_ids[next(iter(ref.claim_revision_ids))],
    )
    assert claim_revision.status == "sealed"
    assert claim_revision.fresh_until.replace(tzinfo=timezone.utc) == datetime(
        2026, 10, 25, tzinfo=timezone.utc
    )
    links = dossier.db.execute(select(ClaimEvidenceLink)).scalars().all()
    assert [(link.passage_id, link.evidence_role) for link in links] == [
        (dossier.passages["role"].id, "original_primary")
    ]
    assert (
        revision.input_manifest["recorded_by"] == "system:company-exposure-assessment"
    )


def test_identical_attempt_replays_existing_revision(dossier):
    role = verified_claim(
        "role", passage=dossier.passages["role"], supported_as_of=ROLE_DATE
    )
    result = dossier.service.assess(dossier.attempt(role))
    first = dossier.service.persist_assessment(result, expected_prior_revision_id=None)
    again = dossier.service.persist_assessment(result, expected_prior_revision_id=None)
    assert again.replayed and again.id == first.id
    assert (
        dossier.db.execute(
            select(func.count()).select_from(AssessmentRevision)
        ).scalar()
        == 1
    )


@pytest.mark.case("I06")
@pytest.mark.exposure_layer("unit")
def test_missing_document_worsens_coverage_not_truth(dossier):
    role = verified_claim(
        "role", passage=dossier.passages["role"], supported_as_of=ROLE_DATE
    )
    _, first = dossier.persist(dossier.attempt(role))
    gone = CoverageItem("sec_edgar", CoverageOutcome.FETCH_FAILED, "http_404")
    result, ref = dossier.persist(dossier.attempt(coverage=[gone]))
    selected = result.claim("role")
    assert (selected.action, selected.conclusion) == ("carried_forward", "supported")
    assert ref.revision_number == 2 and not ref.unchanged
    revision = dossier.db.get(AssessmentRevision, ref.id)
    assert revision.coverage == [
        {
            "route": "sec_edgar",
            "outcome": "fetch_failed",
            "reason": "http_404",
            "detail": {},
        }
    ]
    assert ref.claim_revision_ids == first.claim_revision_ids


def test_unverified_newer_candidate_never_replaces_primary_claim(dossier):
    role = verified_claim(
        "role", passage=dossier.passages["role"], supported_as_of=ROLE_DATE
    )
    dossier.persist(dossier.attempt(role))
    rumor = verified_claim(
        "role",
        passage=dossier.passages["other"],
        supported_as_of=FY2025_DATE.replace(month=9),
        basis="secondary_reported",
        conclusion="unknown",
        evidence_role="original_secondary",
    )
    result = dossier.service.assess(dossier.attempt(rumor))
    assert result.claim("role").reason == "unverified_candidate"
    assert result.claim("role").supported_as_of == ROLE_DATE


def test_newer_primary_support_supersedes_with_a_new_revision(dossier):
    role = verified_claim(
        "role", passage=dossier.passages["role"], supported_as_of=ROLE_DATE
    )
    _, first = dossier.persist(dossier.attempt(role))
    newer = replace(role, supported_as_of=datetime(2026, 8, 1, tzinfo=timezone.utc))
    result, ref = dossier.persist(dossier.attempt(newer))
    assert result.claim("role").action == "replaced"
    key = next(iter(ref.claim_revision_ids))
    revision = dossier.db.get(ExposureClaimRevision, ref.claim_revision_ids[key])
    assert revision.revision_number == 2
    assert revision.supersedes_revision_id == first.claim_revision_ids[key]
    assert revision.supersession_kind == "supersession"


def test_disputed_claim_holds_only_its_own_proposition(dossier):
    role = verified_claim(
        "role", passage=dossier.passages["role"], supported_as_of=ROLE_DATE
    )
    customer = verified_claim(
        "customer_relationship",
        passage=dossier.passages["other"],
        product_key="probe-cards",
        supported_as_of=ROLE_DATE,
        conclusion="disputed",
    )
    result, ref = dossier.persist(dossier.attempt(role, customer))
    assert result.claim("customer_relationship").hold_kinds == ("disputed",)
    assert result.claim("role").hold_kinds == ()
    registry = HoldRegistry(dossier.db)
    held = {
        key
        for key, revision_id in ref.claim_revision_ids.items()
        if registry.active(
            "claim", dossier.db.get(ExposureClaimRevision, revision_id).claim_id
        )
    }
    assert held == {result.claim("customer_relationship").proposition_key}


def test_theme_definition_change_holds_carried_claims(dossier):
    role = verified_claim(
        "role", passage=dossier.passages["role"], supported_as_of=ROLE_DATE
    )
    dossier.persist(dossier.attempt(role))
    changed = replace(dossier.scope, theme_fingerprint="e" * 64)
    result = dossier.service.assess(dossier.attempt(scope=changed))
    assert result.claim("role").hold_kinds == ("policy",)


def test_concurrent_winner_is_rebuilt_not_overwritten(dossier):
    role = verified_claim(
        "role", passage=dossier.passages["role"], supported_as_of=ROLE_DATE
    )
    _, base = dossier.persist(dossier.attempt(role))
    materiality = _materiality(dossier, "FY2025", FY2025_DATE)
    customer = verified_claim(
        "customer_relationship",
        passage=dossier.passages["other"],
        product_key="probe-cards",
        supported_as_of=ROLE_DATE,
    )
    first = dossier.service.assess(dossier.attempt(materiality))
    second = dossier.service.assess(dossier.attempt(customer))
    assert first.prior_revision_id == second.prior_revision_id == base.id

    winner = dossier.service.persist_assessment(
        first, expected_prior_revision_id=base.id
    )
    loser = dossier.service.persist_assessment(
        second, expected_prior_revision_id=base.id
    )
    assert (winner.revision_number, loser.revision_number) == (2, 3)
    assert loser.rebuilt
    kinds = {
        dossier.db.get(ExposureClaimRevision, rid).claim_id: rid
        for rid in loser.claim_revision_ids.values()
    }
    assert len(kinds) == 3  # role, winner's materiality and loser's customer claim
    revision = dossier.db.get(AssessmentRevision, loser.id)
    assert revision.prior_revision_id == winner.id
    assert revision.input_manifest["rebuilt_against"] == str(winner.id)
