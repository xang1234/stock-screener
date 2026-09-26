from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select

from app.domain.company_exposure.contracts import FreshnessState
from app.models.company_exposure import ExposureClaimRevision, ResearchProviderAttempt
from app.services.company_exposure.freshness import (
    ExposureSafety,
    HoldRegistry,
    claim_freshness,
    refresh_due_holds,
)
from tests.fixtures.company_exposure.factory import verified_claim

ROLE_DATE = datetime(2025, 8, 1, tzinfo=timezone.utc)
ROLE_EXPIRY = ROLE_DATE + timedelta(days=450)


def test_claim_windows_follow_kind_and_substantive_date():
    at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert claim_freshness("role", ROLE_DATE, at) == (
        ROLE_EXPIRY,
        FreshnessState.CURRENT,
        (),
    )
    until, state, holds = claim_freshness(
        "customer_relationship", ROLE_DATE, at.replace(month=3)
    )
    assert (until, state, holds) == (
        ROLE_DATE + timedelta(days=180),
        FreshnessState.STALE,
        ("stale",),
    )
    assert claim_freshness("role", None, at) == (
        None,
        FreshnessState.UNDATED,
        ("undated",),
    )
    assert claim_freshness("materiality", None, at) == (
        None,
        FreshnessState.CURRENT,
        (),
    )


def test_boundary_at_fresh_until_is_stale():
    assert (
        claim_freshness("role", ROLE_DATE, ROLE_EXPIRY - timedelta(microseconds=1))[1]
        == FreshnessState.CURRENT
    )
    assert claim_freshness("role", ROLE_DATE, ROLE_EXPIRY)[1] == FreshnessState.STALE


@pytest.fixture
def selected_role(dossier):
    role = verified_claim(
        "role", passage=dossier.passages["role"], supported_as_of=ROLE_DATE
    )
    _, ref = dossier.persist(dossier.attempt(role))
    return next(iter(ref.claim_revision_ids.values()))


def _no_provider_attempts(db):
    return (
        db.execute(select(func.count()).select_from(ResearchProviderAttempt)).scalar()
        == 0
    )


@pytest.mark.case("I05")
@pytest.mark.case("R12")
@pytest.mark.exposure_layer("unit")
def test_expiry_blocks_new_use_without_research(dossier, selected_role):
    safety = ExposureSafety(dossier.db, clock=dossier.clock.now)
    assert safety.evaluate([selected_role]).allowed
    at = dossier.clock.advance_to(ROLE_EXPIRY)
    decision = safety.evaluate([selected_role], at=at)
    assert decision.allowed is False
    assert decision.reasons == (f"{selected_role}:stale",)

    report = refresh_due_holds(dossier.db, at)
    assert len(report.new_holds) == 1
    assert refresh_due_holds(dossier.db, at).new_holds == ()  # idempotent
    assert _no_provider_attempts(dossier.db)
    # The existing accepted revision is not deleted or rewritten: only held.
    revision = dossier.db.get(ExposureClaimRevision, selected_role)
    assert revision.status == "sealed" and revision.freshness_state == "current"


@pytest.mark.case("I05")
@pytest.mark.exposure_layer("unit")
def test_disputed_hold_blocks_only_its_claim(dossier):
    role = verified_claim(
        "role", passage=dossier.passages["role"], supported_as_of=ROLE_DATE
    )
    other = verified_claim(
        "role",
        passage=dossier.passages["other"],
        product_key="probe-cards",
        supported_as_of=ROLE_DATE,
    )
    _, ref = dossier.persist(dossier.attempt(role, other))
    held_id, free_id = ref.claim_revision_ids.values()
    registry = HoldRegistry(dossier.db)
    hold = registry.apply(
        "claim",
        dossier.db.get(ExposureClaimRevision, held_id).claim_id,
        "disputed",
        reason="customer disputes relationship",
    )
    safety = ExposureSafety(dossier.db, clock=dossier.clock.now)
    assert safety.evaluate([held_id]).reasons == (f"{held_id}:hold:disputed",)
    assert safety.evaluate([free_id]).allowed

    with pytest.raises(ValueError, match="lift_requires_new_support"):
        registry.lift(hold, actor="admin:alice", reason="resolved", lift_support={})
    registry.lift(
        hold,
        actor="admin:alice",
        reason="resolved",
        lift_support={"passage_id": str(dossier.passages["role"].id)},
    )
    assert safety.evaluate([held_id]).allowed


def test_undated_and_unverified_support_are_held_from_automatic_use(dossier):
    undated = verified_claim(
        "role", passage=dossier.passages["role"], supported_as_of=None
    )
    secondary = verified_claim(
        "role",
        passage=dossier.passages["other"],
        product_key="probe-cards",
        supported_as_of=ROLE_DATE,
        basis="secondary_reported",
        conclusion="unknown",
        evidence_role="original_secondary",
    )
    _, ref = dossier.persist(dossier.attempt(undated, secondary))
    safety = ExposureSafety(dossier.db, clock=dossier.clock.now)
    reasons = set(safety.evaluate(ref.claim_revision_ids.values()).reasons)
    ids = {
        dossier.db.get(ExposureClaimRevision, i).support_basis: i
        for i in ref.claim_revision_ids.values()
    }
    assert f"{ids['primary_explicit']}:undated" in reasons
    assert f"{ids['secondary_reported']}:not_primary_supported" in reasons


def test_safety_token_pins_time_and_reasons(dossier, selected_role):
    safety = ExposureSafety(dossier.db, clock=dossier.clock.now)
    first = safety.evaluate([selected_role])
    assert first.token == safety.evaluate([selected_role]).token
    assert first.minimum_expiry == ROLE_EXPIRY
    later = safety.evaluate([selected_role], at=dossier.clock.now() + timedelta(days=1))
    assert later.token != first.token
