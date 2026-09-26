"""Claim freshness and provider-free expiry holds (spec §11).

Freshness attaches to the claim kind and its substantive evidence date:
450 days for stable roles/products, 180 days for customer relationships and
commercial transitions; undated support is held from automatic use.
Materiality is valid for its stated period, not a timer.

``refresh_due_holds`` runs locally (no provider call) and records
stale/undated holds for currently selected claims, even when research
acquisition is disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.domain.company_exposure.contracts import (
    ClaimKind,
    FreshnessState,
    as_utc,
    utc_now,
)
from app.domain.company_exposure.policy import freshness_deadline, freshness_state
from app.models.company_exposure import (
    AssessmentClaimSelection,
    AssessmentRevision,
    ExposureClaim,
    ExposureClaimRevision,
)
from app.services.company_exposure.fence import research_write
from app.services.company_exposure.holds import HoldRegistry


def claim_freshness(
    kind: ClaimKind | str, supported_as_of: datetime | None, at: datetime
) -> tuple[datetime | None, FreshnessState, tuple[str, ...]]:
    """``(fresh_until, state, hold reasons)`` for newly verified support."""

    fresh_until = freshness_deadline(as_utc(supported_as_of), kind)
    state = freshness_state(kind, fresh_until, at)
    holds = () if state == FreshnessState.CURRENT else (state.value,)
    return fresh_until, state, holds


@dataclass(frozen=True, slots=True)
class HoldReport:
    evaluated_at: datetime
    new_holds: tuple[UUID, ...]


def current_selected_claim_revisions(
    session: Session,
) -> list[tuple[ExposureClaim, ExposureClaimRevision]]:
    """Claims selected by each dossier's latest sealed revision."""

    latest = (
        select(
            AssessmentRevision.assessment_id,
            func.max(AssessmentRevision.revision_number).label("number"),
        )
        .where(AssessmentRevision.status == "sealed")
        .group_by(AssessmentRevision.assessment_id)
        .subquery()
    )
    current = (
        select(AssessmentRevision.id)
        .join(
            latest,
            (AssessmentRevision.assessment_id == latest.c.assessment_id)
            & (AssessmentRevision.revision_number == latest.c.number),
        )
        .scalar_subquery()
    )
    return list(
        session.execute(
            select(ExposureClaim, ExposureClaimRevision)
            .join(
                AssessmentClaimSelection,
                AssessmentClaimSelection.claim_revision_id == ExposureClaimRevision.id,
            )
            .join(ExposureClaim, ExposureClaim.id == ExposureClaimRevision.claim_id)
            .where(AssessmentClaimSelection.assessment_revision_id.in_(current))
        ).tuples()
    )


def refresh_due_holds(
    session: Session, at: datetime | None = None, *, clock=utc_now
) -> HoldReport:
    """Provider-free: hold expired or undated claims from new automatic use."""

    at = at or clock()
    registry = HoldRegistry(session, clock=clock)
    created: list[UUID] = []
    with research_write(session):
        for claim, revision in current_selected_claim_revisions(session):
            fresh_until = as_utc(revision.fresh_until)
            state = freshness_state(claim.claim_kind, fresh_until, at)
            if state == FreshnessState.CURRENT:
                continue
            hold, new = registry.apply(
                "claim_revision",
                revision.id,
                state.value,
                reason=f"freshness evaluation at {at.isoformat()}",
                detail={
                    "fresh_until": None
                    if fresh_until is None
                    else fresh_until.isoformat()
                },
            )
            if new:
                created.append(hold.id)
    return HoldReport(evaluated_at=at, new_holds=tuple(created))


__all__ = (
    "HoldReport",
    "claim_freshness",
    "current_selected_claim_revisions",
    "refresh_due_holds",
)
