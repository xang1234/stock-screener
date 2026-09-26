"""Freshness, use holds and blocking-only safety checks (spec §11).

* Freshness attaches to the claim kind and its substantive evidence date:
  450 days for stable roles/products, 180 days for customer relationships
  and commercial transitions; undated support is held from automatic use.
  Materiality is valid for its stated period, not a timer.
* Holds are append-only apply/lift revisions per subject stream; a lift must
  reference the hold and cite new support.
* ``ExposureSafety.evaluate`` can only block or defer. It never substitutes
  newer evidence into an old decision.
* ``refresh_due_holds`` runs locally (no provider call) and records stale
  holds for expired claims, even when research acquisition is disabled.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.domain.company_exposure.contracts import (
    PRIMARY_SUPPORT_BASES,
    ClaimKind,
    FreshnessState,
    content_hash,
    utc_now,
)
from app.domain.company_exposure.policy import freshness_deadline, is_fresh
from app.models.company_exposure import (
    AssessmentClaimSelection,
    AssessmentRevision,
    ExposureClaim,
    ExposureClaimRevision,
    ExposureUseHoldRevision,
    IssuerSecurityLinkRevision,
)
from app.services.company_exposure.fence import research_write

SYSTEM_ACTOR = "system:company-exposure-freshness"


def aware_utc(value: datetime | None) -> datetime | None:
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def claim_freshness(
    kind: ClaimKind | str, supported_as_of: datetime | None, at: datetime
) -> tuple[datetime | None, FreshnessState, tuple[str, ...]]:
    kind = ClaimKind(kind)
    if kind == ClaimKind.MATERIALITY:
        return None, FreshnessState.CURRENT, ()
    deadline = freshness_deadline(aware_utc(supported_as_of), kind)
    if deadline is None:
        return None, FreshnessState.UNDATED, ("undated",)
    if is_fresh(deadline, at):
        return deadline, FreshnessState.CURRENT, ()
    return deadline, FreshnessState.STALE, ("stale",)


def stream_key(subject_kind: str, subject_id, hold_kind: str) -> str:
    return f"{subject_kind}:{subject_id}:{hold_kind}"


class HoldRegistry:
    def __init__(self, session: Session, *, clock: Callable[[], datetime] = utc_now):
        self.session = session
        self.clock = clock

    def _latest(self, key: str) -> ExposureUseHoldRevision | None:
        return self.session.execute(
            select(ExposureUseHoldRevision)
            .where(ExposureUseHoldRevision.stream_key == key)
            .order_by(ExposureUseHoldRevision.revision_number.desc())
            .limit(1)
        ).scalar_one_or_none()

    def active(self, subject_kind: str, subject_id) -> list[ExposureUseHoldRevision]:
        rows = self.session.execute(
            select(ExposureUseHoldRevision).where(
                ExposureUseHoldRevision.subject_kind == subject_kind,
                ExposureUseHoldRevision.subject_id == str(subject_id),
            )
        ).scalars()
        latest: dict[str, ExposureUseHoldRevision] = {}
        for row in rows:
            current = latest.get(row.stream_key)
            if current is None or row.revision_number > current.revision_number:
                latest[row.stream_key] = row
        return [row for row in latest.values() if row.action == "apply"]

    def apply(
        self,
        subject_kind: str,
        subject_id,
        hold_kind: str,
        *,
        reason: str,
        actor: str = SYSTEM_ACTOR,
        detail: dict | None = None,
    ) -> ExposureUseHoldRevision:
        key = stream_key(subject_kind, subject_id, hold_kind)
        latest = self._latest(key)
        if latest is not None and latest.action == "apply":
            return latest
        row = ExposureUseHoldRevision(
            subject_kind=subject_kind,
            subject_id=str(subject_id),
            hold_kind=hold_kind,
            stream_key=key,
            revision_number=(latest.revision_number + 1) if latest else 1,
            action="apply",
            reason=reason,
            actor=actor,
            detail=detail or {},
        )
        self.session.add(row)
        self.session.flush()
        return row

    def lift(
        self,
        hold: ExposureUseHoldRevision,
        *,
        actor: str,
        reason: str,
        lift_support: dict,
    ) -> ExposureUseHoldRevision:
        if not lift_support:
            raise ValueError("lift_requires_new_support")
        latest = self._latest(hold.stream_key)
        if latest is None or latest.id != hold.id or latest.action != "apply":
            raise ValueError("hold_not_active")
        row = ExposureUseHoldRevision(
            subject_kind=hold.subject_kind,
            subject_id=hold.subject_id,
            hold_kind=hold.hold_kind,
            stream_key=hold.stream_key,
            revision_number=latest.revision_number + 1,
            action="lift",
            reason=reason,
            actor=actor,
            lifted_hold_id=hold.id,
            lift_support=lift_support,
            detail={},
        )
        self.session.add(row)
        self.session.flush()
        return row


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    allowed: bool
    evaluated_at: datetime
    reasons: tuple[str, ...]
    token: str
    minimum_expiry: datetime | None = None


class ExposureSafety:
    """Current blocking-only check for a new automatic use of pinned claims."""

    def __init__(self, session: Session, *, clock: Callable[[], datetime] = utc_now):
        self.session = session
        self.clock = clock
        self.holds = HoldRegistry(session, clock=clock)

    def evaluate(
        self,
        claim_revision_ids: Iterable[UUID],
        *,
        link_revision_id: UUID | None = None,
        at: datetime | None = None,
    ) -> SafetyDecision:
        at = at or self.clock()
        reasons: list[str] = []
        expiries: list[datetime] = []
        revisions = []
        for revision_id in claim_revision_ids:
            revision = self.session.get(ExposureClaimRevision, revision_id)
            if revision is None:
                reasons.append(f"{revision_id}:claim_revision_missing")
                continue
            revisions.append(revision)
            claim = self.session.get(ExposureClaim, revision.claim_id)
            fresh_until = aware_utc(revision.fresh_until)
            if revision.status != "sealed":
                reasons.append(f"{revision_id}:unsealed")
            if revision.conclusion != "supported" or revision.support_basis not in {
                b.value for b in PRIMARY_SUPPORT_BASES
            }:
                reasons.append(f"{revision_id}:not_primary_supported")
            if claim.claim_kind != ClaimKind.MATERIALITY.value:
                if fresh_until is None:
                    reasons.append(f"{revision_id}:undated")
                elif not is_fresh(fresh_until, at):
                    reasons.append(f"{revision_id}:stale")
                else:
                    expiries.append(fresh_until)
            for hold in (
                *self.holds.active("claim_revision", revision.id),
                *self.holds.active("claim", revision.claim_id),
            ):
                reasons.append(f"{revision_id}:hold:{hold.hold_kind}")
        if link_revision_id is not None:
            link = self.session.get(IssuerSecurityLinkRevision, link_revision_id)
            latest = None
            if link is not None:
                latest = self.session.execute(
                    select(IssuerSecurityLinkRevision)
                    .where(IssuerSecurityLinkRevision.security_id == link.security_id)
                    .order_by(IssuerSecurityLinkRevision.revision_number.desc())
                    .limit(1)
                ).scalar_one()
            if (
                link is None
                or link.state != "accepted"
                or (
                    latest is not None
                    and latest.id != link.id
                    and latest.state in {"accepted", "rejected"}
                )
            ):
                reasons.append("issuer_link_changed")
            for hold in self.holds.active("issuer_link", link_revision_id):
                reasons.append(f"issuer_link:hold:{hold.hold_kind}")
        token = content_hash(
            {
                "at": at,
                "claims": sorted(str(r.id) for r in revisions),
                "reasons": sorted(reasons),
            }
        )
        return SafetyDecision(
            allowed=not reasons,
            evaluated_at=at,
            reasons=tuple(reasons),
            token=token,
            minimum_expiry=min(expiries) if expiries else None,
        )


@dataclass(frozen=True, slots=True)
class HoldReport:
    evaluated_at: datetime
    new_holds: tuple[UUID, ...]


def current_selected_claim_revisions(session: Session) -> list[ExposureClaimRevision]:
    """Claim revisions selected by each dossier's latest sealed revision."""

    latest = (
        select(
            AssessmentRevision.assessment_id,
            func.max(AssessmentRevision.revision_number).label("number"),
        )
        .where(AssessmentRevision.status == "sealed")
        .group_by(AssessmentRevision.assessment_id)
        .subquery()
    )
    revision_ids = session.execute(
        select(AssessmentRevision.id).join(
            latest,
            (AssessmentRevision.assessment_id == latest.c.assessment_id)
            & (AssessmentRevision.revision_number == latest.c.number),
        )
    ).scalars()
    ids = list(revision_ids)
    if not ids:
        return []
    return list(
        session.execute(
            select(ExposureClaimRevision)
            .join(
                AssessmentClaimSelection,
                AssessmentClaimSelection.claim_revision_id == ExposureClaimRevision.id,
            )
            .where(AssessmentClaimSelection.assessment_revision_id.in_(ids))
        ).scalars()
    )


def refresh_due_holds(
    session: Session, at: datetime | None = None, *, clock=utc_now
) -> HoldReport:
    """Provider-free: hold expired or undated claims from new automatic use."""

    at = at or clock()
    registry = HoldRegistry(session, clock=clock)
    created: list[UUID] = []
    with research_write(session):
        for revision in current_selected_claim_revisions(session):
            claim = session.get(ExposureClaim, revision.claim_id)
            if claim.claim_kind == ClaimKind.MATERIALITY.value:
                continue
            fresh_until = aware_utc(revision.fresh_until)
            if fresh_until is None:
                kind = "undated"
            elif not is_fresh(fresh_until, at):
                kind = "stale"
            else:
                continue
            before = {h.id for h in registry.active("claim_revision", revision.id)}
            hold = registry.apply(
                "claim_revision",
                revision.id,
                kind,
                reason=f"freshness evaluation at {at.isoformat()}",
                detail={
                    "fresh_until": None
                    if fresh_until is None
                    else fresh_until.isoformat()
                },
            )
            if hold.id not in before:
                created.append(hold.id)
    return HoldReport(evaluated_at=at, new_holds=tuple(created))
