"""Blocking-only safety check for a new automatic use of pinned claims (§11).

``ExposureSafety.evaluate`` can only block or defer. It never substitutes
newer evidence into an old decision; callers recheck immediately before an
automatic action and keep the returned token.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.company_exposure.contracts import (
    FreshnessState,
    as_utc,
    content_hash,
    utc_now,
)
from app.domain.company_exposure.policy import (
    freshness_state,
    is_fresh,
    is_primary_support,
)
from app.models.company_exposure import (
    ExposureClaim,
    ExposureClaimRevision,
    IssuerSecurityLinkRevision,
)
from app.services.company_exposure.holds import HoldRegistry


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    allowed: bool
    evaluated_at: datetime
    reasons: tuple[str, ...]
    token: str
    minimum_expiry: datetime | None = None


class ExposureSafety:
    def __init__(self, session: Session, *, clock: Callable[[], datetime] = utc_now):
        self.session = session
        self.clock = clock
        self.holds = HoldRegistry(session, clock=clock)

    def _claim_reasons(
        self, revision: ExposureClaimRevision, at: datetime
    ) -> list[str]:
        reasons = []
        if revision.status != "sealed":
            reasons.append("unsealed")
        if not is_primary_support(revision.support_basis, revision.conclusion):
            reasons.append("not_primary_supported")
        kind = self.session.get(ExposureClaim, revision.claim_id).claim_kind
        state = freshness_state(kind, as_utc(revision.fresh_until), at)
        if state != FreshnessState.CURRENT:
            reasons.append(state.value)
        reasons.extend(
            f"hold:{kind}"
            for kind in sorted(
                self.holds.active_kinds_for_claim(revision.claim_id, revision.id)
            )
        )
        return reasons

    def _link_reasons(self, link_revision_id: UUID) -> list[str]:
        link = self.session.get(IssuerSecurityLinkRevision, link_revision_id)
        reasons = []
        if link is None or link.state != "accepted":
            reasons.append("issuer_link_changed")
        else:
            latest = self.session.execute(
                select(IssuerSecurityLinkRevision)
                .where(IssuerSecurityLinkRevision.security_id == link.security_id)
                .order_by(IssuerSecurityLinkRevision.revision_number.desc())
                .limit(1)
            ).scalar_one()
            if latest.id != link.id and latest.state in {"accepted", "rejected"}:
                reasons.append("issuer_link_changed")
        reasons.extend(
            f"issuer_link:hold:{h.hold_kind}"
            for h in self.holds.active("issuer_link", link_revision_id)
        )
        return reasons

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
        evaluated: list[str] = []
        for revision_id in claim_revision_ids:
            revision = self.session.get(ExposureClaimRevision, revision_id)
            if revision is None:
                reasons.append(f"{revision_id}:claim_revision_missing")
                continue
            evaluated.append(str(revision.id))
            reasons.extend(
                f"{revision_id}:{r}" for r in self._claim_reasons(revision, at)
            )
            fresh_until = as_utc(revision.fresh_until)
            if is_fresh(fresh_until, at):
                expiries.append(fresh_until)
        if link_revision_id is not None:
            reasons.extend(self._link_reasons(link_revision_id))
        token = content_hash(
            {"at": at, "claims": sorted(evaluated), "reasons": sorted(reasons)}
        )
        return SafetyDecision(
            allowed=not reasons,
            evaluated_at=at,
            reasons=tuple(reasons),
            token=token,
            minimum_expiry=min(expiries) if expiries else None,
        )


__all__ = ("ExposureSafety", "SafetyDecision")
