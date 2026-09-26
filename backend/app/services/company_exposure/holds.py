"""Append-only use holds (spec §11).

A hold stream is one ``(subject, hold kind)``; each apply or lift appends a
revision. A lift must reference the active hold and cite new support. Holds
only block new automatic use; they never rewrite a sealed revision.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.domain.company_exposure.contracts import utc_now
from app.models.company_exposure import ExposureUseHoldRevision

SYSTEM_ACTOR = "system:company-exposure-holds"


def stream_key(subject_kind: str, subject_id, hold_kind: str) -> str:
    return f"{subject_kind}:{subject_id}:{hold_kind}"


def _still_applied(rows) -> list[ExposureUseHoldRevision]:
    """Holds whose latest revision in their stream is an apply."""

    latest: dict[str, ExposureUseHoldRevision] = {}
    for row in rows:
        current = latest.get(row.stream_key)
        if current is None or row.revision_number > current.revision_number:
            latest[row.stream_key] = row
    return [row for row in latest.values() if row.action == "apply"]


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

    def _active_for(
        self, subjects: Iterable[tuple[str, object]]
    ) -> list[ExposureUseHoldRevision]:
        """Active holds of several subjects in one query."""

        conditions = [
            and_(
                ExposureUseHoldRevision.subject_kind == kind,
                ExposureUseHoldRevision.subject_id == str(subject_id),
            )
            for kind, subject_id in subjects
        ]
        if not conditions:
            return []
        return _still_applied(
            self.session.execute(
                select(ExposureUseHoldRevision).where(or_(*conditions))
            ).scalars()
        )

    def all_active(self) -> list[ExposureUseHoldRevision]:
        """Every active hold, ordered by stream (operator inspection)."""

        rows = self.session.execute(select(ExposureUseHoldRevision)).scalars()
        return sorted(_still_applied(rows), key=lambda row: row.stream_key)

    def active(self, subject_kind: str, subject_id) -> list[ExposureUseHoldRevision]:
        return self._active_for([(subject_kind, subject_id)])

    def active_kinds_for_claim(
        self, claim_id: UUID, claim_revision_id: UUID | None = None
    ) -> frozenset[str]:
        """Hold kinds on a claim stream and, optionally, one of its revisions."""

        subjects = [("claim", claim_id)]
        if claim_revision_id is not None:
            subjects.append(("claim_revision", claim_revision_id))
        return frozenset(h.hold_kind for h in self._active_for(subjects))

    def apply(
        self,
        subject_kind: str,
        subject_id,
        hold_kind: str,
        *,
        reason: str,
        actor: str = SYSTEM_ACTOR,
        detail: dict | None = None,
    ) -> tuple[ExposureUseHoldRevision, bool]:
        """Return ``(active hold, created)``; an active hold is not re-applied."""

        key = stream_key(subject_kind, subject_id, hold_kind)
        latest = self._latest(key)
        if latest is not None and latest.action == "apply":
            return latest, False
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
        return row, True

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


__all__ = ("HoldRegistry", "stream_key")
