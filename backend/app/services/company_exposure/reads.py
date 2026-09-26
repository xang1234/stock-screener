"""Operational research reads for the shadow slice (plan Task 21A).

``ResearchJobReader.read`` returns explicitly operational progress.
``ResearchJobReader.preview`` returns the exact sealed dossier revision the
job produced, addressed by job and revision identity — never a mutable
"current issuer profile", never a serving generation, and never membership.
Generation-bound product reads arrive with the publication slice.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.company_exposure.contracts import UNKNOWN_MATERIALITY_WORDING
from app.models.company_exposure import (
    AssessmentClaimSelection,
    AssessmentRevision,
    ClaimEvidenceLink,
    ExposureClaim,
    ExposureClaimRevision,
    ExposurePassage,
    ExposureResearchRequest,
    MaterialityMeasure,
    ResearchEvent,
    ResearchWorkItem,
)
from app.services.company_exposure.freshness import HoldRegistry

MAX_EXCERPT_CHARS = 600
OPERATIONAL_VIEW = "research_progress"
SHADOW_VIEW = "shadow_preview"
_STAGE_ORDER = {"resolve_issuer": 0, "acquire": 1, "verify": 2}


class PreviewUnavailable(LookupError):
    def __init__(self, code: str, state: str | None = None):
        super().__init__(code)
        self.code = code
        self.state = state


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _str(value) -> str | None:
    return None if value is None else str(value)


class ResearchJobReader:
    def __init__(self, session: Session):
        self.session = session

    def _events(self, job_id: UUID) -> list[ResearchEvent]:
        return list(
            self.session.execute(
                select(ResearchEvent)
                .where(ResearchEvent.request_id == job_id)
                .order_by(ResearchEvent.sequence)
            ).scalars()
        )

    def read(self, job_id: UUID) -> dict | None:
        request = self.session.get(ExposureResearchRequest, job_id)
        if request is None:
            return None
        events = self._events(job_id)
        items = sorted(
            self.session.execute(
                select(ResearchWorkItem).where(ResearchWorkItem.request_id == job_id)
            ).scalars(),
            key=lambda i: (_STAGE_ORDER.get(i.stage, len(_STAGE_ORDER)), i.stage),
        )
        state = events[-1].state if events else None
        paused = next(
            (
                e.detail
                for e in reversed(events)
                if e.detail and e.detail.get("condition")
            ),
            None,
        )
        revision_id = self._revision_id(events)
        return {
            "view_kind": OPERATIONAL_VIEW,
            "accepted": False,
            "job_id": str(request.id),
            "root_job_id": str(request.effective_root_id),
            "kind": request.kind,
            "security_id": request.security_id,
            "issuer_id": _str(request.issuer_id),
            "economic_theme_id": str(request.economic_theme_id),
            "market": request.market,
            "requested_by": request.requester_principal,
            "created_at": _iso(request.created_at),
            "state": state,
            "condition": None if paused is None else paused.get("condition"),
            "assessment_revision_id": revision_id,
            "stages": [
                {
                    "stage": item.stage,
                    "status": item.status,
                    "pause_reason": item.pause_reason,
                    "attempts": item.claim_count,
                }
                for item in items
            ],
            "events": [
                {
                    "sequence": e.sequence,
                    "state": e.state,
                    "detail": e.detail,
                    "at": _iso(e.created_at),
                }
                for e in events
            ],
        }

    @staticmethod
    def _revision_id(events: list[ResearchEvent]) -> str | None:
        for event in reversed(events):
            value = (event.detail or {}).get("assessment_revision_id")
            if value:
                return value
        return None

    def preview(self, job_id: UUID) -> dict:
        request = self.session.get(ExposureResearchRequest, job_id)
        if request is None:
            raise PreviewUnavailable("job_not_found")
        events = self._events(job_id)
        state = events[-1].state if events else None
        revision_id = self._revision_id(events)
        if revision_id is None:
            raise PreviewUnavailable("assessment_not_ready", state)
        revision = self.session.get(AssessmentRevision, UUID(revision_id))
        if revision is None or revision.status != "sealed":
            raise PreviewUnavailable("assessment_not_ready", state)
        manifest = revision.input_manifest or {}
        scope = manifest.get("scope", {})
        return {
            "view_kind": SHADOW_VIEW,
            "authoritative_membership": False,
            "job_id": str(request.id),
            "state": state,
            "assessment_id": str(revision.assessment_id),
            "assessment_revision_id": str(revision.id),
            "revision_number": revision.revision_number,
            "input_manifest_hash": revision.input_manifest_hash,
            "assessed_at": _iso(revision.assessed_at),
            "issuer_id": str(revision.issuer_id),
            "issuer_link_revision_ids": scope.get("issuer_links", []),
            "economic_theme_id": str(revision.economic_theme_id),
            "theme_fingerprint": scope.get("theme_fingerprint"),
            "policies": manifest.get("policies", {}),
            "coverage": revision.coverage or [],
            "unresolved_questions": revision.unresolved_questions or [],
            "conflicts": revision.conflicts or [],
            "claims": self._claims(revision),
        }

    def _claims(self, revision: AssessmentRevision) -> list[dict]:
        holds = HoldRegistry(self.session)
        rows = self.session.execute(
            select(AssessmentClaimSelection, ExposureClaimRevision, ExposureClaim)
            .join(
                ExposureClaimRevision,
                ExposureClaimRevision.id == AssessmentClaimSelection.claim_revision_id,
            )
            .join(ExposureClaim, ExposureClaim.id == AssessmentClaimSelection.claim_id)
            .where(AssessmentClaimSelection.assessment_revision_id == revision.id)
            .order_by(ExposureClaim.claim_kind, ExposureClaim.product_or_activity_key)
        ).all()
        claims = []
        for selection, claim_revision, claim in rows:
            active = sorted(
                {
                    h.hold_kind
                    for h in (
                        *holds.active("claim", claim.id),
                        *holds.active("claim_revision", claim_revision.id),
                    )
                }
            )
            claims.append(
                {
                    "claim_id": str(claim.id),
                    "claim_revision_id": str(claim_revision.id),
                    "claim_kind": claim.claim_kind,
                    "product_or_activity_key": claim.product_or_activity_key,
                    "reporting_scope": claim.reporting_scope,
                    "scope_label": claim.scope_label,
                    "statement": claim_revision.statement,
                    "role": claim_revision.role,
                    "commercial_status": claim_revision.commercial_status,
                    "support_basis": claim_revision.support_basis,
                    "conclusion": claim_revision.conclusion,
                    "freshness_state": claim_revision.freshness_state,
                    "supported_as_of": _iso(claim_revision.supported_as_of),
                    "fresh_until": _iso(claim_revision.fresh_until),
                    "reporting_period": claim_revision.reporting_period,
                    "hold_reasons": claim_revision.hold_reasons or [],
                    "active_holds": active,
                    "carried_forward": bool(selection.carried_forward),
                    "selection_reason": selection.selection_reason,
                    "materiality": self._materiality(claim_revision.id),
                    "evidence": self._evidence(claim_revision.id),
                }
            )
        return claims

    def _materiality(self, claim_revision_id: UUID) -> dict:
        measure = self.session.execute(
            select(MaterialityMeasure).where(
                MaterialityMeasure.claim_revision_id == claim_revision_id
            )
        ).scalar_one_or_none()
        if measure is None or measure.basis == "unknown" or measure.hold_reasons:
            return {
                "basis": "unknown" if measure is None else measure.basis,
                "display": UNKNOWN_MATERIALITY_WORDING,
                "hold_reasons": [] if measure is None else measure.hold_reasons,
            }
        return {
            "basis": measure.basis,
            "metric": measure.metric,
            "value": measure.value_low,
            "value_high": measure.value_high,
            "qualitative_label": measure.qualitative_label,
            "unit": measure.unit,
            "currency": measure.currency,
            "period": measure.period,
            "reporting_scope": measure.reporting_scope,
            "scope_label": measure.scope_label,
            "denominator_definition": measure.denominator_definition,
            "hold_reasons": [],
        }

    def _evidence(self, claim_revision_id: UUID) -> list[dict]:
        rows = self.session.execute(
            select(ClaimEvidenceLink, ExposurePassage)
            .outerjoin(
                ExposurePassage, ExposurePassage.id == ClaimEvidenceLink.passage_id
            )
            .where(ClaimEvidenceLink.claim_revision_id == claim_revision_id)
            .order_by(ClaimEvidenceLink.direction, ClaimEvidenceLink.created_at)
        ).all()
        return [
            {
                "link_id": str(link.id),
                "direction": link.direction,
                "evidence_role": link.evidence_role,
                "passage_id": _str(link.passage_id),
                "document_revision_id": None
                if passage is None
                else str(passage.document_revision_id),
                "quote": (link.quote or "")[:MAX_EXCERPT_CHARS],
                "language": None if passage is None else passage.language,
                "locator": None if passage is None else passage.locator,
            }
            for link, passage in rows
        ]


__all__ = ("PreviewUnavailable", "ResearchJobReader")
