"""Versioned issuer–theme dossiers built from verified claims (spec §5.2, §8).

``assess`` loads the dossier's current sealed selection and applies the pure
selection policy (``selection.py``) to one attempt's verified candidates. It
is deterministic over its pinned inputs and never chooses by job completion
time. ``persist_assessment`` appends one dossier revision under the research
writer fence; if another assessment won meanwhile, it re-selects against the
winner instead of overwriting it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.domain.company_exposure.contracts import (
    CoverageItem,
    as_utc,
    canonical_json,
    content_hash,
    utc_now,
)
from app.models.company_exposure import (
    AssessmentClaimSelection,
    AssessmentRevision,
    ClaimEvidenceLink,
    ExposureClaim,
    ExposureClaimRevision,
    IssuerThemeAssessment,
    MaterialityMeasure,
)
from app.services.company_exposure.claims import (
    VERIFICATION_POLICY,
    AssessmentScope,
    VerifiedClaim,
)
from app.services.company_exposure.fence import research_write
from app.services.company_exposure.freshness import claim_freshness
from app.services.company_exposure.holds import HoldRegistry
from app.services.company_exposure.provenance import origin_groups
from app.services.company_exposure.selection import (
    CurrentClaim,
    PlannedHold,
    SelectedClaim,
    SelectionAction,
    candidate_hash,
    decimal_text,
    materiality_payload,
    select_claims,
)

FRESHNESS_POLICY = "freshness-v1"
SELECTION_POLICY = "selection-v1"
SYSTEM_PRINCIPAL = "system:company-exposure-assessment"


def _jsonable(value):
    """Stored JSON never contains UUID/datetime/Decimal objects."""

    return json.loads(canonical_json(value))


def coverage_payload(items) -> list[dict]:
    return sorted(
        (item.to_dict() for item in items),
        key=lambda c: (c["route"], c["outcome"], c["reason"] or ""),
    )


def _gaps(coverage: list[dict]) -> list[dict]:
    return [c for c in coverage if not CoverageItem.from_dict(c).complete]


@dataclass(frozen=True, slots=True)
class AssessmentAttemptInput:
    scope: AssessmentScope
    claims: tuple[VerifiedClaim, ...] = ()
    document_revision_ids: tuple[UUID, ...] = ()
    coverage: tuple[CoverageItem, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    model_attempt_refs: tuple[str, ...] = ()
    request_id: UUID | None = None
    assessed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class AssessmentResult:
    input: AssessmentAttemptInput
    prior_revision_id: UUID | None
    selections: tuple[SelectedClaim, ...]
    planned_holds: tuple[PlannedHold, ...]
    conflicts: tuple[dict, ...]
    set_aside: tuple[dict, ...]
    manifest: dict
    manifest_hash: str
    assessed_at: datetime

    def claim(self, kind: str, product_key: str | None = None) -> SelectedClaim:
        matches = [
            s
            for s in self.selections
            if s.claim_kind == kind
            and (product_key is None or s.product_or_activity_key == product_key)
        ]
        if len(matches) != 1:
            raise LookupError(f"{kind}:{product_key}:{len(matches)}_matches")
        return matches[0]

    @property
    def changes(self) -> tuple[SelectedClaim, ...]:
        return tuple(
            s for s in self.selections if s.action != SelectionAction.CARRIED_FORWARD
        )


@dataclass(frozen=True, slots=True)
class AssessmentRevisionRef:
    id: UUID
    assessment_id: UUID
    revision_number: int
    replayed: bool = False
    unchanged: bool = False
    rebuilt: bool = False
    claim_revision_ids: dict[str, UUID] = field(default_factory=dict)


class ExposureAssessmentService:
    def __init__(self, session: Session, *, clock: Callable[[], datetime] = utc_now):
        self.session = session
        self.clock = clock
        self.holds = HoldRegistry(session, clock=clock)

    # ----------------------------------------------------------------- reads
    def dossier(self, scope: AssessmentScope) -> IssuerThemeAssessment | None:
        return self.session.execute(
            select(IssuerThemeAssessment).where(
                IssuerThemeAssessment.issuer_id == scope.issuer_id,
                IssuerThemeAssessment.economic_theme_id == scope.economic_theme_id,
            )
        ).scalar_one_or_none()

    def latest_revision(self, assessment_id: UUID | None) -> AssessmentRevision | None:
        if assessment_id is None:
            return None
        return self.session.execute(
            select(AssessmentRevision)
            .where(
                AssessmentRevision.assessment_id == assessment_id,
                AssessmentRevision.status == "sealed",
            )
            .order_by(AssessmentRevision.revision_number.desc())
            .limit(1)
        ).scalar_one_or_none()

    def _current(self, revision: AssessmentRevision | None) -> dict[str, CurrentClaim]:
        if revision is None:
            return {}
        rows = self.session.execute(
            select(ExposureClaim, ExposureClaimRevision, MaterialityMeasure.period)
            .join(
                AssessmentClaimSelection,
                AssessmentClaimSelection.claim_id == ExposureClaim.id,
            )
            .join(
                ExposureClaimRevision,
                ExposureClaimRevision.id == AssessmentClaimSelection.claim_revision_id,
            )
            .outerjoin(
                MaterialityMeasure,
                MaterialityMeasure.claim_revision_id == ExposureClaimRevision.id,
            )
            .where(AssessmentClaimSelection.assessment_revision_id == revision.id)
        ).all()
        current = {}
        for claim, claim_revision, period in rows:
            claim_holds = self.holds.active_kinds_for_claim(claim.id)
            current[claim.proposition_key] = CurrentClaim(
                proposition_key=claim.proposition_key,
                claim_id=claim.id,
                claim_revision_id=claim_revision.id,
                claim_kind=claim.claim_kind,
                product_or_activity_key=claim.product_or_activity_key,
                reporting_scope=claim.reporting_scope,
                scope_label=claim.scope_label,
                conclusion=claim_revision.conclusion,
                support_basis=claim_revision.support_basis,
                commercial_status=claim_revision.commercial_status,
                supported_as_of=as_utc(claim_revision.supported_as_of),
                reporting_period=claim_revision.reporting_period,
                fresh_until=as_utc(claim_revision.fresh_until),
                materiality_period=period,
                theme_fingerprint=claim_revision.evaluated_theme_fingerprint,
                claim_holds=claim_holds,
                revision_holds=self.holds.active_kinds_for_claim(
                    claim.id, claim_revision.id
                )
                - claim_holds,
            )
        return current

    # ------------------------------------------------------------ selection
    def assess(self, attempt: AssessmentAttemptInput) -> AssessmentResult:
        dossier = self.dossier(attempt.scope)
        prior = self.latest_revision(dossier.id if dossier else None)
        return self._assess_against(attempt, prior)

    def _assess_against(
        self, attempt: AssessmentAttemptInput, prior: AssessmentRevision | None
    ) -> AssessmentResult:
        scope = attempt.scope
        at = as_utc(attempt.assessed_at) or self.clock()
        selection = select_claims(self._current(prior), attempt.claims, scope, at)
        manifest = {
            "scope": {
                "issuer": scope.issuer_id,
                "theme": scope.economic_theme_id,
                "theme_fingerprint": scope.theme_fingerprint,
                "issuer_links": sorted(scope.link_revision_ids),
            },
            "policies": {
                "verification": VERIFICATION_POLICY,
                "freshness": FRESHNESS_POLICY,
                "selection": SELECTION_POLICY,
            },
            "prior_revision_id": None if prior is None else prior.id,
            "document_revision_ids": sorted(
                str(d) for d in attempt.document_revision_ids
            ),
            "candidates": sorted(candidate_hash(c) for c in attempt.claims),
            "model_attempt_refs": sorted(attempt.model_attempt_refs),
            "coverage": coverage_payload(attempt.coverage),
            "unresolved_questions": sorted(attempt.unresolved_questions),
        }
        return AssessmentResult(
            input=attempt,
            prior_revision_id=None if prior is None else prior.id,
            selections=selection.selections,
            planned_holds=selection.holds,
            conflicts=selection.conflicts,
            set_aside=selection.set_aside,
            manifest=manifest,
            manifest_hash=content_hash(manifest),
            assessed_at=at,
        )

    # --------------------------------------------------------------- writes
    def persist_assessment(
        self, result: AssessmentResult, *, principal: str = SYSTEM_PRINCIPAL
    ) -> AssessmentRevisionRef:
        """Append one dossier revision; never overwrite a concurrent winner.

        No provider or network call happens here; the caller has already
        verified claims outside the writer fence.
        """

        scope = result.input.scope
        with research_write(self.session):
            dossier = self._lock_dossier(scope)
            replay = self._replay(dossier.id, result.manifest_hash)
            if replay is not None:
                return AssessmentRevisionRef(
                    replay.id, dossier.id, replay.revision_number, replayed=True
                )
            latest = self.latest_revision(dossier.id)
            latest_id = None if latest is None else latest.id
            rebuilt = latest_id != result.prior_revision_id
            if rebuilt:
                # Another assessment won: select again against it, keeping both.
                result = self._assess_against(result.input, latest)
                replay = self._replay(dossier.id, result.manifest_hash)
                if replay is not None:
                    return AssessmentRevisionRef(
                        replay.id,
                        dossier.id,
                        replay.revision_number,
                        replayed=True,
                        rebuilt=True,
                    )
            if latest is not None and self._unchanged(result, latest):
                self._apply_holds(result, {})
                return AssessmentRevisionRef(
                    latest.id,
                    dossier.id,
                    latest.revision_number,
                    unchanged=True,
                    rebuilt=rebuilt,
                )
            claim_ids, revision_ids = self._write_selected_claims(scope, result)
            revision = self._write_revision(
                dossier, latest, result, revision_ids, claim_ids, principal, rebuilt
            )
            self._apply_holds(result, claim_ids)
            return AssessmentRevisionRef(
                revision.id,
                dossier.id,
                revision.revision_number,
                rebuilt=rebuilt,
                claim_revision_ids=revision_ids,
            )

    def _replay(
        self, assessment_id: UUID, manifest_hash: str
    ) -> AssessmentRevision | None:
        return self.session.execute(
            select(AssessmentRevision).where(
                AssessmentRevision.assessment_id == assessment_id,
                AssessmentRevision.input_manifest_hash == manifest_hash,
            )
        ).scalar_one_or_none()

    def _lock_dossier(self, scope: AssessmentScope) -> IssuerThemeAssessment:
        if self.dossier(scope) is None:
            try:
                with self.session.begin_nested():
                    self.session.add(
                        IssuerThemeAssessment(
                            issuer_id=scope.issuer_id,
                            economic_theme_id=scope.economic_theme_id,
                        )
                    )
            except IntegrityError:
                pass  # created concurrently; lock the winner below
        return self.session.execute(
            select(IssuerThemeAssessment)
            .where(
                IssuerThemeAssessment.issuer_id == scope.issuer_id,
                IssuerThemeAssessment.economic_theme_id == scope.economic_theme_id,
            )
            .with_for_update()
        ).scalar_one()

    @staticmethod
    def _unchanged(result: AssessmentResult, latest: AssessmentRevision) -> bool:
        """Same selection and the same coverage gaps: nothing to record.

        Successful captures differ only in audit detail (captured/unchanged),
        which lives in capture events, not in a new dossier revision.
        """

        return (
            not result.changes
            and _gaps(result.manifest["coverage"]) == _gaps(latest.coverage or [])
            and result.manifest["unresolved_questions"]
            == (latest.unresolved_questions or [])
        )

    def _write_selected_claims(
        self, scope: AssessmentScope, result: AssessmentResult
    ) -> tuple[dict[str, UUID], dict[str, UUID]]:
        claim_ids: dict[str, UUID] = {}
        revision_ids: dict[str, UUID] = {}
        for selected in result.selections:
            key = selected.proposition_key
            if selected.action == SelectionAction.CARRIED_FORWARD:
                claim_ids[key] = selected.prior_claim_id
                revision_ids[key] = selected.prior_claim_revision_id
            else:
                claim, revision = self._write_claim(scope, selected, result)
                claim_ids[key], revision_ids[key] = claim.id, revision.id
        return claim_ids, revision_ids

    def _write_revision(
        self, dossier, latest, result, revision_ids, claim_ids, principal, rebuilt
    ) -> AssessmentRevision:
        scope = result.input.scope
        selected = {k: str(v) for k, v in sorted(revision_ids.items())}
        manifest = {
            **result.manifest,
            "selected_claim_revisions": selected,
            "recorded_by": principal,
            "rebuilt_against": str(latest.id) if rebuilt and latest else None,
        }
        revision = AssessmentRevision(
            assessment_id=dossier.id,
            issuer_id=scope.issuer_id,
            economic_theme_id=scope.economic_theme_id,
            revision_number=1 if latest is None else latest.revision_number + 1,
            input_manifest_hash=result.manifest_hash,
            input_manifest=_jsonable(manifest),
            prior_revision_id=None if latest is None else latest.id,
            request_id=result.input.request_id,
            coverage=result.manifest["coverage"],
            unresolved_questions=result.manifest["unresolved_questions"],
            conflicts=_jsonable(list(result.conflicts)),
            assessed_at=result.assessed_at,
            status="unsealed",
        )
        self.session.add(revision)
        self.session.flush()
        for choice in result.selections:
            self.session.add(
                AssessmentClaimSelection(
                    assessment_revision_id=revision.id,
                    claim_id=claim_ids[choice.proposition_key],
                    claim_revision_id=revision_ids[choice.proposition_key],
                    issuer_id=scope.issuer_id,
                    economic_theme_id=scope.economic_theme_id,
                    carried_forward=choice.action == SelectionAction.CARRIED_FORWARD,
                    selection_reason=choice.reason,
                )
            )
        self.session.flush()
        self._seal(revision, {"manifest": result.manifest_hash, "selected": selected})
        return revision

    def _claim_row(
        self, scope: AssessmentScope, selected: SelectedClaim
    ) -> ExposureClaim:
        claim = self.session.execute(
            select(ExposureClaim).where(
                ExposureClaim.proposition_key == selected.proposition_key
            )
        ).scalar_one_or_none()
        if claim is None:
            claim = ExposureClaim(
                proposition_key=selected.proposition_key,
                issuer_id=scope.issuer_id,
                economic_theme_id=scope.economic_theme_id,
                claim_kind=selected.claim_kind,
                product_or_activity_key=selected.product_or_activity_key,
                reporting_scope=selected.reporting_scope,
                scope_label=selected.scope_label,
                normalized_proposition=(
                    f"{selected.claim_kind}:{selected.product_or_activity_key}:"
                    f"{selected.reporting_scope}:{selected.scope_label or ''}"
                ),
            )
            self.session.add(claim)
            self.session.flush()
        return claim

    def _write_claim(self, scope, selected: SelectedClaim, result: AssessmentResult):
        candidate = selected.candidate
        claim = self._claim_row(scope, selected)
        number = (
            self.session.execute(
                select(func.max(ExposureClaimRevision.revision_number)).where(
                    ExposureClaimRevision.claim_id == claim.id
                )
            ).scalar()
            or 0
        ) + 1
        _, _, freshness_holds = claim_freshness(
            candidate.claim_kind, candidate.supported_as_of, result.assessed_at
        )
        revision = ExposureClaimRevision(
            claim_id=claim.id,
            issuer_id=scope.issuer_id,
            economic_theme_id=scope.economic_theme_id,
            revision_number=number,
            statement=candidate.statement or claim.normalized_proposition,
            evaluated_theme_fingerprint=scope.theme_fingerprint,
            role=candidate.role,
            commercial_status=selected.commercial_status,
            support_basis=selected.support_basis,
            conclusion=selected.conclusion,
            freshness_state=selected.freshness_state,
            hold_reasons=list(
                dict.fromkeys([*candidate.hold_reasons, *freshness_holds])
            ),
            reporting_period=candidate.reporting_period,
            source_publication_time=as_utc(candidate.source_publication_time),
            assessed_at=result.assessed_at,
            supported_as_of=as_utc(candidate.supported_as_of),
            fresh_until=selected.fresh_until,
            verification_policy_version=VERIFICATION_POLICY,
            model_attempt_refs=sorted(result.input.model_attempt_refs),
            supersedes_revision_id=selected.prior_claim_revision_id,
            supersession_kind=None
            if selected.prior_claim_revision_id is None
            else "supersession",
            status="unsealed",
        )
        self.session.add(revision)
        self.session.flush()
        self._write_evidence(scope, revision, candidate)
        self._seal(revision, {"candidate": candidate_hash(candidate), "number": number})
        return claim, revision

    def _write_evidence(self, scope, revision, candidate: VerifiedClaim) -> None:
        origins = origin_groups(
            self.session, [c.passage_id for c in candidate.evidence]
        )
        for cited in candidate.evidence:
            self.session.add(
                ClaimEvidenceLink(
                    claim_revision_id=revision.id,
                    direction=cited.direction,
                    evidence_role=cited.role.value,
                    passage_id=cited.passage_id,
                    quote=cited.quote,
                    locator={"passage_id": str(cited.passage_id)},
                    attribution={"origin_group": origins.get(cited.passage_id)},
                    join_scope={
                        "issuer": str(scope.issuer_id),
                        "theme": str(scope.economic_theme_id),
                    },
                )
            )
        measure = candidate.materiality
        if measure is not None:
            payload = materiality_payload(measure)
            self.session.add(
                MaterialityMeasure(
                    claim_revision_id=revision.id,
                    basis=payload["basis"],
                    metric=measure.metric,
                    value_low=decimal_text(measure.value),
                    value_high=decimal_text(measure.value_high),
                    qualitative_label=payload["qualitative_label"],
                    unit=measure.unit,
                    currency=measure.currency,
                    denominator_definition=measure.denominator_definition,
                    reporting_scope=measure.reporting_scope,
                    scope_label=measure.scope_label,
                    period=measure.period,
                    formula=measure.formula,
                    operand_refs=payload["operands"],
                    supporting_passage_ids=sorted(
                        {o.passage_id for o in measure.operands if o.passage_id}
                    ),
                    raw_reported=_jsonable(measure.raw_reported),
                    hold_reasons=list(measure.hold_reasons),
                )
            )
        self.session.flush()

    def _seal(self, row, semantics: dict) -> None:
        row.status = "sealed"
        row.semantic_hash = content_hash(semantics)
        row.sealed_at = self.clock()
        self.session.flush()

    def _apply_holds(
        self, result: AssessmentResult, claim_ids: dict[str, UUID]
    ) -> None:
        ids = {
            s.proposition_key: s.prior_claim_id
            for s in result.selections
            if s.prior_claim_id is not None
        } | claim_ids
        for hold in result.planned_holds:
            if hold.proposition_key in ids:
                self.holds.apply(
                    "claim",
                    ids[hold.proposition_key],
                    hold.hold_kind,
                    reason=hold.reason,
                    detail={"manifest": result.manifest_hash},
                )


__all__ = (
    "AssessmentAttemptInput",
    "AssessmentResult",
    "AssessmentRevisionRef",
    "ExposureAssessmentService",
    "coverage_payload",
)
