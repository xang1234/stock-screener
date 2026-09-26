"""Versioned issuer–theme dossiers built from verified claims (spec §5.2, §8, §11).

``assess`` is deterministic over its pinned inputs: the verified candidates of
one attempt, the dossier's current sealed selection, the theme fingerprint and
the verification/freshness policies. It never chooses by job completion time.

Selection per proposition (issuer, theme, kind, product/activity, scope):

* no current claim -> the candidate is selected as a new claim;
* a candidate replaces the current claim only with strictly newer substantive
  support (the document's own date, never its capture time); an equal date is
  a re-download, mirror or translation and changes nothing (E12);
* older, undated or unverified candidates never replace a selected claim, so a
  late archived capture cannot restore an obsolete exposure (E14) and a failed
  or empty search only worsens coverage (I06);
* equal dates with a different conclusion/status are a conflict: the current
  claim stays selected and the proposition is held;
* claims absent from this attempt are carried forward unchanged, so a newer
  period's materiality coexists with an older still-valid role (E15).

A verified ``exposure_end`` holds the related claims of the same product or
activity; removal itself remains a reviewed decision. ``persist_assessment``
compares the expected prior revision with the dossier's actual latest one and
rebuilds the selection against the winner instead of overwriting it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.domain.company_exposure.contracts import (
    PRIMARY_SUPPORT_BASES,
    ClaimKind,
    Conclusion,
    CoverageItem,
    FreshnessState,
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
from app.services.company_exposure.freshness import (
    HoldRegistry,
    aware_utc,
    claim_freshness,
)
from app.services.company_exposure.materiality import MaterialityMeasureResult

FRESHNESS_POLICY = "freshness-v1"
SELECTION_POLICY = "selection-v1"
SYSTEM_PRINCIPAL = "system:company-exposure-assessment"

# Kinds an explicit business exit holds for the same product/activity.
_ENDED_BY_EXIT = frozenset(
    {
        ClaimKind.PARTICIPATION.value,
        ClaimKind.ROLE.value,
        ClaimKind.PRODUCT_APPLICATION.value,
        ClaimKind.CUSTOMER_RELATIONSHIP.value,
        ClaimKind.COMMERCIAL_STATUS.value,
    }
)


def proposition_key(
    issuer_id,
    economic_theme_id,
    kind: str,
    product_key: str,
    scope: str,
    scope_label: str | None,
) -> str:
    return content_hash(
        {
            "issuer": issuer_id,
            "theme": economic_theme_id,
            "kind": kind,
            "product": product_key,
            "scope": scope,
            "scope_label": scope_label,
        }
    )


def _decimal_text(value) -> str | None:
    return None if value is None else format(value, "f")


def materiality_payload(measure: MaterialityMeasureResult | None) -> dict | None:
    if measure is None:
        return None
    return {
        "basis": measure.basis.value,
        "metric": measure.metric,
        "value": _decimal_text(measure.value),
        "value_high": _decimal_text(measure.value_high),
        "unit": measure.unit,
        "currency": measure.currency,
        "period": measure.period,
        "reporting_scope": measure.reporting_scope,
        "scope_label": measure.scope_label,
        "denominator_definition": measure.denominator_definition,
        "qualitative_label": None
        if measure.qualitative_label is None
        else measure.qualitative_label.value,
        "formula": measure.formula,
        "operands": [
            {
                "value": _decimal_text(o.value),
                "unit": o.unit,
                "period": o.period,
                "scope": o.scope,
                "label": o.label,
                "currency": o.currency,
                "passage_id": o.passage_id,
            }
            for o in measure.operands
        ],
        "raw_reported": measure.raw_reported,
        "hold_reasons": list(measure.hold_reasons),
    }


def candidate_hash(candidate: VerifiedClaim) -> str:
    """Semantic identity of a verified candidate (no IDs of the attempt)."""

    return content_hash(
        {
            "kind": candidate.claim_kind.value,
            "product": candidate.product_or_activity_key,
            "scope": candidate.reporting_scope.value,
            "scope_label": candidate.scope_label,
            "statement": candidate.statement,
            "status": candidate.commercial_status.value,
            "basis": candidate.support_basis.value,
            "conclusion": candidate.conclusion.value,
            "role": candidate.role,
            "holds": list(candidate.hold_reasons),
            "evidence": sorted(
                [str(e.passage_id), e.quote, e.role.value, e.direction]
                for e in candidate.evidence
            ),
            "materiality": materiality_payload(candidate.materiality),
            "supported_as_of": aware_utc(candidate.supported_as_of),
            "period": candidate.reporting_period,
        }
    )


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
class SelectedClaim:
    proposition_key: str
    claim_kind: str
    product_or_activity_key: str
    reporting_scope: str
    scope_label: str | None
    action: str  # new | replaced | carried_forward
    reason: str
    conclusion: str
    support_basis: str
    commercial_status: str
    supported_as_of: datetime | None
    reporting_period: str | None
    fresh_until: datetime | None
    freshness_state: str
    materiality_period: str | None = None
    hold_kinds: tuple[str, ...] = ()
    candidate: VerifiedClaim | None = None
    prior_claim_id: UUID | None = None
    prior_claim_revision_id: UUID | None = None

    @property
    def period(self) -> str | None:
        return self.materiality_period or self.reporting_period

    @property
    def ended(self) -> bool:
        return "exposure_end" in self.hold_kinds


@dataclass(frozen=True, slots=True)
class PlannedHold:
    proposition_key: str
    hold_kind: str
    reason: str


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
        return tuple(s for s in self.selections if s.action != "carried_forward")


@dataclass(frozen=True, slots=True)
class AssessmentRevisionRef:
    id: UUID
    assessment_id: UUID
    revision_number: int
    replayed: bool = False
    unchanged: bool = False
    rebuilt: bool = False
    claim_revision_ids: dict[str, UUID] = field(default_factory=dict)


@dataclass(slots=True)
class _Current:
    claim: ExposureClaim
    revision: ExposureClaimRevision
    materiality_period: str | None


def _verified(candidate: VerifiedClaim) -> bool:
    return (
        candidate.support_basis in PRIMARY_SUPPORT_BASES
        and candidate.conclusion in {Conclusion.SUPPORTED, Conclusion.DISPUTED}
    )


def _row_verified(revision: ExposureClaimRevision) -> bool:
    return revision.support_basis in {b.value for b in PRIMARY_SUPPORT_BASES} and (
        revision.conclusion in {"supported", "disputed"}
    )


def _rank(candidate: VerifiedClaim) -> tuple:
    date = aware_utc(candidate.supported_as_of)
    return (
        _verified(candidate),
        date is not None,
        date.timestamp() if date else 0,
        len(candidate.evidence),
        candidate_hash(candidate),
    )


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

    def _current(self, revision: AssessmentRevision | None) -> dict[str, _Current]:
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
        return {
            claim.proposition_key: _Current(claim, claim_revision, period)
            for claim, claim_revision, period in rows
        }

    def _active_hold_kinds(self, current: _Current | None) -> set[str]:
        if current is None:
            return set()
        return {
            h.hold_kind
            for h in (
                *self.holds.active("claim", current.claim.id),
                *self.holds.active("claim_revision", current.revision.id),
            )
        }

    # ------------------------------------------------------------ selection
    def assess(
        self,
        attempt: AssessmentAttemptInput,
        *,
        prior: AssessmentRevision | None = None,
        use_latest: bool = True,
    ) -> AssessmentResult:
        scope = attempt.scope
        at = aware_utc(attempt.assessed_at) or self.clock()
        if use_latest and prior is None:
            dossier = self.dossier(scope)
            prior = self.latest_revision(dossier.id if dossier else None)
        current = self._current(prior)

        grouped: dict[str, list[VerifiedClaim]] = {}
        for candidate in attempt.claims:
            key = proposition_key(
                scope.issuer_id,
                scope.economic_theme_id,
                candidate.claim_kind.value,
                candidate.product_or_activity_key,
                candidate.reporting_scope.value,
                candidate.scope_label,
            )
            grouped.setdefault(key, []).append(candidate)

        selections: dict[str, SelectedClaim] = {}
        holds: list[PlannedHold] = []
        conflicts: list[dict] = []
        set_aside: list[dict] = []

        for key in sorted(set(grouped) | set(current)):
            existing = current.get(key)
            ranked = sorted(grouped.get(key, []), key=_rank, reverse=True)
            best = ranked[0] if ranked else None
            for other in ranked[1:]:
                set_aside.append(
                    {
                        "proposition": key,
                        "reason": "duplicate_candidate",
                        "candidate": candidate_hash(other),
                    }
                )
                if (
                    aware_utc(other.supported_as_of) == aware_utc(best.supported_as_of)
                    and _verified(other)
                    and _verified(best)
                    and (other.conclusion, other.commercial_status)
                    != (best.conclusion, best.commercial_status)
                ):
                    conflicts.append(
                        {"proposition": key, "reason": "same_date_disagreement"}
                    )
                    holds.append(
                        PlannedHold(
                            key,
                            "conflict",
                            "candidates disagree at the same substantive date",
                        )
                    )
            decision = self._decide(existing, best)
            action, reason = decision
            if action == "carried_forward" and best is not None:
                set_aside.append(
                    {
                        "proposition": key,
                        "reason": reason,
                        "candidate": candidate_hash(best),
                    }
                )
                if reason == "same_date_disagreement":
                    conflicts.append({"proposition": key, "reason": reason})
                    holds.append(
                        PlannedHold(
                            key,
                            "conflict",
                            "newer capture disagrees at the same substantive date",
                        )
                    )
            if action == "carried_forward":
                selections[key] = self._carried(key, existing, reason, at)
                if (
                    existing.revision.evaluated_theme_fingerprint
                    != scope.theme_fingerprint
                ):
                    holds.append(
                        PlannedHold(
                            key, "policy", "theme definition changed since verification"
                        )
                    )
            else:
                selections[key] = self._selected(
                    key, best, existing, action, reason, scope, at
                )
                if best.conclusion == Conclusion.DISPUTED:
                    holds.append(
                        PlannedHold(key, "disputed", "conflicting primary evidence")
                    )

        # An explicit, verified exit holds related claims; removal is reviewed.
        for key, selected in selections.items():
            if (
                selected.claim_kind != ClaimKind.EXPOSURE_END.value
                or selected.conclusion != "supported"
            ):
                continue
            if selected.support_basis not in {b.value for b in PRIMARY_SUPPORT_BASES}:
                continue
            for other_key, other in selections.items():
                if (
                    other.claim_kind in _ENDED_BY_EXIT
                    and other.product_or_activity_key
                    == selected.product_or_activity_key
                ):
                    holds.append(
                        PlannedHold(
                            other_key, "exposure_end", "explicit business exit recorded"
                        )
                    )

        holds = list({(h.proposition_key, h.hold_kind): h for h in holds}.values())
        planned: dict[str, set[str]] = {}
        for hold in holds:
            planned.setdefault(hold.proposition_key, set()).add(hold.hold_kind)
        final = []
        for key in sorted(selections):
            selected = selections[key]
            kinds = set(planned.get(key, ()))
            if selected.action == "carried_forward":
                kinds |= self._active_hold_kinds(current.get(key))
            elif current.get(key) is not None:
                # Holds on the claim stream survive a replacement revision.
                kinds |= {
                    h.hold_kind
                    for h in self.holds.active("claim", current[key].claim.id)
                }
            final.append(_with_holds(selected, tuple(sorted(kinds))))

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
            "coverage": _coverage_payload(attempt.coverage),
            "unresolved_questions": sorted(attempt.unresolved_questions),
        }
        return AssessmentResult(
            input=attempt,
            prior_revision_id=None if prior is None else prior.id,
            selections=tuple(final),
            planned_holds=tuple(
                sorted(holds, key=lambda h: (h.proposition_key, h.hold_kind))
            ),
            conflicts=tuple(conflicts),
            set_aside=tuple(set_aside),
            manifest=manifest,
            manifest_hash=content_hash(manifest),
            assessed_at=at,
        )

    @staticmethod
    def _decide(
        existing: _Current | None, best: VerifiedClaim | None
    ) -> tuple[str, str]:
        if best is None:
            return "carried_forward", "not_in_attempt"
        if existing is None:
            return "new", "first_support"
        prior = existing.revision
        prior_date = aware_utc(prior.supported_as_of)
        date = aware_utc(best.supported_as_of)
        if not _verified(best) and _row_verified(prior):
            return "carried_forward", "unverified_candidate"
        if date is None and prior_date is not None:
            return "carried_forward", "undated_candidate"
        if prior_date is not None and date < prior_date:
            return "carried_forward", "older_than_selected"
        if prior_date is not None and date == prior_date:
            if (best.conclusion.value, best.commercial_status.value) != (
                prior.conclusion,
                prior.commercial_status,
            ):
                return "carried_forward", "same_date_disagreement"
            return "carried_forward", "same_substantive_date"
        return "replaced", "newer_support"

    def _selected(
        self, key, candidate, existing, action, reason, scope, at
    ) -> SelectedClaim:
        fresh_until, state, _ = claim_freshness(
            candidate.claim_kind, candidate.supported_as_of, at
        )
        return SelectedClaim(
            proposition_key=key,
            claim_kind=candidate.claim_kind.value,
            product_or_activity_key=candidate.product_or_activity_key,
            reporting_scope=candidate.reporting_scope.value,
            scope_label=candidate.scope_label,
            action=action,
            reason=reason,
            conclusion=candidate.conclusion.value,
            support_basis=candidate.support_basis.value,
            commercial_status=candidate.commercial_status.value,
            supported_as_of=aware_utc(candidate.supported_as_of),
            reporting_period=candidate.reporting_period,
            fresh_until=fresh_until,
            freshness_state=state.value,
            materiality_period=None
            if candidate.materiality is None
            else candidate.materiality.period,
            candidate=candidate,
            prior_claim_id=None if existing is None else existing.claim.id,
            prior_claim_revision_id=None if existing is None else existing.revision.id,
        )

    @staticmethod
    def _carried(key, existing: _Current, reason, at) -> SelectedClaim:
        revision, claim = existing.revision, existing.claim
        fresh_until = aware_utc(revision.fresh_until)
        if claim.claim_kind == ClaimKind.MATERIALITY.value:
            state = FreshnessState.CURRENT
        elif fresh_until is None:
            state = FreshnessState.UNDATED
        elif at < fresh_until:
            state = FreshnessState.CURRENT
        else:
            state = FreshnessState.STALE
        return SelectedClaim(
            proposition_key=key,
            claim_kind=claim.claim_kind,
            product_or_activity_key=claim.product_or_activity_key,
            reporting_scope=claim.reporting_scope,
            scope_label=claim.scope_label,
            action="carried_forward",
            reason=reason,
            conclusion=revision.conclusion,
            support_basis=revision.support_basis,
            commercial_status=revision.commercial_status,
            supported_as_of=aware_utc(revision.supported_as_of),
            reporting_period=revision.reporting_period,
            fresh_until=fresh_until,
            freshness_state=state.value,
            materiality_period=existing.materiality_period,
            prior_claim_id=claim.id,
            prior_claim_revision_id=revision.id,
        )

    # --------------------------------------------------------------- writes
    def persist_assessment(
        self,
        result: AssessmentResult,
        *,
        expected_prior_revision_id: UUID | None,
        principal: str = SYSTEM_PRINCIPAL,
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
            rebuilt = False
            if (
                latest_id != expected_prior_revision_id
                or latest_id != result.prior_revision_id
            ):
                # Another assessment won: select again against it, keeping both.
                result = self.assess(result.input, prior=latest, use_latest=False)
                rebuilt = True
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
                self._apply_holds(result)
                return AssessmentRevisionRef(
                    latest.id,
                    dossier.id,
                    latest.revision_number,
                    unchanged=True,
                    rebuilt=rebuilt,
                )

            claim_revision_ids: dict[str, UUID] = {}
            claim_ids: dict[str, UUID] = {}
            for selected in result.selections:
                if selected.action == "carried_forward":
                    claim_revision_ids[selected.proposition_key] = (
                        selected.prior_claim_revision_id
                    )
                    claim_ids[selected.proposition_key] = selected.prior_claim_id
                    continue
                claim, revision = self._write_claim(scope, selected, result)
                claim_revision_ids[selected.proposition_key] = revision.id
                claim_ids[selected.proposition_key] = claim.id

            number = 1 if latest is None else latest.revision_number + 1
            manifest = {
                **result.manifest,
                "selected_claim_revisions": {
                    k: str(v) for k, v in sorted(claim_revision_ids.items())
                },
                "recorded_by": principal,
                "rebuilt_against": None if not rebuilt else str(latest_id),
            }
            revision = AssessmentRevision(
                assessment_id=dossier.id,
                issuer_id=scope.issuer_id,
                economic_theme_id=scope.economic_theme_id,
                revision_number=number,
                input_manifest_hash=result.manifest_hash,
                input_manifest=_jsonable(manifest),
                prior_revision_id=latest_id,
                request_id=result.input.request_id,
                coverage=result.manifest["coverage"],
                unresolved_questions=result.manifest["unresolved_questions"],
                conflicts=_jsonable(list(result.conflicts)),
                assessed_at=result.assessed_at,
                status="unsealed",
            )
            self.session.add(revision)
            self.session.flush()
            for selected in result.selections:
                self.session.add(
                    AssessmentClaimSelection(
                        assessment_revision_id=revision.id,
                        claim_id=claim_ids[selected.proposition_key],
                        claim_revision_id=claim_revision_ids[selected.proposition_key],
                        issuer_id=scope.issuer_id,
                        economic_theme_id=scope.economic_theme_id,
                        carried_forward=selected.action == "carried_forward",
                        selection_reason=selected.reason,
                    )
                )
            self.session.flush()
            self._seal(
                revision,
                {
                    "manifest": result.manifest_hash,
                    "selected": manifest["selected_claim_revisions"],
                },
            )
            self._apply_holds(result, claim_ids=claim_ids)
            return AssessmentRevisionRef(
                revision.id,
                dossier.id,
                number,
                rebuilt=rebuilt,
                claim_revision_ids=claim_revision_ids,
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
        dossier = self.dossier(scope)
        if dossier is None:
            try:
                with self.session.begin_nested():
                    self.session.add(
                        IssuerThemeAssessment(
                            issuer_id=scope.issuer_id,
                            economic_theme_id=scope.economic_theme_id,
                        )
                    )
            except IntegrityError:
                pass
        return self.session.execute(
            select(IssuerThemeAssessment)
            .where(
                IssuerThemeAssessment.issuer_id == scope.issuer_id,
                IssuerThemeAssessment.economic_theme_id == scope.economic_theme_id,
            )
            .with_for_update()
        ).scalar_one()

    def _unchanged(self, result: AssessmentResult, latest: AssessmentRevision) -> bool:
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

    def _write_claim(self, scope, selected: SelectedClaim, result: AssessmentResult):
        candidate = selected.candidate
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
            source_publication_time=aware_utc(candidate.source_publication_time),
            assessed_at=result.assessed_at,
            supported_as_of=aware_utc(candidate.supported_as_of),
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
        for cited in candidate.evidence:
            self.session.add(
                ClaimEvidenceLink(
                    claim_revision_id=revision.id,
                    direction=cited.direction,
                    evidence_role=cited.role.value,
                    passage_id=cited.passage_id,
                    quote=cited.quote,
                    locator={"passage_id": str(cited.passage_id)},
                    attribution={},
                    join_scope={
                        "issuer": str(scope.issuer_id),
                        "theme": str(scope.economic_theme_id),
                    },
                )
            )
        measure = candidate.materiality
        if measure is not None:
            self.session.add(
                MaterialityMeasure(
                    claim_revision_id=revision.id,
                    basis=measure.basis.value,
                    metric=measure.metric,
                    value_low=_decimal_text(measure.value),
                    value_high=_decimal_text(measure.value_high),
                    qualitative_label=None
                    if measure.qualitative_label is None
                    else measure.qualitative_label.value,
                    unit=measure.unit,
                    currency=measure.currency,
                    denominator_definition=measure.denominator_definition,
                    reporting_scope=measure.reporting_scope,
                    scope_label=measure.scope_label,
                    period=measure.period,
                    formula=measure.formula,
                    operand_refs=materiality_payload(measure)["operands"],
                    supporting_passage_ids=sorted(
                        {o.passage_id for o in measure.operands if o.passage_id}
                    ),
                    raw_reported=_jsonable(measure.raw_reported),
                    hold_reasons=list(measure.hold_reasons),
                )
            )
        self.session.flush()
        self._seal(revision, {"candidate": candidate_hash(candidate), "number": number})
        return claim, revision

    def _seal(self, row, semantics: dict) -> None:
        row.status = "sealed"
        row.semantic_hash = content_hash(semantics)
        row.sealed_at = self.clock()
        self.session.flush()

    def _apply_holds(
        self, result: AssessmentResult, *, claim_ids: dict[str, UUID] | None = None
    ) -> None:
        ids = dict(claim_ids or {})
        for selected in result.selections:
            if selected.prior_claim_id is not None:
                ids.setdefault(selected.proposition_key, selected.prior_claim_id)
        for hold in result.planned_holds:
            claim_id = ids.get(hold.proposition_key)
            if claim_id is None:
                continue
            self.holds.apply(
                "claim",
                claim_id,
                hold.hold_kind,
                reason=hold.reason,
                detail={"manifest": result.manifest_hash},
            )


def _with_holds(selected: SelectedClaim, kinds: tuple[str, ...]) -> SelectedClaim:
    from dataclasses import replace

    return replace(selected, hold_kinds=kinds)


def _gaps(coverage: list[dict]) -> list[dict]:
    return [c for c in coverage if c["outcome"] != "complete_for_requested_scope"]


def _coverage_payload(items) -> list[dict]:
    return sorted(
        (
            {
                "route": item.route,
                "outcome": item.outcome.value,
                "reason": item.reason,
                "detail": _jsonable(item.detail),
            }
            for item in items
        ),
        key=lambda c: (c["route"], c["outcome"], c["reason"] or ""),
    )


def _jsonable(value):
    """Round-trip through canonical JSON so stored JSON has no UUID/datetime objects."""

    import json

    from app.domain.company_exposure.contracts import canonical_json

    return json.loads(canonical_json(value))


__all__ = (
    "AssessmentAttemptInput",
    "AssessmentResult",
    "AssessmentRevisionRef",
    "ExposureAssessmentService",
    "PlannedHold",
    "SelectedClaim",
    "candidate_hash",
    "proposition_key",
)
