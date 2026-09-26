"""Pure claim selection for an issuer–theme dossier (spec §5.2, §8, §11).

No database, clock or network: callers pass a snapshot of the dossier's
currently selected claims and the verified candidates of one attempt.

Per proposition (issuer, theme, kind, product/activity, scope):

* no current claim -> the best candidate is selected as a new claim;
* a candidate replaces the current claim only with strictly newer
  substantive support (the document's own date, never its capture time); an
  equal date is a re-download, mirror or translation and changes nothing
  (E12);
* older, undated or unverified candidates never replace a selected claim, so
  a late archived capture cannot restore an obsolete exposure (E14) and a
  failed or empty search only worsens coverage (I06);
* verified sources that disagree at the same substantive date are a
  conflict: nothing is replaced and the proposition is held;
* claims absent from this attempt are carried forward unchanged, so a newer
  period's materiality coexists with an older still-valid role (E15).

A verified ``exposure_end`` holds the related claims of the same product or
activity; removal itself remains a reviewed decision.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from app.domain.company_exposure.contracts import (
    ClaimKind,
    Conclusion,
    as_utc,
    content_hash,
)
from app.domain.company_exposure.policy import freshness_state, is_primary_support
from app.services.company_exposure.claims import AssessmentScope, VerifiedClaim
from app.services.company_exposure.freshness import claim_freshness
from app.services.company_exposure.materiality import MaterialityMeasureResult

# Kinds an explicit business exit holds for the same product/activity.
_ENDED_BY_EXIT = frozenset(
    {
        ClaimKind.PARTICIPATION,
        ClaimKind.ROLE,
        ClaimKind.PRODUCT_APPLICATION,
        ClaimKind.CUSTOMER_RELATIONSHIP,
        ClaimKind.COMMERCIAL_STATUS,
    }
)
SAME_DATE_DISAGREEMENT = "same_date_disagreement"


class SelectionAction(StrEnum):
    NEW = "new"
    REPLACED = "replaced"
    CARRIED_FORWARD = "carried_forward"


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


def decimal_text(value) -> str | None:
    return None if value is None else format(value, "f")


def materiality_payload(measure: MaterialityMeasureResult | None) -> dict | None:
    if measure is None:
        return None
    return {
        "basis": measure.basis.value,
        "metric": measure.metric,
        "value": decimal_text(measure.value),
        "value_high": decimal_text(measure.value_high),
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
                "value": decimal_text(o.value),
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
            "supported_as_of": as_utc(candidate.supported_as_of),
            "period": candidate.reporting_period,
        }
    )


@dataclass(frozen=True, slots=True)
class CurrentClaim:
    """Snapshot of one claim revision the dossier currently selects."""

    proposition_key: str
    claim_id: UUID
    claim_revision_id: UUID
    claim_kind: str
    product_or_activity_key: str
    reporting_scope: str
    scope_label: str | None
    conclusion: str
    support_basis: str
    commercial_status: str
    supported_as_of: datetime | None
    reporting_period: str | None
    fresh_until: datetime | None
    materiality_period: str | None
    theme_fingerprint: str
    claim_holds: frozenset[str] = frozenset()
    revision_holds: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class SelectedClaim:
    proposition_key: str
    claim_kind: str
    product_or_activity_key: str
    reporting_scope: str
    scope_label: str | None
    action: SelectionAction
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
class Selection:
    selections: tuple[SelectedClaim, ...]
    holds: tuple[PlannedHold, ...] = ()
    conflicts: tuple[dict, ...] = ()
    set_aside: tuple[dict, ...] = field(default_factory=tuple)


def _primary(basis, conclusion) -> bool:
    # Selection keeps a disputed primary claim in place (it is held instead).
    return is_primary_support(basis, conclusion, allow_disputed=True)


def _rank(candidate: VerifiedClaim) -> tuple:
    date = as_utc(candidate.supported_as_of)
    return (
        _primary(candidate.support_basis, candidate.conclusion),
        date is not None,
        date.timestamp() if date else 0,
        len(candidate.evidence),
        candidate_hash(candidate),
    )


def _disagree(date_a, signature_a, date_b, signature_b) -> bool:
    return date_a is not None and date_a == date_b and signature_a != signature_b


def _decide(
    existing: CurrentClaim | None, best: VerifiedClaim | None
) -> tuple[SelectionAction, str]:
    if best is None:
        return SelectionAction.CARRIED_FORWARD, "not_in_attempt"
    if existing is None:
        return SelectionAction.NEW, "first_support"
    prior_date = as_utc(existing.supported_as_of)
    date = as_utc(best.supported_as_of)
    if not _primary(best.support_basis, best.conclusion) and _primary(
        existing.support_basis, existing.conclusion
    ):
        return SelectionAction.CARRIED_FORWARD, "unverified_candidate"
    if date is None and prior_date is not None:
        return SelectionAction.CARRIED_FORWARD, "undated_candidate"
    if prior_date is not None and date < prior_date:
        return SelectionAction.CARRIED_FORWARD, "older_than_selected"
    if prior_date is not None and date == prior_date:
        if _disagree(
            date,
            (best.conclusion, best.commercial_status),
            prior_date,
            (existing.conclusion, existing.commercial_status),
        ):
            return SelectionAction.CARRIED_FORWARD, SAME_DATE_DISAGREEMENT
        return SelectionAction.CARRIED_FORWARD, "same_substantive_date"
    return SelectionAction.REPLACED, "newer_support"


def _from_candidate(key, candidate, existing, action, reason, at) -> SelectedClaim:
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
        supported_as_of=as_utc(candidate.supported_as_of),
        reporting_period=candidate.reporting_period,
        fresh_until=fresh_until,
        freshness_state=state.value,
        materiality_period=None
        if candidate.materiality is None
        else candidate.materiality.period,
        candidate=candidate,
        prior_claim_id=None if existing is None else existing.claim_id,
        prior_claim_revision_id=None
        if existing is None
        else existing.claim_revision_id,
    )


def _carried(existing: CurrentClaim, reason: str, at: datetime) -> SelectedClaim:
    fresh_until = as_utc(existing.fresh_until)
    return SelectedClaim(
        proposition_key=existing.proposition_key,
        claim_kind=existing.claim_kind,
        product_or_activity_key=existing.product_or_activity_key,
        reporting_scope=existing.reporting_scope,
        scope_label=existing.scope_label,
        action=SelectionAction.CARRIED_FORWARD,
        reason=reason,
        conclusion=existing.conclusion,
        support_basis=existing.support_basis,
        commercial_status=existing.commercial_status,
        supported_as_of=as_utc(existing.supported_as_of),
        reporting_period=existing.reporting_period,
        fresh_until=fresh_until,
        freshness_state=freshness_state(existing.claim_kind, fresh_until, at).value,
        materiality_period=existing.materiality_period,
        prior_claim_id=existing.claim_id,
        prior_claim_revision_id=existing.claim_revision_id,
    )


def _select_one(key, existing, candidates, scope, at):
    """Selection, holds, conflicts and set-aside candidates of one proposition."""

    ranked = sorted(candidates, key=_rank, reverse=True)
    best = ranked[0] if ranked else None
    set_aside = [
        {
            "proposition": key,
            "reason": "duplicate_candidate",
            "candidate": candidate_hash(c),
        }
        for c in ranked[1:]
    ]
    action, reason = _decide(existing, best)
    if action == SelectionAction.CARRIED_FORWARD and best is not None:
        set_aside.append(
            {"proposition": key, "reason": reason, "candidate": candidate_hash(best)}
        )
    conflict = reason == SAME_DATE_DISAGREEMENT or any(
        _primary(c.support_basis, c.conclusion)
        and _primary(best.support_basis, best.conclusion)
        and _disagree(
            as_utc(c.supported_as_of),
            (c.conclusion, c.commercial_status),
            as_utc(best.supported_as_of),
            (best.conclusion, best.commercial_status),
        )
        for c in ranked[1:]
    )
    holds = []
    if conflict:
        holds.append(
            PlannedHold(
                key, "conflict", "sources disagree at the same substantive date"
            )
        )
    if action == SelectionAction.CARRIED_FORWARD:
        selected = _carried(existing, reason, at)
        if existing.theme_fingerprint != scope.theme_fingerprint:
            holds.append(
                PlannedHold(
                    key, "policy", "theme definition changed since verification"
                )
            )
        inherited = existing.claim_holds | existing.revision_holds
    else:
        selected = _from_candidate(key, best, existing, action, reason, at)
        if best.conclusion == Conclusion.DISPUTED:
            holds.append(PlannedHold(key, "disputed", "conflicting primary evidence"))
        # Holds on the claim stream survive a replacement revision.
        inherited = existing.claim_holds if existing is not None else frozenset()
    conflicts = (
        [{"proposition": key, "reason": SAME_DATE_DISAGREEMENT}] if conflict else []
    )
    return selected, holds, conflicts, set_aside, inherited


def _exit_holds(selections: Iterable[SelectedClaim]) -> list[PlannedHold]:
    selections = list(selections)
    ended_products = {
        s.product_or_activity_key
        for s in selections
        if s.claim_kind == ClaimKind.EXPOSURE_END
        and is_primary_support(s.support_basis, s.conclusion)
    }
    return [
        PlannedHold(
            s.proposition_key, "exposure_end", "explicit business exit recorded"
        )
        for s in selections
        if s.claim_kind in _ENDED_BY_EXIT
        and s.product_or_activity_key in ended_products
    ]


def select_claims(
    current: Mapping[str, CurrentClaim],
    candidates: Iterable[VerifiedClaim],
    scope: AssessmentScope,
    at: datetime,
) -> Selection:
    grouped: dict[str, list[VerifiedClaim]] = {}
    for candidate in candidates:
        key = proposition_key(
            scope.issuer_id,
            scope.economic_theme_id,
            candidate.claim_kind.value,
            candidate.product_or_activity_key,
            candidate.reporting_scope.value,
            candidate.scope_label,
        )
        grouped.setdefault(key, []).append(candidate)

    selections, holds, conflicts, set_aside = {}, [], [], []
    inherited: dict[str, frozenset[str]] = {}
    for key in sorted(set(grouped) | set(current)):
        selected, key_holds, key_conflicts, key_set_aside, key_inherited = _select_one(
            key, current.get(key), grouped.get(key, []), scope, at
        )
        selections[key] = selected
        holds += key_holds
        conflicts += key_conflicts
        set_aside += key_set_aside
        inherited[key] = key_inherited
    holds += _exit_holds(selections.values())

    unique = {(h.proposition_key, h.hold_kind): h for h in holds}
    planned: dict[str, set[str]] = {}
    for key, kind in unique:
        planned.setdefault(key, set()).add(kind)
    return Selection(
        selections=tuple(
            replace(s, hold_kinds=tuple(sorted(planned.get(k, set()) | inherited[k])))
            for k, s in sorted(selections.items())
        ),
        holds=tuple(
            sorted(unique.values(), key=lambda h: (h.proposition_key, h.hold_kind))
        ),
        conflicts=tuple(conflicts),
        set_aside=tuple(set_aside),
    )


__all__ = (
    "CurrentClaim",
    "PlannedHold",
    "SelectedClaim",
    "Selection",
    "SelectionAction",
    "candidate_hash",
    "materiality_payload",
    "proposition_key",
    "select_claims",
)
