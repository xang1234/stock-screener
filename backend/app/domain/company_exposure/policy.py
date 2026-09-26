"""Pure company-exposure policy decisions.

Functions here take explicit inputs and never read configuration, the
database, the clock or the network. Services call them so that the same
decision is made identically in research, publication and tests.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation

from app.domain.company_exposure.contracts import (
    PRIMARY_SUPPORT_BASES,
    ClaimKind,
    Conclusion,
    FreshnessState,
    ResearchMode,
    SupportBasis,
)

# Spec §11.1 freshness windows for automatic use.
STABLE_ROLE_FRESHNESS_DAYS = 450
TRANSITION_FRESHNESS_DAYS = 180
_TRANSITION_KINDS = frozenset(
    {ClaimKind.CUSTOMER_RELATIONSHIP, ClaimKind.COMMERCIAL_STATUS}
)
_TIME_BOUND_KINDS = frozenset(
    {
        ClaimKind.PARTICIPATION,
        ClaimKind.ROLE,
        ClaimKind.PRODUCT_APPLICATION,
        ClaimKind.CUSTOMER_RELATIONSHIP,
        ClaimKind.COMMERCIAL_STATUS,
        ClaimKind.EXPOSURE_END,
    }
)

# Spec §5.4 bounded synthesis.
MAX_SYNTHESIS_PRIMARY_PREMISES = 3
MAX_SYNTHESIS_LINKS = 2


def may_dispatch_paid_search(
    *, enabled: bool, cap: Decimal | str | None, key_present: bool
) -> bool:
    """A paid search call needs explicit enablement, a positive cap and a key.

    A credential alone never enables spending (spec D13, R01).
    """

    if enabled is not True or not key_present or cap is None:
        return False
    try:
        amount = Decimal(str(cap))
    except (InvalidOperation, ValueError):
        return False
    return amount.is_finite() and amount > 0


def may_dispatch_research(mode: ResearchMode | str) -> bool:
    """Only shadow and live modes may acquire documents or call providers."""

    return ResearchMode(mode) in {ResearchMode.SHADOW, ResearchMode.LIVE}


def freshness_deadline(
    substantive_at: datetime | None, claim_kind: ClaimKind | str
) -> datetime | None:
    """When a claim stops being usable for new automatic actions.

    ``None`` means review-only for automatic use: undated support, or a
    claim kind (materiality) whose validity is its stated period rather
    than a timer.
    """

    kind = ClaimKind(claim_kind)
    if substantive_at is None or kind not in _TIME_BOUND_KINDS:
        return None
    if substantive_at.tzinfo is None:
        raise ValueError("substantive_at must be timezone-aware")
    days = (
        TRANSITION_FRESHNESS_DAYS
        if kind in _TRANSITION_KINDS
        else STABLE_ROLE_FRESHNESS_DAYS
    )
    return substantive_at + timedelta(days=days)


def is_fresh(deadline: datetime | None, at: datetime) -> bool:
    """A claim is current strictly before its deadline."""

    return deadline is not None and at < deadline


def freshness_state(
    claim_kind: ClaimKind | str, fresh_until: datetime | None, at: datetime
) -> FreshnessState:
    """Freshness of a stored claim at ``at`` (materiality has no timer)."""

    if ClaimKind(claim_kind) == ClaimKind.MATERIALITY:
        return FreshnessState.CURRENT
    if fresh_until is None:
        return FreshnessState.UNDATED
    return FreshnessState.CURRENT if is_fresh(fresh_until, at) else FreshnessState.STALE


def is_primary_support(
    basis: SupportBasis | str,
    conclusion: Conclusion | str,
    *,
    allow_disputed: bool = False,
) -> bool:
    """Primary-source support. Selection may keep a disputed claim in
    place; automatic use (``allow_disputed=False``) never accepts one."""

    accepted = {Conclusion.SUPPORTED}
    if allow_disputed:
        accepted.add(Conclusion.DISPUTED)
    return basis in PRIMARY_SUPPORT_BASES and conclusion in accepted


def within_synthesis_bound(primary_leaf_ids, explicit_links) -> bool:
    """Pure admissibility bound; entailment and scope checks remain mandatory."""

    leaves = set(primary_leaf_ids)
    return (
        0 < len(leaves) <= MAX_SYNTHESIS_PRIMARY_PREMISES
        and len(tuple(explicit_links)) <= MAX_SYNTHESIS_LINKS
    )
