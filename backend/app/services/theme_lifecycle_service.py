"""Theme lifecycle state-machine transitions + audit trail helpers."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from ..models.theme import ThemeCluster, ThemeLifecycleTransition

LIFECYCLE_STATES = {"candidate", "active", "dormant", "reactivated", "retired"}
ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "candidate": {"active", "retired"},
    "active": {"dormant", "retired"},
    "dormant": {"reactivated", "retired"},
    "reactivated": {"dormant", "retired"},
    "retired": set(),
}


def apply_lifecycle_transition(
    *,
    db: Session,
    theme: ThemeCluster,
    to_state: str,
    actor: str = "system",
    job_name: str | None = None,
    rule_version: str | None = None,
    reason: str | None = None,
    metadata: dict | None = None,
    transitioned_at: datetime | None = None,
) -> ThemeLifecycleTransition:
    """Validate and persist a lifecycle transition with audit metadata."""
    if to_state not in LIFECYCLE_STATES:
        raise ValueError(f"Unsupported lifecycle state: {to_state}")

    from_state = (theme.lifecycle_state or "candidate").strip()
    if from_state not in LIFECYCLE_STATES:
        raise ValueError(f"Unsupported current lifecycle state: {from_state}")
    if to_state == from_state:
        raise ValueError(f"Lifecycle transition must change state: {from_state} -> {to_state}")
    if to_state not in ALLOWED_TRANSITIONS[from_state]:
        raise ValueError(f"Invalid lifecycle transition: {from_state} -> {to_state}")

    now = transitioned_at or datetime.utcnow()
    theme.lifecycle_state = to_state
    theme.lifecycle_state_updated_at = now
    theme.lifecycle_state_metadata = metadata
    theme.candidate_since_at = theme.candidate_since_at or theme.first_seen_at or now

    if to_state == "active":
        theme.activated_at = theme.activated_at or now
        theme.is_active = True
    elif to_state == "dormant":
        theme.dormant_at = now
        theme.is_active = True
    elif to_state == "reactivated":
        theme.reactivated_at = now
        theme.is_active = True
    elif to_state == "retired":
        theme.retired_at = now
        theme.is_active = False

    transition = ThemeLifecycleTransition(
        theme_cluster_id=theme.id,
        from_state=from_state,
        to_state=to_state,
        actor=actor or "system",
        job_name=job_name,
        rule_version=rule_version,
        reason=reason,
        transition_metadata=metadata,
        transitioned_at=now,
    )
    db.add(transition)
    return transition


def set_initial_lifecycle_defaults(theme: ThemeCluster, *, now: Optional[datetime] = None) -> None:
    """Ensure newly-created themes start with consistent lifecycle defaults."""
    current = now or datetime.utcnow()
    if not theme.lifecycle_state:
        theme.lifecycle_state = "candidate"
    if not theme.candidate_since_at:
        theme.candidate_since_at = theme.first_seen_at or current
    if not theme.lifecycle_state_updated_at:
        theme.lifecycle_state_updated_at = current
