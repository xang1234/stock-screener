"""Publish-readiness evaluation for feature store runs.

Pure policy function that decides whether a feature run may be
promoted to PUBLISHED.  No I/O, no side effects.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .models import DQSeverity, RunStatus
from .quality import DQResult


# ---------------------------------------------------------------------------
# Publish Decision Value Object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublishDecision:
    """Result of evaluating whether a run is ready for publishing."""

    allowed: bool
    blocking_checks: tuple[DQResult, ...]  # empty if allowed
    warnings: tuple[DQResult, ...]  # non-blocking failures

    @property
    def reason(self) -> str:
        if self.allowed:
            return "All critical checks passed"
        if not self.blocking_checks:
            return "Run is not in COMPLETED status"
        names = ", ".join(r.check_name for r in self.blocking_checks)
        return f"Blocked by: {names}"


# ---------------------------------------------------------------------------
# Policy Function
# ---------------------------------------------------------------------------


def evaluate_publish_readiness(
    status: RunStatus,
    dq_results: Sequence[DQResult],
) -> PublishDecision:
    """Evaluate whether a run is ready for publishing.

    Only COMPLETED runs can be published.  CRITICAL DQ failures block.
    Returns a PublishDecision with blocking_checks and warnings separated.
    """
    if status != RunStatus.COMPLETED:
        return PublishDecision(
            allowed=False,
            blocking_checks=(),
            warnings=(),
        )

    blocking = tuple(
        r
        for r in dq_results
        if r.severity == DQSeverity.CRITICAL and not r.passed
    )
    warnings = tuple(
        r
        for r in dq_results
        if r.severity == DQSeverity.WARNING and not r.passed
    )

    return PublishDecision(
        allowed=len(blocking) == 0,
        blocking_checks=blocking,
        warnings=warnings,
    )


def evaluate_force_publish(status: RunStatus) -> PublishDecision:
    """Evaluate whether a run may be force-published by an admin.

    Only QUARANTINED runs can be force-published.  Returns a
    ``PublishDecision`` with ``allowed=True`` if the run is quarantined.
    """
    if status != RunStatus.QUARANTINED:
        return PublishDecision(
            allowed=False,
            blocking_checks=(),
            warnings=(),
        )

    return PublishDecision(
        allowed=True,
        blocking_checks=(),
        warnings=(),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PublishDecision",
    "evaluate_publish_readiness",
    "evaluate_force_publish",
]
