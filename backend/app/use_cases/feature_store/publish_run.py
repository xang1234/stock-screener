"""PublishFeatureRunUseCase — evaluate DQ and publish or quarantine a feature run.

Extracted from BuildDailyFeatureSnapshotUseCase so that publish logic
can be invoked independently (e.g., admin force-publishing a quarantined
run).  The build use case delegates to this after scanning completes.

Zero Celery imports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.domain.common.errors import ValidationError
from app.domain.common.uow import UnitOfWork
from app.domain.feature_store.models import RunStatus
from app.domain.feature_store.publish_policy import (
    evaluate_force_publish,
    evaluate_publish_readiness,
)
from app.domain.feature_store.quality import (
    DQInputs,
    DQResult,
    DQThresholds,
    run_all_dq_checks,
)

logger = logging.getLogger(__name__)


# ── Command (input) ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class PublishRunCommand:
    """Immutable value object describing what to publish."""

    run_id: int
    dq_inputs: DQInputs | None = None  # None → load from DB
    force_publish: bool = False
    dq_thresholds: DQThresholds = field(default_factory=DQThresholds)


# ── Result (output) ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class PublishRunResult:
    """What the publish use case returns to the caller."""

    run_id: int
    status: str  # published | quarantined
    dq_passed: bool
    dq_report: tuple[DQResult, ...]
    warnings: tuple[str, ...]


# ── Use Case ─────────────────────────────────────────────────────────────


class PublishFeatureRunUseCase:
    """Evaluate DQ checks and publish or quarantine a completed feature run.

    Supports two modes:
    - **Normal publish**: run must be COMPLETED, DQ checks gate publish
    - **Force publish**: run must be QUARANTINED, admin override
    """

    def execute(
        self,
        uow: UnitOfWork,
        cmd: PublishRunCommand,
    ) -> PublishRunResult:
        run = uow.feature_runs.get_run(cmd.run_id)

        # ── Force publish path ────────────────────────────────
        if cmd.force_publish:
            decision = evaluate_force_publish(run.status)
            if not decision.allowed:
                raise ValidationError(
                    f"Force-publish requires QUARANTINED status, "
                    f"got {run.status.value}"
                )
            uow.feature_runs.publish_atomically(cmd.run_id)
            uow.commit()
            logger.info("Run %d force-published by admin", cmd.run_id)
            return PublishRunResult(
                run_id=cmd.run_id,
                status=RunStatus.PUBLISHED.value,
                dq_passed=False,
                dq_report=(),
                warnings=(
                    "Force-published: DQ checks bypassed by admin",
                ) + run.warnings,
            )

        # ── Normal publish path ───────────────────────────────
        if run.status != RunStatus.COMPLETED:
            raise ValidationError(
                f"Run must be COMPLETED to publish, got {run.status.value}"
            )

        # Load or use pre-computed DQ inputs
        dq_inputs = cmd.dq_inputs
        if dq_inputs is None:
            dq_inputs = uow.feature_store.get_run_dq_inputs(cmd.run_id)

        # Guard: empty universe makes DQ ratios meaningless
        if dq_inputs.expected_row_count == 0:
            raise ValidationError(
                f"Run {cmd.run_id} has no universe — cannot evaluate DQ"
            )

        # Run all 5 DQ checks
        dq_results = run_all_dq_checks(dq_inputs, cmd.dq_thresholds)

        # Evaluate publish readiness
        decision = evaluate_publish_readiness(run.status, dq_results)
        dq_warnings = tuple(r.message for r in dq_results if not r.passed)

        if decision.allowed:
            uow.feature_runs.publish_atomically(cmd.run_id)
            uow.commit()
            logger.info("Run %d published", cmd.run_id)
            return PublishRunResult(
                run_id=cmd.run_id,
                status=RunStatus.PUBLISHED.value,
                dq_passed=True,
                dq_report=tuple(dq_results),
                warnings=dq_warnings,
            )
        else:
            uow.feature_runs.mark_quarantined(cmd.run_id, dq_results)
            uow.commit()
            logger.warning(
                "Run %d quarantined: %s", cmd.run_id, decision.reason
            )
            return PublishRunResult(
                run_id=cmd.run_id,
                status=RunStatus.QUARANTINED.value,
                dq_passed=False,
                dq_report=tuple(dq_results),
                warnings=dq_warnings,
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PublishFeatureRunUseCase",
    "PublishRunCommand",
    "PublishRunResult",
]
