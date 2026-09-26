"""Bounded Celery entry points for company-exposure research.

Every task here routes to the dedicated ``exposure_research`` queue and its
own worker (``celery-exposure-research``), never to the price-fetch or
general queues. Queue isolation does not create extra upstream rate
entitlement: every HTTP attempt still acquires the shared provider pacing
keys. Research mode defaults to ``disabled``; nothing dispatches until an
operator configures it.
"""

from __future__ import annotations

import logging
from uuid import uuid4

from app.celery_app import celery_app
from app.domain.company_exposure.contracts import ResearchMode, TaskOutcome

logger = logging.getLogger(__name__)

EXPOSURE_RESEARCH_QUEUE = "exposure_research"
TASK_PREFIX = "app.tasks.company_exposure_tasks."


def _as_dict(outcome: TaskOutcome) -> dict:
    return {"status": outcome.status, "reason": outcome.reason, **outcome.detail}


@celery_app.task(name=TASK_PREFIX + "process_exposure_work", ignore_result=False)
def process_exposure_work(max_steps: int = 1, *, runner_factory=None) -> dict:
    """Advance at most ``max_steps`` leased verify/refresh stages.

    Each claim is committed before the stage runs, so no row lock spans
    pacing waits, network or provider I/O. A disabled research mode claims
    nothing.
    """

    from app.database import SessionLocal
    from app.infra.db.repositories.company_exposure_work_repo import (
        CompanyExposureWorkRepository,
    )
    from app.services.company_exposure.config import load_config
    from app.services.company_exposure.research import build_runner

    config = load_config()
    if config.research_mode == ResearchMode.DISABLED:
        return _as_dict(TaskOutcome.skipped("research_disabled"))
    factory = runner_factory or build_runner
    worker_id = f"exposure-worker:{uuid4()}"
    session = SessionLocal()
    steps: list[dict] = []
    try:
        runner = factory(session, config)
        repo = CompanyExposureWorkRepository(session, clock=runner.clock)
        for _ in range(max(1, int(max_steps))):
            item = repo.claim_next(worker_id=worker_id)
            if item is None:
                session.commit()
                break
            work_id, lease_token = item.id, item.lease_token
            session.commit()
            try:
                result = runner.run_step(work_id, lease_token)
            except Exception:
                # The lease expires and the stage is retried; nothing is lost.
                session.rollback()
                logger.exception("exposure research stage failed: work_id=%s", work_id)
                steps.append({"work_id": str(work_id), "status": "error"})
                continue
            steps.append(
                {
                    "work_id": str(result.work_id),
                    "request_id": str(result.request_id),
                    "stage": result.stage,
                    "status": result.status,
                    "state": result.state,
                }
            )
    finally:
        session.close()
    if not steps:
        return _as_dict(TaskOutcome.skipped("no_work"))
    return _as_dict(TaskOutcome.completed(steps=steps))
