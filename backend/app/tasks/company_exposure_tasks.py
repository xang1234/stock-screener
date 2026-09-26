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

from app.celery_app import celery_app
from app.domain.company_exposure.contracts import TaskOutcome

logger = logging.getLogger(__name__)

EXPOSURE_RESEARCH_QUEUE = "exposure_research"
TASK_PREFIX = "app.tasks.company_exposure_tasks."


def _as_dict(outcome: TaskOutcome) -> dict:
    return {"status": outcome.status, "reason": outcome.reason, **outcome.detail}


@celery_app.task(name=TASK_PREFIX + "process_exposure_work", ignore_result=False)
def process_exposure_work(max_steps: int = 1) -> dict:
    """Advance at most ``max_steps`` leased research stages.

    The verification stage body is installed by Task 17A; until then the
    task claims no work.
    """

    del max_steps
    return _as_dict(TaskOutcome.skipped("stage_not_installed"))
