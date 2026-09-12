"""Opt-in background preparation of source-bound development history."""

import os

from app.celery_app import celery_app


def tracking_enabled():
    return os.environ.get("THEME_DEVELOPMENT_TRACKING_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
    }


@celery_app.task(name="app.tasks.theme_intelligence_tasks.prepare_developments")
def prepare_developments():
    if not tracking_enabled():
        return {"status": "disabled"}
    from app.database import SessionLocal
    from app.services.theme_development_worker import discover, process_one
    from app.tasks.theme_discovery_tasks import _theme_automation_gate_result

    with SessionLocal.begin() as db:
        gate = _theme_automation_gate_result(db)
        if gate is not None:
            return gate
        queued = discover(db)
    processed = sum(bool(process_one(SessionLocal)) for _ in range(2))
    return {"status": "processed", "queued": queued, "processed": processed}


@celery_app.task(name="app.tasks.theme_intelligence_tasks.refresh_groups")
def refresh_groups():
    from app.database import SessionLocal
    from app.models.theme_intelligence import ThemeEquivalenceOperation
    from app.services.theme_group_refresh import refresh_groups as refresh

    with SessionLocal() as db:
        pipelines = (
            db.query(ThemeEquivalenceOperation.pipeline)
            .filter_by(refresh_pending=True)
            .distinct()
            .all()
        )
        return {pipeline: refresh(db, pipeline) for (pipeline,) in pipelines}
