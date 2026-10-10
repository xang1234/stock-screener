"""Opt-in background preparation of source-bound development history."""

import os

from app.celery_app import celery_app
from app.services.legacy_theme_write_guard import skip_in_economic_authority


def tracking_enabled():
    return os.environ.get("THEME_DEVELOPMENT_TRACKING_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
    }


def _prepare_developments(*, require_legacy_sources=True):
    from app.database import SessionLocal
    from app.services import theme_development_worker as worker
    from app.tasks import theme_discovery_tasks

    with SessionLocal.begin() as db:
        gate = theme_discovery_tasks._theme_automation_gate_result(
            db, require_legacy_sources=require_legacy_sources
        )
        if gate is not None:
            return gate
        queued = worker.discover(db)
    processed = sum(bool(worker.process_one(SessionLocal)) for _ in range(2))
    return {"status": "processed", "queued": queued, "processed": processed}


# The legacy producer writes legacy links, so it keeps the #472 fence.
_prepare_legacy_developments = skip_in_economic_authority(_prepare_developments)


@celery_app.task(name="app.tasks.theme_intelligence_tasks.prepare_developments")
def prepare_developments():
    """Record developments from the serving authority's evidence (#513).

    Under economic authority the economic producer runs: it writes only
    economic links, fenced by ``producer_write``. Otherwise the legacy producer
    runs under the legacy write fence, which skips it if a cutover lands first.
    """
    if not tracking_enabled():
        return {"status": "disabled"}
    from app.database import SessionLocal
    from app.services.theme_development_preparation import economic_authority

    with SessionLocal() as db:
        economic = economic_authority(db)
    # Economic evidence need not come from a legacy content source (Social).
    return (
        _prepare_developments(require_legacy_sources=False)
        if economic
        else _prepare_legacy_developments()
    )


@celery_app.task(name="app.tasks.theme_intelligence_tasks.refresh_groups")
@skip_in_economic_authority
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
