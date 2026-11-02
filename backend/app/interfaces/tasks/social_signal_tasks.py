"""Dedicated Celery entry points for the private Social Signal pipeline."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone

from sqlalchemy import func, select

from app.celery_app import celery_app


def _now(value=None):
    if value is None:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _run_result(value):
    result = asdict(value)
    result["coverage_summary"] = list(value.coverage_summary)
    result["reason_codes"] = list(value.reason_codes)
    return result


def _schedule_resume(run_id, now):
    from app.config import settings
    from app.database import SessionLocal
    from app.services.social_llm_budget_service import SocialLLMBudgetService
    reset = SocialLLMBudgetService(
        SessionLocal,
        daily_limit_usd=settings.social_llm_daily_budget_usd,
        budget_timezone=settings.social_llm_budget_timezone,
    ).status(now).next_reset_at
    resume_social_analysis.apply_async(args=[run_id], eta=reset, queue="social_ingestion")


@celery_app.task(
    bind=True,
    name="app.interfaces.tasks.social_signal_tasks.refresh_social_signals",
)
def refresh_social_signals(self, origin="scheduled", scheduled_for=None):
    from app.wiring.use_case_factories import get_refresh_social_signals_use_case
    now = _now(scheduled_for)
    result = asyncio.run(get_refresh_social_signals_use_case().execute(origin, now))
    if result.processing_status == "deferred" and "analysis_incomplete" in result.reason_codes:
        _schedule_resume(result.run_id, now)
    return _run_result(result)


def _latest_publication(db):
    """A generation created before the latest publication would republish stale reads."""
    from app.infra.db.models.social_signals import SocialSignalRun
    return db.scalar(select(func.max(SocialSignalRun.created_at)).where(SocialSignalRun.status == "published"))


def _resume_superseded(run_id):
    from app.database import SessionLocal
    from app.infra.db.models.social_signals import SocialSignalRun
    with SessionLocal() as db:
        run = db.get(SocialSignalRun, run_id)
        if run is None:
            return False
        if run.status == "failed":
            return run.application_progress_json.get("failure", {}).get("reason_code") == "generation_superseded"
        # A publication closes the older generations it can see; one created
        # later with an older timestamp is caught here. Admin retries of
        # published or analysis-failed generations are deliberate.
        published_at = _latest_publication(db)
        return run.status == "running" and published_at is not None and run.created_at <= published_at


def _latest_resumable_run(now):
    from app.database import SessionLocal
    from app.infra.db.models.social_signals import SocialSignalRun, SocialSourceRegistry
    from app.infra.db.repositories.social_refresh_support import RETAINED_EVIDENCE
    with SessionLocal() as db:
        registry = db.get(SocialSourceRegistry, 1)
        if registry is None or registry.mode == "off" or registry.provider == "disabled":
            return None
        recent = SocialSignalRun.created_at >= now - RETAINED_EVIDENCE
        candidates = select(SocialSignalRun.id).where(SocialSignalRun.status == "running", recent)
        published_at = _latest_publication(db)
        if published_at is not None:
            candidates = candidates.where(SocialSignalRun.created_at > published_at)
        # A replay is created after the run it replays, so the same bound finds every replay of a candidate.
        replayed_run_ids = set(db.scalars(select(
            SocialSignalRun.application_progress_json[("replay", "saved_run_id")].as_string()
        ).where(recent)))
        candidate_ids = db.scalars(candidates.order_by(
            SocialSignalRun.created_at.desc(), SocialSignalRun.id.desc()
        )).all()
        runs = (db.get(SocialSignalRun, run_id) for run_id in candidate_ids if run_id not in replayed_run_ids)
        return next((run.id for run in runs if (
            set(run.application_progress_json.get("observations", {}))
            == set(run.application_progress_json.get("sources", {}))
            and set(run.source_outcomes_json) == set(run.application_progress_json.get("sources", {}))
            and all(
                run.source_outcomes_json[source_id].get("read_status") in {"success", "failed"}
                for source_id in run.source_outcomes_json
            )
            and any(
                outcome.get("read_status") == "success"
                for outcome in run.source_outcomes_json.values()
            )
        )), None)


@celery_app.task(
    bind=True,
    name="app.interfaces.tasks.social_signal_tasks.resume_social_analysis",
)
def resume_social_analysis(self, saved_run_id=None, scheduled_for=None):
    from app.wiring.use_case_factories import get_refresh_social_signals_use_case
    if saved_run_id is not None and _resume_superseded(saved_run_id):
        return {"status": "skipped", "reason_codes": ["generation_superseded"]}
    now = _now(scheduled_for)
    saved_run_id = saved_run_id or _latest_resumable_run(now)
    if saved_run_id is None:
        return {"status": "skipped", "reason_codes": ["no_resumable_generation"]}
    result = asyncio.run(get_refresh_social_signals_use_case().execute(
        f"budget-resume:{saved_run_id}", now, saved_run_id=saved_run_id
    ))
    if result.processing_status == "deferred":
        _schedule_resume(result.run_id, now)
    return _run_result(result)


@celery_app.task(
    bind=True,
    name="app.interfaces.tasks.social_signal_tasks.validate_social_source",
)
def validate_social_source(self, source_id, actor="admin"):
    from app.use_cases.social_signals.validate_source import SocialSourceValidationDeferred
    from app.wiring.use_case_factories import get_validate_social_source_use_case
    try:
        outcome = get_validate_social_source_use_case().execute(int(source_id), actor)
    except SocialSourceValidationDeferred as exc:
        raise self.retry(exc=exc, countdown=30, max_retries=20)
    result = asdict(outcome)
    result["tested_at"] = outcome.tested_at.isoformat()
    return result


__all__ = ["refresh_social_signals", "resume_social_analysis", "validate_social_source"]
