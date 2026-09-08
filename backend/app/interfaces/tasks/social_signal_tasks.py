"""Dedicated Celery entry points for the private Social Signal pipeline."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone

from sqlalchemy import select

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


def _latest_resumable_run():
    from app.database import SessionLocal
    from app.infra.db.models.social_signals import SocialSignalRun, SocialSourceRegistry
    with SessionLocal() as db:
        registry = db.get(SocialSourceRegistry, 1)
        if registry is None or registry.mode == "off" or registry.provider == "disabled":
            return None
        runs = db.scalars(select(SocialSignalRun).where(
            SocialSignalRun.status == "running"
        ).order_by(SocialSignalRun.created_at.desc(), SocialSignalRun.id.desc())).all()
        return next((run.id for run in runs if (
            set(run.application_progress_json.get("observations", {}))
            == set(run.application_progress_json.get("sources", {}))
            and all(
                run.source_outcomes_json.get(source_id, {}).get("read_status")
                == "success"
                for source_id in run.application_progress_json.get("sources", {})
            )
        )), None)


@celery_app.task(
    bind=True,
    name="app.interfaces.tasks.social_signal_tasks.resume_social_analysis",
)
def resume_social_analysis(self, saved_run_id=None, scheduled_for=None):
    from app.wiring.use_case_factories import get_refresh_social_signals_use_case
    saved_run_id = saved_run_id or _latest_resumable_run()
    if saved_run_id is None:
        return {"status": "skipped", "reason_codes": ["no_resumable_generation"]}
    now = _now(scheduled_for)
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
