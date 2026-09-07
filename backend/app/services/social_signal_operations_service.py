"""Redacted operational health projection for Social Signals."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from app.config import settings
from app.infra.db.models.social_analysis import SocialExtractionWork, SocialLLMBudgetDay
from app.infra.db.models.social_signals import (
    SocialSignalRun, SocialSourceConfiguration, SocialSourceRegistry,
)
from app.services.social_signal_runtime_gate import (
    MANUAL_COOLDOWN_KEY,
    PROVIDER_COOLDOWN_KEY,
    PROVIDER_LEASE_KEY,
)


_PUBLIC_REASON_CODES = {
    "analysis_incomplete",
    "bounded_provider_read",
    "daily_budget_exhausted",
    "invalid_provider_json",
    "invalid_provider_schema",
    "provider_error",
    "provider_lease_unavailable",
    "provider_network_error",
    "provider_unavailable",
    "rate_limited",
    "reauthentication_required",
    "social_runtime_changed",
    "source_participation_failed",
}


def _utc(value):
    return value.replace(tzinfo=timezone.utc) if value is not None and value.tzinfo is None else value


class SocialSignalOperationsService:
    def __init__(self, *, redis_client=None, clock=None):
        self.redis = redis_client
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def snapshot(self, db):
        now = self.clock()
        if self.redis is None:
            try:
                from app.services.redis_pool import get_redis_client
                self.redis = get_redis_client()
            except Exception:
                self.redis = False
        registry = db.get(SocialSourceRegistry, 1)
        sources = db.scalars(select(SocialSourceConfiguration).order_by(
            SocialSourceConfiguration.content_source_id
        )).all()
        run = db.scalar(select(SocialSignalRun).order_by(
            SocialSignalRun.created_at.desc(), SocialSignalRun.id.desc()
        ).limit(1))
        work = db.scalars(select(SocialExtractionWork)).all()
        waiting_states = {"pending", "waiting_budget", "failed_retryable", "running"}
        waiting = [row for row in work if row.state in waiting_states]
        oldest = min((_utc(row.created_at) for row in waiting), default=None)
        budget = db.scalar(select(SocialLLMBudgetDay).where(
            SocialLLMBudgetDay.period_start_utc <= now,
            SocialLLMBudgetDay.period_end_utc > now,
        ).order_by(SocialLLMBudgetDay.period_start_utc.desc()).limit(1))
        observations = (run.application_progress_json.get("observations", {}) if run else {})
        collected = [datetime.fromisoformat(value["observed_at"])
                     for value in observations.values() if value.get("observed_at")]
        last_collection = max(collected, default=None)
        source_outcomes = run.source_outcomes_json if run else {}
        prepared = run.application_progress_json.get("prepared", {}) if run else {}
        context = prepared.get("context") or {}

        def ttl(key):
            if not self.redis:
                return None
            value = self.redis.ttl(key)
            return value if isinstance(value, int) and value > 0 else None

        spent = budget.actual_usd if budget else 0
        reserved = budget.reserved_usd if budget else 0
        limit = budget.limit_usd if budget else 0
        return {
            "generated_at": now.isoformat(),
            "mode": registry.mode if registry else "off",
            "provider": registry.provider if registry else "disabled",
            "registry_version": registry.version if registry else 0,
            "run_id": run.id if run else None,
            "run_status": run.status if run else None,
            "collection_status": "complete" if run and set(observations) == set(run.application_progress_json.get("sources", {})) else "incomplete",
            "processing_status": "complete" if run and run.status in {"staged", "completed", "published"} else "pending",
            "history_by_source": {key: value.get("history_status", "unknown") for key, value in source_outcomes.items()},
            "source_count": len(sources),
            "enabled_source_count": sum(row.lifecycle_state == "enabled" for row in sources),
            "last_collection_at": last_collection.isoformat() if last_collection else None,
            "social_fresh": bool(last_collection and (now - last_collection).total_seconds() <= settings.social_stale_after_hours * 3600),
            "formula_version": context.get("formula_version"),
            "extraction_versions": context.get("extraction_versions", []),
            "model_labels": sorted({row.actual_model or row.selected_model for row in work}),
            "budget": {
                "spent_usd": str(spent), "reserved_usd": str(reserved),
                "remaining_usd": str(max(0, limit - spent - reserved)),
                "next_reset_at": _utc(budget.period_end_utc).isoformat() if budget else None,
            },
            "backlog": {
                "waiting": len(waiting),
                "failed": sum(row.state == "failed_terminal" for row in work),
                "outside_window": sum(row.state == "outside_window" for row in work),
                "skipped": sum(row.state == "outside_window" for row in work),
                "oldest_age_seconds": (now - oldest).total_seconds() if oldest else None,
            },
            "provider_lease_ttl_seconds": ttl(PROVIDER_LEASE_KEY),
            "manual_cooldown_ttl_seconds": ttl(MANUAL_COOLDOWN_KEY),
            "provider_cooldown_ttl_seconds": ttl(
                PROVIDER_COOLDOWN_KEY.format(
                    provider=registry.provider if registry else "disabled"
                )
            ),
            "reason_codes": sorted({
                reason for value in source_outcomes.values()
                for reason in value.get("coverage_reason_codes", [])
                if reason in _PUBLIC_REASON_CODES
            }),
        }


__all__ = ["SocialSignalOperationsService"]
