"""Redacted operational health projection for Social Signals."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json

from sqlalchemy import func, select

from app.config import settings
from app.infra.db.models.social_analysis import (
    SocialExtractionWork, SocialLLMBudgetDay, SocialThemeAssociation,
)
from app.infra.db.models.social_signals import (
    SocialSignalRun, SocialSourceConfiguration, SocialSourceRegistry,
)
from app.models.app_settings import AppSetting
from app.services.social_signal_runtime_gate import (
    MANUAL_COOLDOWN_KEY,
    PROVIDER_COOLDOWN_KEY,
    PROVIDER_LEASE_KEY,
)
from app.services.social_llm_budget_service import social_budget_period


_PUBLIC_REASON_CODES = {
    "analysis_failed",
    "analysis_incomplete",
    "bounded_provider_read",
    "daily_budget_exhausted",
    "invalid_provider_json",
    "invalid_provider_schema",
    "provider_error",
    "provider_lease_unavailable",
    "provider_network_error",
    "provider_timeout",
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
        observations = (run.application_progress_json.get("observations", {}) if run else {})
        enabled_sources = [source for source in sources if source.lifecycle_state == "enabled"]
        successful_collections = [
            _utc(source.last_successful_collection_at)
            for source in enabled_sources
            if source.last_successful_collection_at is not None
        ]
        last_collection = max(successful_collections, default=None)
        source_outcomes = run.source_outcomes_json if run else {}
        pinned_source_ids = set(
            run.application_progress_json.get("sources", {}) if run else {}
        )
        collection_complete = bool(
            run
            and pinned_source_ids
            and set(observations) == pinned_source_ids
            and set(source_outcomes) == pinned_source_ids
            and all(
                source_outcomes[source_id].get("read_status") == "success"
                for source_id in pinned_source_ids
            )
        )
        social_fresh = bool(enabled_sources) and all(
            source.last_successful_collection_at is not None
            and (
                now - _utc(source.last_successful_collection_at)
            ).total_seconds() <= settings.social_stale_after_hours * 3600
            for source in enabled_sources
        )
        prepared = run.application_progress_json.get("prepared", {}) if run else {}
        failure = run.application_progress_json.get("failure", {}) if run else {}
        context = prepared.get("context") or {}
        unknown_identity_count = db.scalar(select(func.count()).select_from(
            SocialThemeAssociation
        ).where(
            SocialThemeAssociation.company_key.is_(None),
            SocialThemeAssociation.state == "proposed",
        )) or 0
        policy_rows = db.scalars(select(AppSetting).where(AppSetting.key.in_({
            "social_llm_daily_limit_usd", "social_llm_budget_timezone",
            "social_llm_pricing", "social_llm_pricing_blocks",
        }))).all()
        policy = {row.key: row.value for row in policy_rows}
        budget_limit = policy.get(
            "social_llm_daily_limit_usd",
            format(settings.social_llm_daily_budget_usd.normalize(), "f"),
        )
        budget_timezone = policy.get(
            "social_llm_budget_timezone", settings.social_llm_budget_timezone
        )
        pricing_status, pricing_version, blocked_models = "absent", None, []
        if "social_llm_pricing" in policy:
            try:
                pricing = json.loads(policy["social_llm_pricing"])
                if (not isinstance(pricing.get("version"), str)
                        or not isinstance(pricing.get("models"), dict)):
                    raise ValueError("invalid_pricing")
                pricing_status, pricing_version = "configured", pricing["version"]
                blocks = json.loads(policy.get("social_llm_pricing_blocks", "{}"))
                blocked_models = sorted(
                    model for model, value in blocks.items()
                    if isinstance(value, dict) and (
                        value.get("version") == pricing_version
                        or (isinstance(value.get("versions"), dict)
                            and pricing_version in value["versions"])
                    )
                )
                if blocked_models:
                    pricing_status = "configured_with_blocks"
            except (TypeError, ValueError, json.JSONDecodeError):
                pricing_status, pricing_version, blocked_models = "invalid", None, []
        budget_rows = []
        try:
            _local_day, period_start, next_reset_at = social_budget_period(
                now, budget_timezone
            )
            budget_rows = db.scalars(select(SocialLLMBudgetDay).where(
                SocialLLMBudgetDay.period_start_utc < next_reset_at,
                SocialLLMBudgetDay.period_end_utc > period_start,
            )).all()
        except (TypeError, ValueError, KeyError):
            next_reset_at = None

        def ttl(key):
            if not self.redis:
                return None
            value = self.redis.ttl(key)
            return value if isinstance(value, int) and value > 0 else None

        spent = sum((row.actual_usd for row in budget_rows), Decimal(0))
        reserved = sum((row.reserved_usd for row in budget_rows), Decimal(0))
        try:
            limit = Decimal(str(budget_limit))
        except (InvalidOperation, ValueError):
            limit = Decimal(0)
        return {
            "generated_at": now.isoformat(),
            "mode": registry.mode if registry else "off",
            "provider": registry.provider if registry else "disabled",
            "registry_version": registry.version if registry else 0,
            "run_id": run.id if run else None,
            "run_status": run.status if run else None,
            "collection_status": "complete" if collection_complete else "incomplete",
            "processing_status": (
                "complete"
                if run and run.status in {"staged", "completed", "published"}
                else "failed"
                if run and run.status == "failed"
                else "pending"
            ),
            "history_by_source": {key: value.get("history_status", "unknown") for key, value in source_outcomes.items()},
            "source_count": sum(row.lifecycle_state != "archived" for row in sources),
            "archived_source_count": sum(row.lifecycle_state == "archived" for row in sources),
            "enabled_source_count": sum(row.lifecycle_state == "enabled" for row in sources),
            "participating_source_count": len(observations),
            "unknown_company_identity_count": unknown_identity_count,
            "last_collection_at": last_collection.isoformat() if last_collection else None,
            "social_fresh": social_fresh,
            "formula_version": context.get("formula_version"),
            "extraction_versions": context.get("extraction_versions", []),
            "model_labels": sorted({row.actual_model or row.selected_model for row in work}),
            "budget": {
                "limit_usd": str(budget_limit), "timezone": budget_timezone,
                "spent_usd": str(spent), "reserved_usd": str(reserved),
                "remaining_usd": str(max(0, limit - spent - reserved)),
                "next_reset_at": next_reset_at.isoformat() if next_reset_at else None,
                "pricing_status": pricing_status,
                "pricing_version": pricing_version,
                "blocked_models": blocked_models,
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
            "reason_codes": sorted(
                {
                    reason
                    for value in source_outcomes.values()
                    for reason in value.get("coverage_reason_codes", [])
                    if reason in _PUBLIC_REASON_CODES
                }
                | ({failure.get("reason_code")} & _PUBLIC_REASON_CODES)
            ),
        }


__all__ = ["SocialSignalOperationsService"]
