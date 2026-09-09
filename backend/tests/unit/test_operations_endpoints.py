"""Unit tests for Operations job console endpoints."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest
import pytest_asyncio

from app.database import get_db
from app.main import app


class _FakeDb:
    pass


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_operations_jobs_endpoint_returns_inventory(client, monkeypatch):
    from app.api.v1 import operations as module
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    monkeypatch.setattr(
        module._service,
        "list_jobs",
        lambda _db: {
            "jobs": [
                {
                    "task_id": "task-123",
                    "task_name": "app.tasks.cache_tasks.smart_refresh_cache",
                    "queue": "data_fetch_us",
                    "market": "US",
                    "state": "waiting",
                    "worker": None,
                    "age_seconds": 12.0,
                    "wait_reason": "waiting_for_external_fetch_global",
                    "heartbeat_lag_seconds": None,
                    "cancel_strategy": "revoke_and_remove_from_queue",
                    "progress_mode": "determinate",
                    "percent": 60.0,
                    "current": 600,
                    "total": 1000,
                    "message": "Batch 3/5 · refreshing prices",
                }
            ],
            "queues": [{"queue": "data_fetch_us", "depth": 1, "oldest_age_seconds": 12.0}],
            "workers": [{"worker": "general@host", "status": "online", "queues": ["celery"], "active": 0, "reserved": 0, "scheduled": 0}],
            "leases": {"external_fetch_global": None, "market_workload": {"US": None, "HK": None, "JP": None, "TW": None}},
            "generated_at": "2026-04-18T12:00:00+00:00",
        },
    )
    app.dependency_overrides[get_db] = lambda: _FakeDb()

    try:
        response = await client.get("/api/v1/operations/jobs")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    payload = response.json()
    assert payload["jobs"][0]["wait_reason"] == "waiting_for_external_fetch_global"
    assert payload["jobs"][0]["progress_mode"] == "determinate"
    assert payload["jobs"][0]["percent"] == 60.0
    assert payload["jobs"][0]["current"] == 600
    assert payload["jobs"][0]["total"] == 1000
    assert payload["jobs"][0]["message"] == "Batch 3/5 · refreshing prices"
    assert payload["queues"][0]["depth"] == 1
    assert payload["workers"][0]["worker"] == "general@host"


@pytest.mark.asyncio
async def test_operations_cancel_endpoint_returns_service_payload(client, monkeypatch):
    from app.api.v1 import operations as module
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    monkeypatch.setattr(
        module._service,
        "cancel_job",
        lambda _db, task_id: {
            "status": "accepted",
            "cancel_strategy": "scan_cancel",
            "message": f"Cancelled {task_id}",
        },
    )
    app.dependency_overrides[get_db] = lambda: _FakeDb()

    try:
        response = await client.post("/api/v1/operations/jobs/task-123/cancel")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    assert response.json() == {
        "status": "accepted",
        "cancel_strategy": "scan_cancel",
        "message": "Cancelled task-123",
    }


@pytest.mark.asyncio
async def test_social_signal_operations_endpoint_returns_redacted_health(client, monkeypatch):
    from app.api.v1 import operations as module
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    monkeypatch.setattr(module._social_service, "snapshot", lambda _db: {
        "mode": "validation", "provider": "xui", "run_id": "run-1",
        "collection_status": "complete", "processing_status": "pending",
        "history_by_source": {"1": "warming_up", "2": "limited"},
        "budget": {"spent_usd": "1.25", "reserved_usd": "0.25", "remaining_usd": "0.50"},
        "backlog": {"waiting": 3, "failed": 0, "outside_window": 2},
        "reason_codes": ["bounded_provider_read"],
    })
    app.dependency_overrides[get_db] = lambda: _FakeDb()
    try:
        response = await client.get("/api/v1/operations/social-signals")
    finally:
        app.dependency_overrides.pop(get_db, None)
    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "xui" and payload["budget"]["remaining_usd"] == "0.50"
    assert "stderr" not in str(payload).lower() and "config_path" not in str(payload).lower()


def test_social_signal_operations_snapshot_uses_db_runtime_and_shared_ttls(db_session):
    from app.services.social_signal_operations_service import SocialSignalOperationsService
    from app.services.social_signal_runtime_gate import (
        MANUAL_COOLDOWN_KEY,
        PROVIDER_COOLDOWN_KEY,
        PROVIDER_LEASE_KEY,
    )
    from app.services.social_source_admin_service import SocialSourceAdminService

    admin = SocialSourceAdminService(db_session)
    admin.ensure_seed_sources()
    runtime = admin.read_runtime()
    admin.apply_runtime("validation", "official", runtime.version, "admin")

    class Redis:
        def ttl(self, key):
            return {
                PROVIDER_LEASE_KEY: 30,
                MANUAL_COOLDOWN_KEY: 60,
                PROVIDER_COOLDOWN_KEY.format(provider="official"): 90,
            }.get(key, -2)

    payload = SocialSignalOperationsService(
        redis_client=Redis(),
        clock=lambda: datetime(2026, 9, 7, 12, tzinfo=timezone.utc),
    ).snapshot(db_session)

    assert (payload["mode"], payload["provider"]) == ("validation", "official")
    assert (payload["source_count"], payload["enabled_source_count"]) == (2, 2)
    assert payload["archived_source_count"] == 0
    assert payload["participating_source_count"] == 0
    assert payload["unknown_company_identity_count"] == 0
    assert payload["provider_lease_ttl_seconds"] == 30
    assert payload["manual_cooldown_ttl_seconds"] == 60
    assert payload["provider_cooldown_ttl_seconds"] == 90
    assert payload["budget"]["limit_usd"] == "2"
    assert payload["budget"]["timezone"] == "Asia/Singapore"
    assert payload["budget"]["pricing_status"] == "absent"


def test_social_health_requires_every_enabled_source_to_be_fresh(db_session):
    from app.infra.db.models.social_signals import (
        SocialSignalRun,
        SocialSourceConfiguration,
    )
    from app.services.social_signal_operations_service import SocialSignalOperationsService
    from app.services.social_source_admin_service import SocialSourceAdminService

    now = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)
    admin = SocialSourceAdminService(db_session)
    admin.ensure_seed_sources()
    runtime = admin.read_runtime()
    runtime = admin.apply_runtime("validation", "official", runtime.version, "admin")
    fresh_source = db_session.get(SocialSourceConfiguration, 1)
    stale_source = db_session.get(SocialSourceConfiguration, 2)
    fresh_source.last_successful_collection_at = now - timedelta(hours=1)
    stale_source.last_successful_collection_at = now - timedelta(hours=100)
    db_session.add(SocialSignalRun(
        id="failed-latest",
        registry_id=1,
        registry_version=runtime.version,
        mode="validation",
        provider="official",
        status="running",
        source_outcomes_json={"1": {"read_status": "failed"}},
        application_progress_json={
            "sources": {"1": {}},
            "observations": {"1": {"observed_at": now.isoformat()}},
        },
        feature_run_ids_json={},
        exposure_dates_json={},
        coverage_json={},
        created_at=now,
    ))
    db_session.commit()

    payload = SocialSignalOperationsService(
        redis_client=False,
        clock=lambda: now,
    ).snapshot(db_session)

    assert payload["last_collection_at"] == (now - timedelta(hours=1)).isoformat()
    assert payload["social_fresh"] is False
    assert payload["collection_status"] == "incomplete"

    stale_source.lifecycle_state = "disabled"
    db_session.commit()

    payload = SocialSignalOperationsService(
        redis_client=False,
        clock=lambda: now,
    ).snapshot(db_session)

    assert payload["social_fresh"] is True


def test_social_health_reports_terminal_analysis_failure(db_session):
    from app.infra.db.models.social_signals import SocialSignalRun
    from app.services.social_signal_operations_service import SocialSignalOperationsService
    from app.services.social_source_admin_service import SocialSourceAdminService

    now = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)
    admin = SocialSourceAdminService(db_session)
    admin.ensure_seed_sources()
    runtime = admin.read_runtime()
    runtime = admin.apply_runtime("live", "official", runtime.version, "admin")
    db_session.add(SocialSignalRun(
        id="failed-analysis",
        registry_id=1,
        registry_version=runtime.version,
        mode="live",
        provider="official",
        status="failed",
        source_outcomes_json={
            "1": {"read_status": "success", "processing_status": "failed"},
            "2": {"read_status": "success", "processing_status": "failed"},
        },
        application_progress_json={
            "sources": {"1": {}, "2": {}},
            "observations": {"1": {}, "2": {}},
            "failure": {"reason_code": "analysis_failed", "work_ids": [1]},
        },
        feature_run_ids_json={},
        exposure_dates_json={},
        coverage_json={},
        created_at=now,
        completed_at=now,
    ))
    db_session.commit()

    payload = SocialSignalOperationsService(
        redis_client=False,
        clock=lambda: now,
    ).snapshot(db_session)

    assert payload["processing_status"] == "failed"
    assert payload["collection_status"] == "complete"
    assert payload["reason_codes"] == ["analysis_failed"]


def test_social_signal_health_reports_pricing_blocks_without_secret_configuration(db_session):
    from app.models.app_settings import AppSetting
    from app.services.social_signal_operations_service import SocialSignalOperationsService
    from app.services.social_source_admin_service import SocialSourceAdminService

    SocialSourceAdminService(db_session).ensure_seed_sources()
    db_session.add_all([
        AppSetting(key="social_llm_daily_limit_usd", value="2", category="social"),
        AppSetting(key="social_llm_budget_timezone", value="Asia/Singapore", category="social"),
        AppSetting(key="social_llm_pricing", category="social", value='{"version":"v2","models":{"small":{"provider":"openai","actual_models":["small"],"input_usd_per_million":"1","output_usd_per_million":"2"}}}'),
        AppSetting(key="social_llm_pricing_blocks", category="social", value='{"small":{"versions":{"v2":"billing_mismatch"}}}'),
    ])
    db_session.commit()

    payload = SocialSignalOperationsService(
        redis_client=False,
        clock=lambda: datetime(2026, 9, 7, 12, tzinfo=timezone.utc),
    ).snapshot(db_session)

    assert payload["budget"]["pricing_status"] == "configured_with_blocks"
    assert payload["budget"]["pricing_version"] == "v2"
    assert payload["budget"]["blocked_models"] == ["small"]
    assert "billing_mismatch" not in str(payload)
