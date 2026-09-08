from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
from app.domain.social_signals.records import (
    SocialReadRequest,
    SocialRunResult,
    SocialSourceBatch,
    SocialSourceOutcome,
    SourceTestOutcome,
)
import pytest


NOW = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)


def test_all_social_tasks_route_only_to_dedicated_queue_and_schedule_is_six_hourly():
    from app.celery_app import _social_refresh_hour_expression, celery_app
    names = (
        "refresh_social_signals", "resume_social_analysis", "validate_social_source",
    )
    for name in names:
        task = f"app.interfaces.tasks.social_signal_tasks.{name}"
        assert celery_app.conf.task_routes[task] == {"queue": "social_ingestion"}
    entry = celery_app.conf.beat_schedule["social-signal-refresh-six-hourly"]
    assert entry["task"].endswith("refresh_social_signals")
    assert entry["options"] == {"queue": "social_ingestion"}
    assert str(entry["schedule"]) == "<crontab: 17 0,6,12,18 * * * (m/h/dM/MY/d)>"
    assert _social_refresh_hour_expression(3) == "0,3,6,9,12,15,18,21"


def test_scheduled_run_identity_uses_configured_refresh_cadence(monkeypatch):
    from app.config import settings
    from app.wiring.use_case_factories import _social_run_id

    monkeypatch.setattr(settings, "social_refresh_hours", 3)
    midnight = datetime(2026, 9, 7, 0, 17, tzinfo=timezone.utc)
    five_thirty = datetime(2026, 9, 7, 2, 30, tzinfo=timezone.utc)

    assert _social_run_id("scheduled", midnight) != _social_run_id(
        "scheduled", five_thirty
    )


def test_refresh_delivery_uses_runtime_use_case_and_schedules_only_analysis_pause(monkeypatch):
    from app.interfaces.tasks import social_signal_tasks as tasks
    from app.wiring import use_case_factories
    seen = []

    class UseCase:
        async def execute(self, origin, now):
            seen.append((origin, now))
            return SocialRunResult("run-1", "live", "deferred", False, (), ("analysis_incomplete",))

    monkeypatch.setattr(use_case_factories, "get_refresh_social_signals_use_case", lambda: UseCase())
    monkeypatch.setattr(tasks, "_schedule_resume", lambda run_id, now: seen.append(("resume", run_id, now)))
    result = tasks.refresh_social_signals.run("scheduled", NOW.isoformat())
    assert result["reason_codes"] == ["analysis_incomplete"]
    assert seen == [("scheduled", NOW), ("resume", "run-1", NOW)]


def test_off_delivery_is_a_noop_and_does_not_schedule_resume(monkeypatch):
    from app.interfaces.tasks import social_signal_tasks as tasks
    from app.wiring import use_case_factories

    class UseCase:
        async def execute(self, origin, now):
            return SocialRunResult("", "off", "skipped", False, (), ("social_runtime_unavailable",))

    monkeypatch.setattr(use_case_factories, "get_refresh_social_signals_use_case", lambda: UseCase())
    monkeypatch.setattr(tasks, "_schedule_resume", lambda *_: (_ for _ in ()).throw(AssertionError("scheduled")))
    assert tasks.refresh_social_signals.run()["processing_status"] == "skipped"


def test_budget_resume_ignores_runs_with_failed_source_collection(
    db_session, monkeypatch
):
    from sqlalchemy.orm import sessionmaker

    from app import database
    from app.infra.db.models.social_signals import (
        SocialSignalRun,
        SocialSourceRegistry,
    )
    from app.interfaces.tasks import social_signal_tasks as tasks

    registry = db_session.get(SocialSourceRegistry, 1)
    if registry is None:
        registry = SocialSourceRegistry(id=1, version=1)
        db_session.add(registry)
    elif registry.version is None:
        registry.version = 1
    registry.mode, registry.provider = "live", "official"
    db_session.add(SocialSignalRun(
        id="failed-collection", registry_id=1, registry_version=registry.version,
        mode="live", provider="official", status="running",
        source_outcomes_json={
            "1": {"read_status": "success"},
            "2": {"read_status": "failed"},
        },
        application_progress_json={
            "sources": {"1": {}, "2": {}},
            "observations": {"1": {}, "2": {}},
        },
        feature_run_ids_json={}, exposure_dates_json={}, coverage_json={},
    ))
    db_session.commit()
    monkeypatch.setattr(
        database,
        "SessionLocal",
        sessionmaker(db_session.get_bind(), expire_on_commit=False),
    )

    assert tasks._latest_resumable_run() is None


def test_source_task_returns_redacted_typed_outcome_without_outer_provider_retry(monkeypatch):
    from app.interfaces.tasks import social_signal_tasks as tasks
    from app.wiring import use_case_factories

    class UseCase:
        def execute(self, source_id, actor):
            return SourceTestOutcome("official", "provider_error", 0, NOW, "invalid_provider_schema")

    monkeypatch.setattr(use_case_factories, "get_validate_social_source_use_case", lambda: UseCase())
    result = tasks.validate_social_source.run(9, "admin")
    assert result == {
        "provider": "official", "status": "provider_error", "sample_count": 0,
        "tested_at": NOW.isoformat(), "reason_code": "invalid_provider_schema",
    }


class Redis:
    def __init__(self):
        self.values = {}
        self.ttls = {}

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.values:
            return False
        self.values[key], self.ttls[key] = value, ex
        return True

    def ttl(self, key):
        return self.ttls.get(key, -2)

    def get(self, key):
        return self.values.get(key)

    def register_script(self, script):
        def release(*, keys, args):
            if self.values.get(keys[0]) == args[0]:
                del self.values[keys[0]]
                return 1
            return 0
        return release


def test_redis_lease_owner_cannot_be_released_by_another_worker():
    from app.services.social_signal_runtime_gate import RedisSocialSignalGate
    gate = RedisSocialSignalGate(Redis())
    assert gate.acquire("owner-a", 30)
    assert not gate.acquire("owner-b", 30)
    gate.release("owner-b")
    assert not gate.acquire("owner-b", 30)
    gate.release("owner-a")
    assert gate.acquire("owner-b", 30)


def test_llm_request_gate_serializes_callers_and_retains_start_spacing():
    from app.services.social_signal_runtime_gate import RedisSocialLLMRequestGate

    redis = Redis()
    first = RedisSocialLLMRequestGate(redis)
    second = RedisSocialLLMRequestGate(redis)

    assert first.acquire("owner-a", 600)
    assert not second.acquire("owner-b", 600)
    first.mark_started(5)
    first.release("owner-a")

    assert second.acquire("owner-b", 600)
    assert second.wait_seconds() == 5


def test_manual_cooldown_is_shared_but_separate_from_provider_lease():
    from app.services.social_signal_runtime_gate import RedisSocialSignalGate
    gate = RedisSocialSignalGate(Redis())
    assert gate.acquire_manual_cooldown("manual-1", 3600) == (True, 3600)
    assert gate.acquire_manual_cooldown("manual-2", 3600) == (False, 3600)
    assert gate.acquire("scheduled", 60) is True


def test_provider_cooldown_survives_provider_reconstruction_without_another_read():
    from app.services.social_signal_runtime_gate import (
        RedisSocialSignalGate,
        SharedCooldownSocialProvider,
    )

    request = SocialReadRequest(
        "request-1", "9", "123", "initial", NOW, 5, NOW - timedelta(days=14),
    )

    class RateLimitedProvider:
        def __init__(self):
            self.calls = 0

        def read_source(self, value):
            self.calls += 1
            return SocialSourceBatch(value, (), SocialSourceOutcome(
                "failed", "failed", "limited", ("rate_limited",), (),
                None, None, 0, None, "rate_limited",
                rate_limit_reset_at=NOW + timedelta(minutes=10),
            ))

    redis = Redis()
    first = RateLimitedProvider()
    initial = SharedCooldownSocialProvider(
        "official", first, RedisSocialSignalGate(redis), clock=lambda: NOW,
    ).read_source(request)
    second = RateLimitedProvider()
    replay = SharedCooldownSocialProvider(
        "official", second, RedisSocialSignalGate(redis), clock=lambda: NOW,
    ).read_source(request)

    assert initial.outcome.error_code == "rate_limited" and first.calls == 1
    assert replay.outcome.error_code == "rate_limited" and second.calls == 0
    assert replay.outcome.rate_limit_reset_at == NOW + timedelta(minutes=10)


def test_task_registry_rejects_manual_refresh_inside_shared_cooldown():
    from app.services.task_registry_service import TaskCooldownError, TaskRegistryService

    class Gate:
        def acquire_manual_cooldown(self, owner, ttl):
            return False, 47

    service = TaskRegistryService(social_gate=Gate())
    with pytest.raises(TaskCooldownError) as error:
        service.trigger_task("social-signal-refresh", object())
    assert error.value.retry_after == 47


def test_task_registry_releases_manual_cooldown_when_dispatch_fails():
    from app.services.social_signal_runtime_gate import RedisSocialSignalGate
    from app.services.task_registry_service import TaskRegistryService

    class BrokenTask:
        def apply_async(self, **_kwargs):
            raise RuntimeError("broker unavailable")

    gate = RedisSocialSignalGate(Redis())
    service = TaskRegistryService(social_gate=gate)
    service._task_imports["social-signal-refresh"] = BrokenTask()

    with pytest.raises(RuntimeError, match="broker unavailable"):
        service.trigger_task("social-signal-refresh", object())

    assert gate.acquire_manual_cooldown("next-attempt", 3600) == (True, 3600)


def test_scheduled_run_ids_collapse_duplicate_deliveries_within_one_cadence_slot():
    from app.wiring.use_case_factories import _social_run_id

    first = _social_run_id("scheduled", datetime(2026, 9, 7, 12, 18, tzinfo=timezone.utc))
    duplicate = _social_run_id("scheduled", datetime(2026, 9, 7, 13, 5, tzinfo=timezone.utc))
    next_slot = _social_run_id("scheduled", datetime(2026, 9, 7, 18, 18, tzinfo=timezone.utc))

    assert first == duplicate
    assert first != next_slot


def test_task_registry_projects_db_runtime_and_dispatches_manual_refresh_to_social_queue(db_session):
    from app.services.social_source_admin_service import SocialSourceAdminService
    from app.services.task_registry_service import TaskRegistryService

    admin = SocialSourceAdminService(db_session)
    admin.ensure_seed_sources()
    service = TaskRegistryService(social_gate=SimpleNamespace(
        acquire_manual_cooldown=lambda owner, ttl: (True, ttl)
    ))
    task = next(item for item in service.get_all_scheduled_tasks(db_session)
                if item["name"] == "social-signal-refresh")
    assert task["is_enabled"] is False

    db_session.rollback()
    runtime = admin.read_runtime()
    admin.apply_runtime("validation", "official", runtime.version, "admin")
    task = next(item for item in service.get_all_scheduled_tasks(db_session)
                if item["name"] == "social-signal-refresh")
    assert task["is_enabled"] is True

    calls = []
    service._task_imports["social-signal-refresh"] = SimpleNamespace(
        apply_async=lambda **kwargs: calls.append(kwargs) or SimpleNamespace(id="social-task-1")
    )
    result = service.trigger_task("social-signal-refresh", db_session)
    assert result["task_id"] == "social-task-1"
    assert calls == [{"kwargs": {"origin": "manual"}, "headers": None, "queue": "social_ingestion"}]


@pytest.mark.asyncio
async def test_manual_refresh_endpoint_returns_retry_after_during_cooldown(monkeypatch):
    from app.api.v1 import config, tasks as api
    from app.database import get_db
    from app.main import app
    from app.services import server_auth
    from app.services.task_registry_service import TaskCooldownError

    class Service:
        def trigger_task(self, task_name, db):
            raise TaskCooldownError(47)

    monkeypatch.setattr(api, "get_task_registry_service", lambda: Service())
    monkeypatch.setattr(config.settings, "admin_api_key", "admin-secret")
    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    app.dependency_overrides[get_db] = lambda: object()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/tasks/social-signal-refresh/run",
                headers={"X-Admin-Key": "admin-secret"},
            )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 429
    assert response.headers["retry-after"] == "47"
    assert response.json()["detail"] == "manual_refresh_cooldown"


@pytest.mark.asyncio
async def test_generic_social_refresh_endpoint_requires_admin_key(monkeypatch):
    from app.api.v1 import config, tasks as api
    from app.database import get_db
    from app.main import app
    from app.services import server_auth

    class Service:
        called = False

        def trigger_task(self, task_name, db):
            self.called = True
            return {
                "task_id": "unexpected", "task_name": task_name,
                "status": "queued", "execution_id": 1, "message": "queued",
            }

    service = Service()
    monkeypatch.setattr(api, "get_task_registry_service", lambda: service)
    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    monkeypatch.setattr(config.settings, "admin_api_key", "admin-secret")
    app.dependency_overrides[get_db] = lambda: object()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/api/v1/tasks/social-signal-refresh/run")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 401
    assert service.called is False
