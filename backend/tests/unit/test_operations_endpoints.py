"""Unit tests for Operations job console endpoints."""

from __future__ import annotations

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
