"""Unit tests for runtime capability endpoints."""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

from app.database import get_db
from app.domain.scanning.defaults import get_default_scan_profile
from app.main import app


class _FakeUISnapshotService:
    def ui_snapshot_flags(self) -> dict[str, bool]:
        return {
            "enabled": True,
            "scan": True,
            "breadth": False,
            "groups": True,
            "themes": False,
        }


class _FakeDb:
    pass


class _FakeBootstrapStatus:
    bootstrap_required = True
    empty_system = True
    primary_market = "US"
    enabled_markets = ["US", "HK"]
    bootstrap_state = "running"
    supported_markets = ["US", "HK", "JP", "TW"]


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_app_capabilities_includes_scan_defaults(client, monkeypatch):
    from app.api.v1 import app_runtime as module

    monkeypatch.setattr(
        type(module.settings),
        "capability_flags",
        lambda _self: {"themes": True, "chatbot": True, "tasks": True},
    )
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_ui_snapshot_service",
        lambda: _FakeUISnapshotService(),
    )
    monkeypatch.setattr(module, "get_runtime_bootstrap_status", lambda _db: _FakeBootstrapStatus())
    app.dependency_overrides[get_db] = lambda: _FakeDb()

    try:
        response = await client.get("/api/v1/app-capabilities")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    data = response.json()
    assert data["scan_defaults"] == get_default_scan_profile()
    assert data["ui_snapshots"] == _FakeUISnapshotService().ui_snapshot_flags()
    assert data["features"] == {"themes": True, "chatbot": True, "tasks": True}
    assert data["bootstrap_required"] is True
    assert data["primary_market"] == "US"
    assert data["enabled_markets"] == ["US", "HK"]
    assert data["bootstrap_state"] == "running"
    assert data["api_base_path"] == "/api"


@pytest.mark.asyncio
async def test_runtime_bootstrap_status_endpoint_returns_persisted_state(client, monkeypatch):
    from app.api.v1 import app_runtime as module

    monkeypatch.setattr(module, "get_runtime_bootstrap_status", lambda _db: _FakeBootstrapStatus())
    app.dependency_overrides[get_db] = lambda: _FakeDb()

    try:
        response = await client.get("/api/v1/runtime/bootstrap-status")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    payload = response.json()
    assert payload["bootstrap_required"] is True
    assert payload["empty_system"] is True
    assert payload["primary_market"] == "US"
    assert payload["enabled_markets"] == ["US", "HK"]
    assert payload["bootstrap_state"] == "running"


@pytest.mark.asyncio
async def test_runtime_bootstrap_start_persists_preferences_and_queues_orchestration(client, monkeypatch):
    from app.api.v1 import app_runtime as module

    saved = {}

    def _save(_db, *, primary_market, enabled_markets, bootstrap_state):
        saved["primary_market"] = primary_market
        saved["enabled_markets"] = enabled_markets
        saved["bootstrap_state"] = bootstrap_state
        return type(
            "SavedPrefs",
            (),
            {
                "primary_market": primary_market,
                "enabled_markets": enabled_markets,
                "bootstrap_state": bootstrap_state,
            },
        )()

    bootstrap_status = type(
        "BootstrapStatus",
        (),
        {
            "bootstrap_required": True,
            "empty_system": True,
            "primary_market": "HK",
            "enabled_markets": ["HK", "US"],
            "bootstrap_state": "running",
            "supported_markets": ["US", "HK", "JP", "TW"],
        },
    )()

    monkeypatch.setattr(module, "save_runtime_preferences", _save)
    monkeypatch.setattr(module, "queue_local_runtime_bootstrap", lambda **_kwargs: "task-bootstrap-123")
    monkeypatch.setattr(module, "get_runtime_bootstrap_status", lambda _db: bootstrap_status)
    app.dependency_overrides[get_db] = lambda: _FakeDb()

    try:
        response = await client.post(
            "/api/v1/runtime/bootstrap",
            json={"primary_market": "HK", "enabled_markets": ["HK", "US"]},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    assert saved == {
        "primary_market": "HK",
        "enabled_markets": ["HK", "US"],
        "bootstrap_state": "running",
    }
    payload = response.json()
    assert payload["task_id"] == "task-bootstrap-123"
    assert payload["primary_market"] == "HK"
    assert payload["enabled_markets"] == ["HK", "US"]
    assert payload["bootstrap_state"] == "running"
