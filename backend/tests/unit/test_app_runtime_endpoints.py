"""Unit tests for runtime capability endpoints."""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

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


class _FakeSetupService:
    def __init__(self, state: dict):
        self._state = state

    def get_options(self) -> list[dict]:
        return [
            {
                "id": "quick_start",
                "label": "Quick Start",
                "description": "Install starter data now.",
                "recommended": True,
            },
            {
                "id": "download_latest",
                "label": "Download Latest Before Opening",
                "description": "Wait for a core refresh.",
                "recommended": False,
            },
        ]

    def get_status(self) -> dict:
        return self._state

    def get_legacy_bootstrap_status(self) -> dict:
        return {
            key: self._state[key]
            for key in [
                "status",
                "job_id",
                "message",
                "current_step",
                "started_at",
                "completed_at",
                "current",
                "total",
                "percent",
                "steps",
                "warnings",
                "error",
            ]
        }

    def start_setup(self, *, mode: str = "quick_start", force: bool = False) -> dict:
        state = dict(self._state)
        state["mode"] = mode
        state["status"] = "running" if mode == "download_latest" else "completed"
        state["message"] = f"setup:{mode}:{force}"
        return state


class _FakeUpdateService:
    def __init__(self, state: dict):
        self._state = state

    def get_status(self) -> dict:
        return self._state

    def start_update(self, *, scope: str = "manual", triggered_by: str = "manual", force: bool = False) -> dict:
        state = dict(self._state)
        state["scope"] = scope
        state["triggered_by"] = triggered_by
        state["message"] = f"update:{scope}:{force}"
        return state


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_app_capabilities_includes_scan_defaults(client, monkeypatch):
    from app.api.v1 import app_runtime as module

    setup_state = {
        "status": "completed",
        "mode": "quick_start",
        "job_id": None,
        "message": "Starter data installed",
        "current_step": None,
        "started_at": None,
        "completed_at": "2025-01-10T12:00:00Z",
        "current": 2,
        "total": 2,
        "percent": 100.0,
        "steps": [],
        "warnings": [],
        "error": None,
        "starter_baseline_active": True,
        "app_ready": True,
        "data_status": {
            "local_data_present": True,
            "starter_baseline_active": True,
            "setup_completed_at": "2025-01-10T12:00:00Z",
            "prices": {"ready": True, "last_success_at": "2025-01-10", "message": None},
            "breadth": {"ready": True, "last_success_at": "2025-01-10", "message": None},
            "groups": {"ready": True, "last_success_at": "2025-01-10", "message": None},
            "fundamentals": {"ready": False, "last_success_at": None, "message": "Not ready"},
            "universe": {"ready": True, "last_success_at": "2025-01-10T12:00:00Z", "message": None},
        },
    }
    update_state = {
        "status": "idle",
        "scope": None,
        "triggered_by": None,
        "job_id": None,
        "message": "Automatic updates are idle",
        "current_step": None,
        "started_at": None,
        "completed_at": None,
        "last_success_at": None,
        "current": 0,
        "total": 0,
        "percent": 0.0,
        "steps": [],
        "warnings": [],
        "error": None,
        "data_status": setup_state["data_status"],
    }

    monkeypatch.setattr(module.settings, "desktop_mode", True)
    monkeypatch.setattr(
        type(module.settings),
        "capability_flags",
        lambda _self: {"themes": True, "chatbot": True, "tasks": True},
    )
    monkeypatch.setattr(module, "_get_setup_state", lambda: setup_state)
    monkeypatch.setattr(module, "_get_update_state", lambda: update_state)
    monkeypatch.setattr(module, "_get_bootstrap_state", lambda: _FakeSetupService(setup_state).get_legacy_bootstrap_status())
    monkeypatch.setattr(module, "_get_setup_options", lambda: _FakeSetupService(setup_state).get_options())
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_ui_snapshot_service",
        lambda: _FakeUISnapshotService(),
    )

    response = await client.get("/api/v1/app-capabilities")

    assert response.status_code == 200
    data = response.json()
    assert data["scan_defaults"] == get_default_scan_profile()
    assert data["ui_snapshots"] == _FakeUISnapshotService().ui_snapshot_flags()
    assert data["bootstrap_required"] is False
    assert data["setup_required"] is False
    assert data["setup"]["mode"] == "quick_start"
    assert data["setup_options"][0]["id"] == "quick_start"
    assert data["data_status"]["prices"]["ready"] is True
    assert data["update"]["status"] == "idle"


@pytest.mark.asyncio
async def test_desktop_setup_and_update_endpoints_delegate_to_services(client, monkeypatch):
    from app.api.v1 import app_runtime as module

    setup_state = {
        "status": "idle",
        "mode": None,
        "job_id": None,
        "message": "Desktop setup has not started",
        "current_step": None,
        "started_at": None,
        "completed_at": None,
        "current": 0,
        "total": 0,
        "percent": 0.0,
        "steps": [],
        "warnings": [],
        "error": None,
        "starter_baseline_active": False,
        "app_ready": False,
        "data_status": {
            "local_data_present": False,
            "starter_baseline_active": False,
            "setup_completed_at": None,
            "prices": {"ready": False, "last_success_at": None, "message": None},
            "breadth": {"ready": False, "last_success_at": None, "message": None},
            "groups": {"ready": False, "last_success_at": None, "message": None},
            "fundamentals": {"ready": False, "last_success_at": None, "message": None},
            "universe": {"ready": False, "last_success_at": None, "message": None},
        },
    }
    update_state = {
        "status": "idle",
        "scope": None,
        "triggered_by": None,
        "job_id": None,
        "message": "Automatic updates are idle",
        "current_step": None,
        "started_at": None,
        "completed_at": None,
        "last_success_at": None,
        "current": 0,
        "total": 0,
        "percent": 0.0,
        "steps": [],
        "warnings": [],
        "error": None,
        "data_status": setup_state["data_status"],
    }

    monkeypatch.setattr(module.settings, "desktop_mode", True)
    monkeypatch.setattr("app.wiring.bootstrap.get_desktop_setup_service", lambda: _FakeSetupService(setup_state))
    monkeypatch.setattr("app.wiring.bootstrap.get_desktop_update_service", lambda: _FakeUpdateService(update_state))

    setup_response = await client.post("/api/v1/app/setup", params={"mode": "download_latest", "force": "true"})
    update_response = await client.post("/api/v1/app/update/run-now", params={"scope": "weekly", "force": "true"})
    bootstrap_response = await client.post("/api/v1/app/bootstrap", params={"force": "false"})

    assert setup_response.status_code == 200
    assert setup_response.json()["message"] == "setup:download_latest:True"
    assert update_response.status_code == 200
    assert update_response.json()["message"] == "update:weekly:True"
    assert bootstrap_response.status_code == 200
    assert bootstrap_response.json()["message"] == "setup:quick_start:False"
