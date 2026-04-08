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

    response = await client.get("/api/v1/app-capabilities")

    assert response.status_code == 200
    data = response.json()
    assert data["scan_defaults"] == get_default_scan_profile()
    assert data["ui_snapshots"] == _FakeUISnapshotService().ui_snapshot_flags()
    assert data["features"] == {"themes": True, "chatbot": True, "tasks": True}
    assert data["api_base_path"] == "/api"
