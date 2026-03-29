"""No-data contract tests for groups endpoints."""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

from app.main import app
from app.wiring.bootstrap import get_ui_snapshot_service


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_get_current_rankings_returns_404_when_no_rankings_exist(monkeypatch, client):
    class _FakeGroupRankService:
        def get_current_rankings(self, db, limit=197):  # noqa: ARG002
            return []

    monkeypatch.setattr(
        "app.api.v1.groups._get_group_rank_service",
        lambda: _FakeGroupRankService(),
    )

    response = await client.get("/api/v1/groups/rankings/current")

    assert response.status_code == 404
    assert response.json()["detail"] == "No ranking data available. Run a calculation first."


@pytest.mark.asyncio
async def test_get_groups_bootstrap_returns_404_when_snapshot_not_published(client):
    class _FakeSnapshotService:
        def get_groups_bootstrap(self):
            return None

    app.dependency_overrides[get_ui_snapshot_service] = lambda: _FakeSnapshotService()
    try:
        response = await client.get("/api/v1/groups/bootstrap")
    finally:
        app.dependency_overrides.pop(get_ui_snapshot_service, None)

    assert response.status_code == 404
    assert response.json()["detail"] == "No published groups bootstrap snapshot is available"
