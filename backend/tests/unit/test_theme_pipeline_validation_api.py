"""Validation tests for pipeline-scoped theme endpoints."""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

from app.main import app
from app.services import server_auth


@pytest_asyncio.fixture
async def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_extract_rejects_invalid_pipeline(client):
    response = await client.post("/api/v1/themes/extract", params={"pipeline": "invalid"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_calculate_metrics_rejects_invalid_pipeline(client):
    response = await client.post("/api/v1/themes/calculate-metrics", params={"pipeline": "invalid"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_run_pipeline_async_rejects_invalid_pipeline(client):
    response = await client.post("/api/v1/themes/pipeline/run", params={"pipeline": "invalid"})
    assert response.status_code == 400
