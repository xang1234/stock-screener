from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

from app.main import app
from app.services import server_auth

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _disable_server_auth():
    original_enabled = server_auth.settings.server_auth_enabled
    server_auth.settings.server_auth_enabled = False
    yield
    server_auth.settings.server_auth_enabled = original_enabled
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_strategy_profiles_endpoint_lists_known_profiles(client):
    response = await client.get("/api/v1/strategy-profiles")

    assert response.status_code == 200
    payload = response.json()
    assert [profile["profile"] for profile in payload["profiles"]] == [
        "default",
        "growth",
        "momentum",
        "risk_off",
    ]
    assert payload["profiles"][1]["scan_defaults"]["criteria"]["custom_filters"]["rs_rating_min"] >= 80


@pytest.mark.asyncio
async def test_strategy_profile_detail_endpoint_falls_back_to_default(client):
    response = await client.get("/api/v1/strategy-profiles/not-a-real-profile")

    assert response.status_code == 200
    payload = response.json()
    assert payload["profile"] == "default"
    assert payload["digest"]["leader_sort"] == "composite_score"
