"""Endpoint tests for /themes/twitter/session bridge APIs."""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

from app.main import app
from app.services import server_auth
from app.services.xui_session_bridge_service import XUISessionBridgeError
from app.config import settings


@pytest_asyncio.fixture
async def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_get_twitter_session_status_returns_provider_payload(monkeypatch: pytest.MonkeyPatch, client):
    monkeypatch.setattr(settings, "x_ingest_provider", "xui")
    monkeypatch.setattr(
        "app.api.v1.themes.XUISessionBridgeService.get_auth_status",
        lambda self: type(
            "Status",
            (),
            {
                "authenticated": True,
                "status_code": "authenticated",
                "message": "ok",
                "profile": "default",
                "storage_state_path": "/tmp/storage_state.json",
                "provider": "xui",
            },
        )(),
    )

    response = await client.get("/api/v1/themes/twitter/session")
    assert response.status_code == 200
    payload = response.json()
    assert payload["authenticated"] is True
    assert payload["provider"] == "xui"


@pytest.mark.asyncio
async def test_get_twitter_session_status_reports_official_api_mode(
    monkeypatch: pytest.MonkeyPatch,
    client,
):
    monkeypatch.setattr(settings, "x_ingest_provider", "official")
    monkeypatch.setattr(settings, "twitter_bearer_token", "token")

    response = await client.get("/api/v1/themes/twitter/session")

    assert response.status_code == 200
    payload = response.json()
    assert payload["authenticated"] is False
    assert payload["provider"] == "official_x_api"
    assert payload["status_code"] == "configured"


@pytest.mark.asyncio
async def test_challenge_is_unavailable_in_official_api_mode(
    monkeypatch: pytest.MonkeyPatch,
    client,
):
    monkeypatch.setattr(settings, "x_ingest_provider", "official")

    response = await client.post(
        "/api/v1/themes/twitter/session/challenge",
        headers={"Origin": "http://localhost:5173"},
    )

    assert response.status_code == 503
    assert "X_INGEST_PROVIDER=xui" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_twitter_session_status_rejects_invalid_provider(
    monkeypatch: pytest.MonkeyPatch,
    client,
):
    monkeypatch.setattr(settings, "x_ingest_provider", "selenium")

    response = await client.get("/api/v1/themes/twitter/session")

    assert response.status_code == 503
    assert "X_INGEST_PROVIDER" in response.json()["detail"]


@pytest.mark.asyncio
async def test_create_challenge_maps_bridge_error_status(monkeypatch: pytest.MonkeyPatch, client):
    monkeypatch.setattr(settings, "x_ingest_provider", "xui")

    def _raise_disallowed_origin(self, *, origin, client_key):
        raise XUISessionBridgeError(403, "Origin not allowed")

    monkeypatch.setattr(
        "app.api.v1.themes.XUISessionBridgeService.create_import_challenge",
        _raise_disallowed_origin,
    )

    response = await client.post(
        "/api/v1/themes/twitter/session/challenge",
        headers={"Origin": "https://evil.example"},
    )
    assert response.status_code == 403
    assert "Origin not allowed" in response.json()["detail"]


@pytest.mark.asyncio
async def test_import_endpoint_returns_status_payload(monkeypatch: pytest.MonkeyPatch, client):
    monkeypatch.setattr(settings, "x_ingest_provider", "xui")
    monkeypatch.setattr(
        "app.api.v1.themes.XUISessionBridgeService.import_browser_cookies",
        lambda self, **_kwargs: type(
            "Status",
            (),
            {
                "authenticated": False,
                "status_code": "blocked_challenge",
                "message": "Challenge required",
                "profile": "default",
                "storage_state_path": "/tmp/storage_state.json",
                "provider": "xui",
            },
        )(),
    )

    response = await client.post(
        "/api/v1/themes/twitter/session/import",
        headers={"Origin": "http://localhost:5173"},
        json={
            "challenge_id": "abc",
            "challenge_token": "def",
            "cookies": [
                {"name": "auth_token", "value": "1", "domain": "x.com"},
                {"name": "ct0", "value": "2", "domain": "x.com"},
            ],
            "browser": "Chrome",
            "extension_version": "0.1.0",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["authenticated"] is False
    assert payload["status_code"] == "blocked_challenge"
