from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

from app.main import app


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


@pytest.mark.asyncio
async def test_server_auth_status_reports_required_and_unauthenticated(client, monkeypatch):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "desktop_mode", False)
    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", True)
    monkeypatch.setattr(server_auth.settings, "server_auth_password", "secret-pass")
    monkeypatch.setattr(server_auth.settings, "server_auth_session_secret", "secret-signing-key")

    response = await client.get("/api/v1/auth/status")

    assert response.status_code == 200
    assert response.json() == {
        "required": True,
        "configured": True,
        "authenticated": False,
        "mode": "session_cookie",
        "message": "Authentication required.",
    }


@pytest.mark.asyncio
async def test_protected_route_requires_login_and_accepts_session_cookie(client, monkeypatch):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "desktop_mode", False)
    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", True)
    monkeypatch.setattr(server_auth.settings, "server_auth_password", "secret-pass")
    monkeypatch.setattr(server_auth.settings, "server_auth_session_secret", "secret-signing-key")

    unauthenticated = await client.get("/api/v1/chatbot/health")
    assert unauthenticated.status_code == 401

    login = await client.post("/api/v1/auth/login", json={"password": "secret-pass"})
    assert login.status_code == 200
    assert login.json()["authenticated"] is True

    authenticated = await client.get("/api/v1/chatbot/health")
    assert authenticated.status_code == 200
    assert authenticated.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_config_route_accepts_admin_key_without_server_session(client, monkeypatch):
    from app.api.v1 import config as config_api
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "desktop_mode", False)
    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", True)
    monkeypatch.setattr(server_auth.settings, "server_auth_password", "server-secret")
    monkeypatch.setattr(server_auth.settings, "server_auth_session_secret", "signing-secret")
    monkeypatch.setattr(server_auth.settings, "admin_api_key", "admin-secret")
    monkeypatch.setattr(
        config_api,
        "_theme_policy_defaults",
        lambda pipeline: {"matcher": {}, "lifecycle": {}},
    )
    monkeypatch.setattr(config_api, "_get_setting_json", lambda db, key, default: default)

    response = await client.get(
        "/api/v1/config/theme-policies",
        params={"pipeline": "technical"},
        headers={"X-Admin-Key": "admin-secret"},
    )

    assert response.status_code == 200
    assert response.json()["pipeline"] == "technical"


@pytest.mark.asyncio
async def test_protected_route_returns_503_when_auth_required_but_not_configured(client, monkeypatch):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "desktop_mode", False)
    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", True)
    monkeypatch.setattr(server_auth.settings, "server_auth_password", "")
    monkeypatch.setattr(server_auth.settings, "admin_api_key", "")
    monkeypatch.setattr(server_auth.settings, "server_auth_session_secret", "")

    response = await client.get("/api/v1/chatbot/health")

    assert response.status_code == 503
    assert "SERVER_AUTH_PASSWORD" in response.json()["detail"]


@pytest.mark.asyncio
async def test_admin_api_key_does_not_authenticate_general_server_routes(client, monkeypatch):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "desktop_mode", False)
    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", True)
    monkeypatch.setattr(server_auth.settings, "server_auth_password", "server-secret")
    monkeypatch.setattr(server_auth.settings, "server_auth_session_secret", "signing-secret")
    monkeypatch.setattr(server_auth.settings, "admin_api_key", "admin-secret")

    response = await client.get("/api/v1/chatbot/health", headers={"X-Admin-Key": "admin-secret"})

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_admin_api_key_alone_does_not_configure_server_auth(client, monkeypatch):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "desktop_mode", False)
    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", True)
    monkeypatch.setattr(server_auth.settings, "server_auth_password", "")
    monkeypatch.setattr(server_auth.settings, "server_auth_session_secret", "")
    monkeypatch.setattr(server_auth.settings, "admin_api_key", "admin-secret")

    response = await client.get("/api/v1/chatbot/health")

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_login_cookie_does_not_trust_forwarded_proto_header(client, monkeypatch):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "desktop_mode", False)
    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", True)
    monkeypatch.setattr(server_auth.settings, "server_auth_password", "secret-pass")
    monkeypatch.setattr(server_auth.settings, "server_auth_session_secret", "secret-signing-key")
    monkeypatch.setattr(server_auth.settings, "server_auth_secure_cookie", False)

    response = await client.post(
        "/api/v1/auth/login",
        json={"password": "secret-pass"},
        headers={"X-Forwarded-Proto": "https"},
    )

    assert response.status_code == 200
    assert "Secure" not in response.headers["set-cookie"]


@pytest.mark.asyncio
async def test_login_cookie_can_be_forced_secure_with_explicit_setting(client, monkeypatch):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "desktop_mode", False)
    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", True)
    monkeypatch.setattr(server_auth.settings, "server_auth_password", "secret-pass")
    monkeypatch.setattr(server_auth.settings, "server_auth_session_secret", "secret-signing-key")
    monkeypatch.setattr(server_auth.settings, "server_auth_secure_cookie", True)

    response = await client.post("/api/v1/auth/login", json={"password": "secret-pass"})

    assert response.status_code == 200
    assert "Secure" in response.headers["set-cookie"]
