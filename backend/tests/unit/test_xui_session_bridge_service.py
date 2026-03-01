"""Unit tests for XUI browser-session bridge service."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.config import settings
from app.services.xui_session_bridge_service import (
    XUISessionBridgeError,
    XUISessionBridgeService,
)


def _set_bridge_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "xui_bridge_enabled", True)
    monkeypatch.setattr(
        settings,
        "xui_bridge_allowed_origins",
        "http://localhost:80,http://127.0.0.1:80,http://localhost:5173,http://127.0.0.1:5173",
    )
    monkeypatch.setattr(settings, "xui_bridge_challenge_ttl_seconds", 120)
    monkeypatch.setattr(settings, "xui_bridge_max_cookies", 300)
    monkeypatch.setattr(settings, "xui_profile", "default")
    monkeypatch.setattr(settings, "xui_config_path", "/tmp/xui/config.toml")
    monkeypatch.setattr(
        "app.services.xui_session_bridge_service.get_redis_client",
        lambda: None,
    )


def test_create_and_consume_challenge_single_use(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_bridge_defaults(monkeypatch)
    service = XUISessionBridgeService()

    challenge = service.create_import_challenge(
        origin="http://localhost:5173",
        client_key="127.0.0.1",
    )
    assert challenge.challenge_id
    assert challenge.challenge_token

    service.consume_import_challenge(
        challenge_id=challenge.challenge_id,
        challenge_token=challenge.challenge_token,
        origin="http://localhost:5173",
    )

    with pytest.raises(XUISessionBridgeError, match="invalid or expired"):
        service.consume_import_challenge(
            challenge_id=challenge.challenge_id,
            challenge_token=challenge.challenge_token,
            origin="http://localhost:5173",
        )


def test_consume_challenge_rejects_wrong_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_bridge_defaults(monkeypatch)
    service = XUISessionBridgeService()
    challenge = service.create_import_challenge(
        origin="http://localhost:5173",
        client_key="127.0.0.1",
    )

    with pytest.raises(XUISessionBridgeError) as exc_info:
        service.consume_import_challenge(
            challenge_id=challenge.challenge_id,
            challenge_token=challenge.challenge_token,
            origin="http://127.0.0.1:5173",
        )
    assert exc_info.value.status_code == 403


def test_create_challenge_rejects_origin_port_not_in_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_bridge_defaults(monkeypatch)
    monkeypatch.setattr(settings, "xui_bridge_allowed_origins", "http://localhost:80")
    service = XUISessionBridgeService()

    with pytest.raises(XUISessionBridgeError) as exc_info:
        service.create_import_challenge(
            origin="http://localhost:5173",
            client_key="127.0.0.1",
        )
    assert exc_info.value.status_code == 403


def test_import_rejects_non_list_cookie_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_bridge_defaults(monkeypatch)
    service = XUISessionBridgeService()
    challenge = service.create_import_challenge(
        origin="http://localhost:5173",
        client_key="127.0.0.1",
    )

    with pytest.raises(XUISessionBridgeError) as exc_info:
        service.import_browser_cookies(
            challenge_id=challenge.challenge_id,
            challenge_token=challenge.challenge_token,
            cookies="not-a-list",  # type: ignore[arg-type]
            origin="http://localhost:5173",
            client_key="127.0.0.1",
        )
    assert exc_info.value.status_code == 400


def test_import_rejects_missing_required_auth_cookies(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_bridge_defaults(monkeypatch)
    service = XUISessionBridgeService()
    challenge = service.create_import_challenge(
        origin="http://localhost:5173",
        client_key="127.0.0.1",
    )

    with pytest.raises(XUISessionBridgeError) as exc_info:
        service.import_browser_cookies(
            challenge_id=challenge.challenge_id,
            challenge_token=challenge.challenge_token,
            cookies=[
                {"name": "lang", "value": "en", "domain": "x.com", "path": "/"},
            ],
            origin="http://localhost:5173",
            client_key="127.0.0.1",
        )
    assert exc_info.value.status_code == 422
    assert "auth_token" in exc_info.value.detail


def test_import_saves_storage_state_and_returns_status(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_bridge_defaults(monkeypatch)
    saved_payloads: list[dict] = []

    def fake_save_storage_state(**kwargs):
        saved_payloads.append(kwargs["storage_state"])
        return "/tmp/xui/config/profiles/default/session/storage_state.json"

    bindings = SimpleNamespace(
        save_storage_state=fake_save_storage_state,
        probe_auth_status=lambda **_kwargs: SimpleNamespace(
            authenticated=True,
            status_code="authenticated",
            message="ok",
            profile="default",
            storage_state_path="/tmp/xui/config/profiles/default/session/storage_state.json",
        ),
    )
    monkeypatch.setattr(
        "app.services.xui_session_bridge_service._load_xui_auth_bindings",
        lambda: bindings,
    )
    service = XUISessionBridgeService()
    challenge = service.create_import_challenge(
        origin="http://localhost:5173",
        client_key="127.0.0.1",
    )

    result = service.import_browser_cookies(
        challenge_id=challenge.challenge_id,
        challenge_token=challenge.challenge_token,
        cookies=[
            {"name": "auth_token", "value": "abc", "domain": ".x.com", "path": "/"},
            {"name": "ct0", "value": "xyz", "domain": "x.com", "path": "/"},
        ],
        origin="http://localhost:5173",
        client_key="127.0.0.1",
        browser="Chrome",
        extension_version="0.1.0",
    )

    assert result.authenticated is True
    assert result.status_code == "authenticated"
    assert saved_payloads
    assert saved_payloads[0]["cookies"][0]["domain"] == "x.com"


def test_import_surfaces_blocked_status_after_cookie_save(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_bridge_defaults(monkeypatch)

    bindings = SimpleNamespace(
        save_storage_state=lambda **_kwargs: "/tmp/path/storage_state.json",
        probe_auth_status=lambda **_kwargs: SimpleNamespace(
            authenticated=False,
            status_code="blocked_challenge",
            message="Challenge required",
            profile="default",
            storage_state_path="/tmp/path/storage_state.json",
        ),
    )
    monkeypatch.setattr(
        "app.services.xui_session_bridge_service._load_xui_auth_bindings",
        lambda: bindings,
    )
    service = XUISessionBridgeService()
    challenge = service.create_import_challenge(
        origin="http://localhost:5173",
        client_key="127.0.0.1",
    )

    result = service.import_browser_cookies(
        challenge_id=challenge.challenge_id,
        challenge_token=challenge.challenge_token,
        cookies=[
            {"name": "auth_token", "value": "abc", "domain": "twitter.com", "path": "/"},
            {"name": "ct0", "value": "xyz", "domain": "x.com", "path": "/"},
        ],
        origin="http://localhost:5173",
        client_key="127.0.0.1",
    )

    assert result.authenticated is False
    assert result.status_code == "blocked_challenge"


@pytest.mark.asyncio
async def test_get_auth_status_offloads_probe_when_event_loop_is_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_bridge_defaults(monkeypatch)

    def fake_probe(**_kwargs):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise AssertionError("probe_auth_status must not run on an active event loop thread")
        return SimpleNamespace(
            authenticated=True,
            status_code="authenticated",
            message="ok",
            profile="default",
            storage_state_path="/tmp/xui/config/profiles/default/session/storage_state.json",
        )

    bindings = SimpleNamespace(
        save_storage_state=lambda **_kwargs: "/tmp/path/storage_state.json",
        probe_auth_status=fake_probe,
    )
    monkeypatch.setattr(
        "app.services.xui_session_bridge_service._load_xui_auth_bindings",
        lambda: bindings,
    )

    service = XUISessionBridgeService()
    result = service.get_auth_status()
    assert result.authenticated is True
    assert result.status_code == "authenticated"
