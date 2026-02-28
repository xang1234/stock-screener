"""Auth login storage-state capture and hardening behavior."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from typing import Any

import pytest

from xui_reader.auth import (
    AuthProbeSnapshot,
    login_and_save_storage_state,
    logout_profile,
    probe_auth_status,
    storage_state_path,
)
from xui_reader.config import init_default_config
from xui_reader.errors import AuthError
from xui_reader.profiles import create_profile


def _fake_storage_state(_config: Any, _login_url: str) -> dict[str, Any]:
    return {
        "cookies": [{"name": "sessionid", "value": "secret-cookie-value"}],
        "origins": [],
    }


def test_login_saves_storage_state_with_restrictive_permissions(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_default_config(config_path)
    create_profile("default", config_path)

    destination = login_and_save_storage_state(
        profile_name="default",
        config_path=config_path,
        capture_fn=_fake_storage_state,
    )

    assert destination == storage_state_path("default", config_path)
    assert destination.exists()
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded["cookies"][0]["value"] == "secret-cookie-value"
    mode = stat.S_IMODE(os.stat(destination).st_mode)
    assert mode == 0o600


def test_login_requires_initialized_profile(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_default_config(config_path)

    with pytest.raises(AuthError, match="profiles create"):
        login_and_save_storage_state(
            profile_name="default",
            config_path=config_path,
            capture_fn=_fake_storage_state,
        )


def test_login_rejects_empty_storage_state(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_default_config(config_path)
    create_profile("default", config_path)

    with pytest.raises(AuthError, match="storage_state is empty"):
        login_and_save_storage_state(
            profile_name="default",
            config_path=config_path,
            capture_fn=lambda _config, _url: {"cookies": [], "origins": []},
        )


def test_login_rejects_storage_state_missing_origins_array(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_default_config(config_path)
    create_profile("default", config_path)

    with pytest.raises(AuthError, match="cookies/origins arrays"):
        login_and_save_storage_state(
            profile_name="default",
            config_path=config_path,
            capture_fn=lambda _config, _url: {"cookies": []},
        )


def test_login_rejects_storage_state_missing_cookies_array(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_default_config(config_path)
    create_profile("default", config_path)

    with pytest.raises(AuthError, match="cookies/origins arrays"):
        login_and_save_storage_state(
            profile_name="default",
            config_path=config_path,
            capture_fn=lambda _config, _url: {"origins": []},
        )


def test_login_wraps_write_permission_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.toml"
    init_default_config(config_path)
    create_profile("default", config_path)

    def raise_permission_error(*_args: object, **_kwargs: object) -> object:
        raise PermissionError("denied")

    monkeypatch.setattr("xui_reader.auth.os.open", raise_permission_error)

    with pytest.raises(AuthError, match="Could not persist storage_state"):
        login_and_save_storage_state(
            profile_name="default",
            config_path=config_path,
            capture_fn=_fake_storage_state,
        )


def test_probe_auth_status_reports_authenticated_session(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_default_config(config_path)
    create_profile("default", config_path)
    login_and_save_storage_state(
        profile_name="default",
        config_path=config_path,
        capture_fn=_fake_storage_state,
    )

    result = probe_auth_status(
        profile_name="default",
        config_path=config_path,
        probe_fn=lambda _config, _path: AuthProbeSnapshot(
            current_url="https://x.com/home",
            page_title="Home / X",
            body_text="",
        ),
    )

    assert result.authenticated is True
    assert result.status_code == "authenticated"


def test_probe_auth_status_detects_login_wall(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_default_config(config_path)
    create_profile("default", config_path)
    login_and_save_storage_state(
        profile_name="default",
        config_path=config_path,
        capture_fn=_fake_storage_state,
    )

    result = probe_auth_status(
        profile_name="default",
        config_path=config_path,
        probe_fn=lambda _config, _path: AuthProbeSnapshot(
            current_url="https://x.com/i/flow/login",
            page_title="Log in / X",
            body_text="",
        ),
    )

    assert result.authenticated is False
    assert result.status_code == "blocked_login_wall"
    assert "xui auth login" in " ".join(result.next_steps)


def test_probe_auth_status_does_not_false_positive_on_home_login_copy(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_default_config(config_path)
    create_profile("default", config_path)
    login_and_save_storage_state(
        profile_name="default",
        config_path=config_path,
        capture_fn=_fake_storage_state,
    )

    result = probe_auth_status(
        profile_name="default",
        config_path=config_path,
        probe_fn=lambda _config, _path: AuthProbeSnapshot(
            current_url="https://x.com/home",
            page_title="Home / X",
            body_text="Use this button to log in to another account.",
        ),
    )

    assert result.authenticated is True
    assert result.status_code == "authenticated"


def test_probe_auth_status_detects_login_prompt_from_body_when_not_home(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_default_config(config_path)
    create_profile("default", config_path)
    login_and_save_storage_state(
        profile_name="default",
        config_path=config_path,
        capture_fn=_fake_storage_state,
    )

    result = probe_auth_status(
        profile_name="default",
        config_path=config_path,
        probe_fn=lambda _config, _path: AuthProbeSnapshot(
            current_url="https://x.com/explore",
            page_title="Explore / X",
            body_text="Your session expired. Please log in to continue.",
        ),
    )

    assert result.authenticated is False
    assert result.status_code == "blocked_login_wall"


def test_probe_auth_status_detects_challenge(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_default_config(config_path)
    create_profile("default", config_path)
    login_and_save_storage_state(
        profile_name="default",
        config_path=config_path,
        capture_fn=_fake_storage_state,
    )

    result = probe_auth_status(
        profile_name="default",
        config_path=config_path,
        probe_fn=lambda _config, _path: AuthProbeSnapshot(
            current_url="https://x.com/account/access",
            page_title="Security challenge",
            body_text="Confirm it's you",
        ),
    )

    assert result.authenticated is False
    assert result.status_code == "blocked_challenge"


def test_probe_auth_status_fail_closed_when_probe_errors(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_default_config(config_path)
    create_profile("default", config_path)
    login_and_save_storage_state(
        profile_name="default",
        config_path=config_path,
        capture_fn=_fake_storage_state,
    )

    def failing_probe(_config: object, _path: Path) -> AuthProbeSnapshot:
        raise AuthError("probe failed")

    result = probe_auth_status(
        profile_name="default",
        config_path=config_path,
        probe_fn=failing_probe,
    )

    assert result.authenticated is False
    assert result.status_code == "unconfirmed"
    assert "fail-closed" in result.message


def test_probe_auth_status_reports_missing_storage_state(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_default_config(config_path)
    create_profile("default", config_path)

    result = probe_auth_status(profile_name="default", config_path=config_path)

    assert result.authenticated is False
    assert result.status_code == "missing_storage_state"


def test_logout_profile_removes_storage_state(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_default_config(config_path)
    create_profile("default", config_path)
    stored = login_and_save_storage_state(
        profile_name="default",
        config_path=config_path,
        capture_fn=_fake_storage_state,
    )
    assert stored.exists()

    result = logout_profile(profile_name="default", config_path=config_path)

    assert result.removed is True
    assert stored.exists() is False


def test_logout_profile_is_idempotent_when_storage_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_default_config(config_path)
    create_profile("default", config_path)

    result = logout_profile(profile_name="default", config_path=config_path)

    assert result.removed is False
    assert "Already logged out" in result.message
