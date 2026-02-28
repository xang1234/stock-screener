"""Profile lifecycle behavior and guardrails."""

from __future__ import annotations

from pathlib import Path

import pytest

from xui_reader.config import init_default_config, load_runtime_config
from xui_reader.errors import ProfileError
from xui_reader.profiles import create_profile, delete_profile, list_profiles, switch_profile


def test_create_profile_bootstraps_expected_subdirectories(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_default_config(config_path)

    created = create_profile("default", config_path)

    assert created == tmp_path / "profiles" / "default"
    assert (created / "session").is_dir()
    assert (created / "artifacts").is_dir()
    assert (created / "logs").is_dir()


def test_switch_profile_with_create_updates_default_profile(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_default_config(config_path)
    create_profile("default", config_path)

    switched = switch_profile("ops", config_path, create_missing=True)
    loaded = load_runtime_config(config_path)

    assert switched == tmp_path / "profiles" / "ops"
    assert loaded.app.default_profile == "ops"


def test_delete_profile_blocks_active_profile(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_default_config(config_path)
    create_profile("default", config_path)

    with pytest.raises(ProfileError, match="Cannot delete active profile"):
        delete_profile("default", config_path)


def test_list_profiles_includes_active_marker(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_default_config(config_path)
    create_profile("default", config_path)
    create_profile("ops", config_path)
    switch_profile("ops", config_path)

    profiles, active = list_profiles(config_path)

    assert active == "ops"
    assert [profile.name for profile in profiles] == ["default", "ops"]
    assert [profile.active for profile in profiles] == [False, True]


def test_list_profiles_wraps_os_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.toml"
    init_default_config(config_path)
    create_profile("default", config_path)

    def raise_permission_error(*_args: object, **_kwargs: object) -> object:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "iterdir", raise_permission_error)

    with pytest.raises(ProfileError, match="Could not list profiles"):
        list_profiles(config_path)


def test_create_profile_wraps_os_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.toml"
    init_default_config(config_path)

    def raise_permission_error(*_args: object, **_kwargs: object) -> object:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "mkdir", raise_permission_error)

    with pytest.raises(ProfileError, match="Could not create or initialize profile"):
        create_profile("default", config_path)


def test_switch_profile_wraps_os_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.toml"
    init_default_config(config_path)
    create_profile("default", config_path)

    original_exists = Path.exists

    def raise_permission_error(path_obj: Path, *_args: object, **_kwargs: object) -> object:
        if path_obj.name == "ops":
            raise PermissionError("denied")
        return original_exists(path_obj)

    monkeypatch.setattr(Path, "exists", raise_permission_error)

    with pytest.raises(ProfileError, match="Could not access profile"):
        switch_profile("ops", config_path)


def test_delete_profile_wraps_os_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.toml"
    init_default_config(config_path)
    create_profile("default", config_path)
    create_profile("ops", config_path)

    def raise_permission_error(*_args: object, **_kwargs: object) -> object:
        raise PermissionError("denied")

    monkeypatch.setattr("xui_reader.profiles.shutil.rmtree", raise_permission_error)

    with pytest.raises(ProfileError, match="Could not delete profile"):
        delete_profile("ops", config_path)
