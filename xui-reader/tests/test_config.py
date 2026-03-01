"""Config init/show defaults and validation behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from xui_reader.config import (
    default_config,
    default_config_toml,
    init_default_config,
    load_runtime_config,
    resolve_config_path,
)
from xui_reader.errors import ConfigError
from xui_reader.models import SourceKind


def test_resolve_config_path_uses_explicit_path() -> None:
    path = resolve_config_path("~/tmp/xui-test.toml")
    assert str(path).endswith("xui-test.toml")


def test_resolve_config_path_uses_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_path = tmp_path / "env-config.toml"
    monkeypatch.setenv("XUI_CONFIG", str(env_path))
    assert resolve_config_path() == env_path


def test_init_default_config_writes_template(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    written = init_default_config(config_path)
    assert written == config_path
    assert config_path.exists()
    assert default_config_toml().strip() in config_path.read_text(encoding="utf-8")


def test_init_default_config_requires_force_for_overwrite(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("existing", encoding="utf-8")
    with pytest.raises(ConfigError, match="--force"):
        init_default_config(config_path)


def test_load_runtime_config_reports_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.toml"
    with pytest.raises(ConfigError, match="Run `xui config init"):
        load_runtime_config(missing)


def test_load_runtime_config_reports_invalid_value(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """[app]
default_profile = ""
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="app.default_profile"):
        load_runtime_config(config_path)


def test_load_runtime_config_reports_invalid_source_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """[app]
default_profile = "default"

[[sources]]
id = "list:abc"
kind = "list"
enabled = true
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r"sources\[0\]\.list_id"):
        load_runtime_config(config_path)


def test_load_runtime_config_reports_invalid_browser_engine(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """[app]
default_profile = "default"

[browser]
engine = "edge"
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="browser.engine"):
        load_runtime_config(config_path)


def test_load_runtime_config_parses_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_default_config(config_path)
    loaded = load_runtime_config(config_path)

    assert loaded.app.default_profile == "default"
    assert loaded.app.default_format == "pretty"
    assert loaded.browser.navigation_timeout_ms == 30_000
    assert loaded.collection.max_scrolls == 10
    assert loaded.checkpoints.mode == "id"
    assert loaded.scheduler.interval_sec == 3600
    assert loaded.storage.db_filename == "tweets.sqlite3"
    assert loaded.selectors.override_filename == "selectors/override.json"
    assert loaded.sources[0].kind == SourceKind.LIST
    assert loaded.sources[1].kind == SourceKind.USER


def test_headless_default_is_consistent_between_dataclass_and_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_default_config(config_path)
    loaded = load_runtime_config(config_path)
    assert default_config().browser.headless is True
    assert loaded.browser.headless is True


def test_init_default_config_wraps_os_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.toml"

    def raise_permission_error(*_args: object, **_kwargs: object) -> str:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "write_text", raise_permission_error)

    with pytest.raises(ConfigError, match="Could not write config file"):
        init_default_config(config_path)


def test_load_runtime_config_wraps_os_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("[app]\ndefault_profile = \"default\"\n", encoding="utf-8")

    def raise_permission_error(*_args: object, **_kwargs: object) -> str:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_text", raise_permission_error)

    with pytest.raises(ConfigError, match="Could not read config file"):
        load_runtime_config(config_path)
