"""CLI smoke tests for scaffold entrypoint behavior."""

from pathlib import Path

import pytest

from xui_reader import __version__

pytest.importorskip("typer")

from typer.testing import CliRunner

from xui_reader.cli import app

runner = CliRunner()


def test_cli_help_lists_scaffold_command_groups() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "auth" in result.output
    assert "profiles" in result.output
    assert "list" in result.output
    assert "user" in result.output
    assert "read" in result.output
    assert "watch" in result.output
    assert "doctor" in result.output
    assert "config" in result.output
    assert "--profile" in result.output
    assert "--format" in result.output
    assert "--headful" in result.output
    assert "--headless" in result.output
    assert "--debug" in result.output
    assert "--timeout-ms" in result.output


def test_cli_version_flag_prints_package_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_config_init_and_show_json(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"

    init_result = runner.invoke(app, ["config", "init", "--path", str(config_path)])
    assert init_result.exit_code == 0
    assert config_path.exists()

    show_result = runner.invoke(app, ["config", "show", "--path", str(config_path), "--json"])
    assert show_result.exit_code == 0
    assert '"path":' in show_result.output
    assert '"default_profile": "default"' in show_result.output
