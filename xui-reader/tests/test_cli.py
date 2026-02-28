"""CLI smoke tests for scaffold entrypoint behavior."""

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
