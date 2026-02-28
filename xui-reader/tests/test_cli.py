"""CLI smoke tests for scaffold entrypoint behavior."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from xui_reader import __version__
from xui_reader.auth import AuthProbeSnapshot, storage_state_path
from xui_reader.diagnostics.base import DiagnosticReport, DiagnosticSection
from xui_reader.models import TweetItem
from xui_reader.scheduler.read import MultiSourceReadResult, SourceReadOutcome
from xui_reader.scheduler.watch import WatchCycleResult, WatchRunResult

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


def test_config_show_reports_actionable_error_for_missing_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.toml"
    result = runner.invoke(app, ["config", "show", "--path", str(missing_path)])
    assert result.exit_code == 2
    assert "Config show failed:" in result.output
    assert "Run `xui config init" in result.output


def test_config_init_reports_force_hint_when_file_exists(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("existing", encoding="utf-8")
    result = runner.invoke(app, ["config", "init", "--path", str(config_path)])
    assert result.exit_code == 2
    assert "Config init failed:" in result.output
    assert "--force" in result.output


def test_profiles_lifecycle_and_active_delete_safety(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"

    init_result = runner.invoke(app, ["config", "init", "--path", str(config_path)])
    assert init_result.exit_code == 0

    create_default = runner.invoke(
        app, ["profiles", "create", "default", "--path", str(config_path)]
    )
    assert create_default.exit_code == 0

    create_ops = runner.invoke(app, ["profiles", "create", "ops", "--path", str(config_path)])
    assert create_ops.exit_code == 0

    switch_ops = runner.invoke(app, ["profiles", "switch", "ops", "--path", str(config_path)])
    assert switch_ops.exit_code == 0
    assert "Active profile set to 'ops'" in switch_ops.output

    delete_active = runner.invoke(app, ["profiles", "delete", "ops", "--path", str(config_path)])
    assert delete_active.exit_code == 2
    assert "Cannot delete active profile 'ops'" in delete_active.output

    switch_default = runner.invoke(
        app, ["profiles", "switch", "default", "--path", str(config_path)]
    )
    assert switch_default.exit_code == 0

    delete_ops = runner.invoke(app, ["profiles", "delete", "ops", "--path", str(config_path)])
    assert delete_ops.exit_code == 0

    list_result = runner.invoke(app, ["profiles", "list", "--path", str(config_path)])
    assert list_result.exit_code == 0
    assert "* default" in list_result.output
    assert "ops" not in list_result.output


def test_auth_login_does_not_echo_storage_state_secrets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.toml"
    init_result = runner.invoke(app, ["config", "init", "--path", str(config_path)])
    assert init_result.exit_code == 0

    create_default = runner.invoke(
        app, ["profiles", "create", "default", "--path", str(config_path)]
    )
    assert create_default.exit_code == 0

    def fake_capture(_config: object, _login_url: str) -> dict[str, object]:
        return {
            "cookies": [{"name": "sessionid", "value": "super-secret-cookie"}],
            "origins": [],
        }

    monkeypatch.setattr("xui_reader.auth._capture_storage_state_via_playwright", fake_capture)

    login_result = runner.invoke(
        app,
        ["auth", "login", "--path", str(config_path), "--profile", "default"],
    )
    assert login_result.exit_code == 0
    assert "Saved storage_state to" in login_result.output
    assert "super-secret-cookie" not in login_result.output

    saved_path = storage_state_path("default", config_path)
    assert saved_path.exists()


def test_auth_status_is_fail_closed_when_storage_state_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_result = runner.invoke(app, ["config", "init", "--path", str(config_path)])
    assert init_result.exit_code == 0
    create_default = runner.invoke(
        app, ["profiles", "create", "default", "--path", str(config_path)]
    )
    assert create_default.exit_code == 0

    status_result = runner.invoke(
        app,
        ["auth", "status", "--path", str(config_path), "--profile", "default"],
    )
    assert status_result.exit_code == 2
    assert "missing_storage_state" in status_result.output
    assert "xui auth login" in status_result.output


def test_auth_status_returns_success_for_authenticated_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.toml"
    init_result = runner.invoke(app, ["config", "init", "--path", str(config_path)])
    assert init_result.exit_code == 0
    create_default = runner.invoke(
        app, ["profiles", "create", "default", "--path", str(config_path)]
    )
    assert create_default.exit_code == 0

    def fake_capture(_config: object, _login_url: str) -> dict[str, object]:
        return {"cookies": [{"name": "sessionid", "value": "secret"}], "origins": []}

    monkeypatch.setattr("xui_reader.auth._capture_storage_state_via_playwright", fake_capture)
    login_result = runner.invoke(
        app, ["auth", "login", "--path", str(config_path), "--profile", "default"]
    )
    assert login_result.exit_code == 0

    monkeypatch.setattr(
        "xui_reader.auth._probe_auth_with_playwright",
        lambda _config, _path: AuthProbeSnapshot(
            current_url="https://x.com/home", page_title="Home / X", body_text=""
        ),
    )
    status_result = runner.invoke(
        app, ["auth", "status", "--path", str(config_path), "--profile", "default"]
    )
    assert status_result.exit_code == 0
    assert "authenticated" in status_result.output


def test_auth_status_surfaces_challenge_guidance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.toml"
    init_result = runner.invoke(app, ["config", "init", "--path", str(config_path)])
    assert init_result.exit_code == 0
    create_default = runner.invoke(
        app, ["profiles", "create", "default", "--path", str(config_path)]
    )
    assert create_default.exit_code == 0

    monkeypatch.setattr(
        "xui_reader.auth._capture_storage_state_via_playwright",
        lambda _config, _login_url: {
            "cookies": [{"name": "sessionid", "value": "secret"}],
            "origins": [],
        },
    )
    login_result = runner.invoke(
        app, ["auth", "login", "--path", str(config_path), "--profile", "default"]
    )
    assert login_result.exit_code == 0

    monkeypatch.setattr(
        "xui_reader.auth._probe_auth_with_playwright",
        lambda _config, _path: AuthProbeSnapshot(
            current_url="https://x.com/account/access",
            page_title="Security challenge",
            body_text="Confirm it's you",
        ),
    )
    status_result = runner.invoke(
        app, ["auth", "status", "--path", str(config_path), "--profile", "default"]
    )
    assert status_result.exit_code == 2
    assert "blocked_challenge" in status_result.output
    assert "xui auth login" in status_result.output


def test_auth_logout_removes_storage_state_and_reports_relogin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.toml"
    init_result = runner.invoke(app, ["config", "init", "--path", str(config_path)])
    assert init_result.exit_code == 0
    create_default = runner.invoke(
        app, ["profiles", "create", "default", "--path", str(config_path)]
    )
    assert create_default.exit_code == 0
    monkeypatch.setattr(
        "xui_reader.auth._capture_storage_state_via_playwright",
        lambda _config, _login_url: {
            "cookies": [{"name": "sessionid", "value": "secret"}],
            "origins": [],
        },
    )
    login_result = runner.invoke(
        app, ["auth", "login", "--path", str(config_path), "--profile", "default"]
    )
    assert login_result.exit_code == 0

    logout_result = runner.invoke(
        app, ["auth", "logout", "--path", str(config_path), "--profile", "default"]
    )
    assert logout_result.exit_code == 0
    assert "Removed storage_state" in logout_result.output
    assert "Re-login" in logout_result.output


def test_doctor_reports_guidance_when_no_sources_configured(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")

    result = runner.invoke(app, ["doctor", "--path", str(config_path)])
    assert result.exit_code == 0
    assert "Doctor preflight" in result.output
    assert "No configured sources; skipping optional smoke checks." in result.output
    assert "Guidance:" in result.output


def test_doctor_reports_guidance_when_all_sources_disabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[[sources]]
id = "list:1"
kind = "list"
list_id = "1"
enabled = false
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["doctor", "--path", str(config_path)])
    assert result.exit_code == 0
    assert "All configured sources are disabled; skipping optional smoke checks." in result.output
    assert "Selected smoke sources: none" in result.output


def test_doctor_json_reports_selected_sources(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_result = runner.invoke(app, ["config", "init", "--path", str(config_path)])
    assert init_result.exit_code == 0

    result = runner.invoke(app, ["doctor", "--path", str(config_path), "--json", "--max-sources", "1"])
    assert result.exit_code == 0
    assert '"ok": false' in result.output
    assert '"selected_source_ids"' in result.output
    assert '"sections"' in result.output


def test_doctor_command_profile_overrides_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_result = runner.invoke(app, ["config", "init", "--path", str(config_path)])
    assert init_result.exit_code == 0

    captured: dict[str, object] = {}

    def fake_doctor(
        _config: object,
        *,
        profile_name: str | None,
        config_path: str | None,
        max_sources: int,
    ) -> DiagnosticReport:
        captured["profile_name"] = profile_name
        captured["config_path"] = config_path
        captured["max_sources"] = max_sources
        return DiagnosticReport(ok=True)

    monkeypatch.setattr("xui_reader.cli.run_doctor_preflight", fake_doctor)

    result = runner.invoke(
        app,
        ["doctor", "--path", str(config_path), "--profile", "ops"],
    )

    assert result.exit_code == 0
    assert captured["profile_name"] == "ops"
    assert captured["config_path"] == str(config_path)
    assert captured["max_sources"] == 2


def test_doctor_uses_global_profile_when_command_profile_omitted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    init_result = runner.invoke(app, ["config", "init", "--path", str(config_path)])
    assert init_result.exit_code == 0

    captured: dict[str, object] = {}

    def fake_doctor(
        _config: object,
        *,
        profile_name: str | None,
        config_path: str | None,
        max_sources: int,
    ) -> DiagnosticReport:
        captured["profile_name"] = profile_name
        captured["config_path"] = config_path
        captured["max_sources"] = max_sources
        return DiagnosticReport(ok=True)

    monkeypatch.setattr("xui_reader.cli.run_doctor_preflight", fake_doctor)

    result = runner.invoke(
        app,
        ["--profile", "ops", "doctor", "--path", str(config_path)],
    )

    assert result.exit_code == 0
    assert captured["profile_name"] == "ops"
    assert captured["config_path"] == str(config_path)
    assert captured["max_sources"] == 2


def test_doctor_json_redacts_sensitive_details(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_result = runner.invoke(app, ["config", "init", "--path", str(config_path)])
    assert init_result.exit_code == 0

    report = DiagnosticReport(
        ok=False,
        checks=("FAIL auth: token leak",),
        details={"selected_source_ids": "list:1", "authorization": "Bearer super-secret"},
        sections=(
            DiagnosticSection(
                name="auth",
                ok=False,
                summary="Auth failed",
                details={"storage_state": '{"cookies":[{"name":"sessionid","value":"secret"}]}'},
            ),
        ),
    )
    monkeypatch.setattr("xui_reader.cli.run_doctor_preflight", lambda *_args, **_kwargs: report)

    result = runner.invoke(app, ["doctor", "--path", str(config_path), "--json"])
    assert result.exit_code == 0
    assert "super-secret" not in result.output
    assert "sessionid" not in result.output
    assert "<redacted>" in result.output


def test_read_json_reports_outcomes_and_merged_items(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.toml"
    init_result = runner.invoke(app, ["config", "init", "--path", str(config_path)])
    assert init_result.exit_code == 0

    payload = MultiSourceReadResult(
        items=(
            TweetItem(
                tweet_id="10",
                created_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
                author_handle="@a",
                text="hello",
                source_id="list:1",
            ),
        ),
        outcomes=(
            SourceReadOutcome(
                source_id="list:1",
                source_kind="list",
                ok=True,
                item_count=1,
                error=None,
            ),
            SourceReadOutcome(
                source_id="user:a",
                source_kind="user",
                ok=False,
                item_count=0,
                error="collect failed",
            ),
        ),
    )
    monkeypatch.setattr("xui_reader.cli.run_configured_read", lambda *_args, **_kwargs: payload)

    result = runner.invoke(app, ["read", "--path", str(config_path), "--json"])
    assert result.exit_code == 0
    assert '"succeeded_sources": 1' in result.output
    assert '"failed_sources": 1' in result.output
    assert '"source_id": "user:a"' in result.output


def test_watch_json_reports_cycle_timing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_result = runner.invoke(app, ["config", "init", "--path", str(config_path)])
    assert init_result.exit_code == 0

    payload = WatchRunResult(
        cycles=(
            WatchCycleResult(
                cycle=1,
                started_at=datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc),
                next_run_at=None,
                sleep_seconds=0.0,
                emitted_items=2,
                succeeded_sources=1,
                failed_sources=0,
            ),
        )
    )
    monkeypatch.setattr("xui_reader.cli.run_configured_watch", lambda *_args, **_kwargs: payload)

    result = runner.invoke(app, ["watch", "--path", str(config_path), "--json"])
    assert result.exit_code == 0
    assert '"cycle": 1' in result.output
    assert '"emitted_items": 2' in result.output
    assert '"exit_code": 0' in result.output
    assert '"exit_state": "success"' in result.output


def test_watch_returns_budget_stop_exit_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_result = runner.invoke(app, ["config", "init", "--path", str(config_path)])
    assert init_result.exit_code == 0

    payload = WatchRunResult(
        cycles=(
            WatchCycleResult(
                cycle=1,
                started_at=datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc),
                next_run_at=datetime(2026, 3, 1, 0, 5, tzinfo=timezone.utc),
                sleep_seconds=300.0,
                emitted_items=1,
                succeeded_sources=1,
                failed_sources=0,
            ),
            WatchCycleResult(
                cycle=2,
                started_at=datetime(2026, 3, 1, 0, 5, tzinfo=timezone.utc),
                next_run_at=None,
                sleep_seconds=0.0,
                emitted_items=1,
                succeeded_sources=1,
                failed_sources=0,
            ),
        )
    )
    monkeypatch.setattr("xui_reader.cli.run_configured_watch", lambda *_args, **_kwargs: payload)

    result = runner.invoke(app, ["watch", "--path", str(config_path), "--max-cycles", "2"])
    assert result.exit_code == 4
    assert "exit_state=budget_stop" in result.output


def test_watch_returns_auth_fail_exit_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    init_result = runner.invoke(app, ["config", "init", "--path", str(config_path)])
    assert init_result.exit_code == 0

    payload = WatchRunResult(
        cycles=(
            WatchCycleResult(
                cycle=1,
                started_at=datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc),
                next_run_at=None,
                sleep_seconds=0.0,
                emitted_items=0,
                succeeded_sources=0,
                failed_sources=2,
                auth_failed_sources=2,
            ),
        )
    )
    monkeypatch.setattr("xui_reader.cli.run_configured_watch", lambda *_args, **_kwargs: payload)

    result = runner.invoke(app, ["watch", "--path", str(config_path), "--json"])
    assert result.exit_code == 5
    assert '"exit_state": "auth_fail"' in result.output
