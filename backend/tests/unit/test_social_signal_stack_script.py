"""Behavior tests for the Social Signal Docker convenience wrapper."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "social-signal-stack.sh"


def _executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _environment(tmp_path: Path, *, worker_image: str = "stock-screener-social-xui:dev"):
    command_log = tmp_path / "commands.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _executable(
        bin_dir / "docker",
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$COMMAND_LOG\"\n",
    )
    _executable(bin_dir / "ssh-add", "#!/usr/bin/env bash\nexit 0\n")
    profile = tmp_path / "xui-profile"
    profile.mkdir()
    (profile / "config.toml").write_text("[fixture]\n", encoding="utf-8")
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("github.com fixture-key\n", encoding="utf-8")
    env_file = tmp_path / ".env.social"
    env_file.write_text(
        "\n".join(
            (
                "SERVER_AUTH_PASSWORD=fixture-password",
                "XUI_READER_REF=" + "a" * 40,
                f"GITHUB_KNOWN_HOSTS_FILE={known_hosts}",
                f"XUI_PROFILE_DIR={profile}",
                f"SOCIAL_WORKER_IMAGE={worker_image}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "COMMAND_LOG": str(command_log),
        "SOCIAL_STACK_ENV_FILE": str(env_file),
    }
    return env, command_log


def test_local_up_builds_then_starts_normal_stack_then_private_worker(tmp_path):
    env, command_log = _environment(tmp_path)

    result = subprocess.run(
        ["bash", str(SCRIPT), "local", "up"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    commands = command_log.read_text(encoding="utf-8").splitlines()
    assert commands[0].startswith("build --ssh default --secret id=github_known_hosts,src=")
    assert "--target social-xui" in commands[0]
    assert "--build-arg XUI_READER_REF=" + "a" * 40 in commands[0]
    assert "-t stock-screener-social-xui:dev" in commands[0]
    assert commands[1].startswith("compose --env-file ")
    assert commands[1].endswith("up -d")
    assert "docker-compose.social" not in commands[1]
    assert "-f " + str(ROOT / "docker-compose.social.yml") in commands[2]
    assert "-f " + str(ROOT / "docker-compose.social-xui.yml") in commands[2]
    assert commands[2].endswith("--profile social up -d --no-build celery-social")


def test_ghcr_up_pulls_immutable_image_before_starting_stack(tmp_path):
    image = "ghcr.io/xang1234/stock-screener-social-xui:sha-" + "b" * 40
    env, command_log = _environment(tmp_path, worker_image=image)

    result = subprocess.run(
        ["bash", str(SCRIPT), "ghcr", "up"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    commands = command_log.read_text(encoding="utf-8").splitlines()
    assert commands[0].endswith("--profile social pull celery-social")
    assert commands[1].startswith("compose --env-file ")
    assert commands[1].endswith("up -d")
    assert commands[2].endswith("--profile social up -d --no-build celery-social")
    assert all(not command.startswith("build ") for command in commands)


def test_ghcr_up_rejects_mutable_latest_before_docker_access(tmp_path):
    env, command_log = _environment(
        tmp_path,
        worker_image="ghcr.io/xang1234/stock-screener-social-xui:latest",
    )

    result = subprocess.run(
        ["bash", str(SCRIPT), "ghcr", "up"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "latest is forbidden" in result.stderr
    assert not command_log.exists()


@pytest.mark.parametrize(
    ("action", "expected_suffix"),
    (
        ("status", "--profile social ps celery-social"),
        ("logs", "--profile social logs -f celery-social"),
        ("stop", "--profile social stop celery-social"),
    ),
)
def test_operational_action_targets_only_social_worker(
    tmp_path, action, expected_suffix
):
    env, command_log = _environment(tmp_path)

    result = subprocess.run(
        ["bash", str(SCRIPT), action],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    commands = command_log.read_text(encoding="utf-8").splitlines()
    assert len(commands) == 1
    assert commands[0].endswith(expected_suffix)


def test_explicit_relative_env_file_is_resolved_from_repository(tmp_path):
    env, command_log = _environment(tmp_path)
    env_file = Path(env["SOCIAL_STACK_ENV_FILE"])
    env["SOCIAL_STACK_ENV_FILE"] = os.path.relpath(env_file, ROOT)

    result = subprocess.run(
        ["bash", str(SCRIPT), "status"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert command_log.read_text(encoding="utf-8").splitlines()[0].endswith(
        "--profile social ps celery-social"
    )
