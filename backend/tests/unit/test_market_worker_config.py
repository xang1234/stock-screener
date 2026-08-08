"""Deployment worker queue coverage for supported markets."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from app.tasks.market_queues import SUPPORTED_MARKETS

ROOT = Path(__file__).resolve().parents[3]


def test_docker_compose_profiles_every_market_worker():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    for market in SUPPORTED_MARKETS:
        suffix = market.lower()
        assert f"market_jobs_{suffix}" in compose
        assert f"user_scans_{suffix}" in compose
        assert f'profiles: ["market-{suffix}"]' in compose


def test_docker_compose_datafetch_queues_are_derived_from_enabled_markets():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "compose_enabled_markets.py queues" in compose
    assert "data_fetch_shared,data_fetch_us,data_fetch_hk" not in compose
    assert "-Q \"$$QUEUES\"" in compose


def test_docker_compose_userscans_base_worker_computes_queues_from_enabled_markets():
    # celery-userscans must not rely on Compose profiles being activated to
    # cover the enabled markets' user_scans_<mkt> queues (see stuck-scan bug
    # where a manual scan sat "queued" forever with no worker consuming it).
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "compose_enabled_markets.py queues --queue-set userscans" in compose
    assert "-n userscans-shared@%h" in compose


def test_docker_compose_marketjobs_base_worker_computes_queues_from_enabled_markets():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "  celery-marketjobs:\n" in compose
    assert "compose_enabled_markets.py queues --queue-set marketjobs" in compose
    assert "-n marketjobs-default@%h" in compose


def test_release_overlay_uses_release_image_for_every_market_worker():
    release = (ROOT / "docker-compose.release.yml").read_text(encoding="utf-8")

    for service in ("backend", "celery-general", "celery-datafetch", "celery-userscans", "celery-marketjobs", "celery-beat"):
        assert (
            f"  {service}:\n"
            "    build: !reset null\n"
            "    <<: *backend-release-service"
        ) in release

    for market in SUPPORTED_MARKETS:
        suffix = market.lower()
        assert (
            f"  celery-marketjobs-{suffix}:\n"
            "    build: !reset null\n"
            "    <<: *backend-release-service"
        ) in release
        assert (
            f"  celery-userscans-{suffix}:\n"
            "    build: !reset null\n"
            "    <<: *backend-release-service"
        ) in release


def test_docker_compose_forwards_opendart_api_key_to_app_env():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "OPENDART_API_KEY: ${OPENDART_API_KEY:-}" in compose


def test_local_celery_script_consumes_every_supported_market_queue():
    script = (ROOT / "backend" / "start_celery.sh").read_text(encoding="utf-8")

    assert "from app.tasks.market_queues import SUPPORTED_MARKETS" in script
    assert "from app.tasks.market_queues import all_data_fetch_queues" in script
    assert 'ENABLED_MARKETS="${ENABLED_MARKETS:-$SUPPORTED_MARKETS}"' in script
    assert '-Q "$DATA_FETCH_QUEUES"' in script
    assert "case \"$MARKET_UPPER\"" not in script


def test_enabled_market_compose_wrapper_reads_env_files_and_preserves_profiles():
    script = (ROOT / "scripts" / "docker-compose-enabled-markets.sh").read_text(encoding="utf-8")

    assert "read_env_value" in script
    assert "env_files_from_args" in script
    assert "RESOLVED_ENV_FILES" in script
    assert "read_env_value_from_files ENABLED_MARKETS" in script
    assert "read_env_value_from_files COMPOSE_PROFILES" in script
    assert "PROFILES=\"$COMPOSE_PROFILES,$MARKET_PROFILES\"" in script
    assert 'COMPOSE_ARGS+=(--env-file "$ENV_FILE_TO_FORWARD")' in script
    assert 'COMPOSE_ARGS+=("--remove-orphans")' in script


def _fake_docker_bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'DOCKER_ARGS'\n"
        "for arg in \"$@\"; do printf '|%s' \"$arg\"; done\n"
        "printf '\\n'\n"
        "printf 'DOCKER_ENABLED_MARKETS=%s\\n' \"$ENABLED_MARKETS\"\n"
        "printf 'DOCKER_COMPOSE_PROFILES=%s\\n' \"$COMPOSE_PROFILES\"\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    if not (bin_dir / "python3.11").exists():
        _write_fake_python(
            bin_dir,
            "python3.11",
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    f"exec {shlex.quote(sys.executable)} \"$@\"",
                ]
            ),
        )
    return bin_dir


def _write_fake_python(bin_dir: Path, name: str, body: str) -> Path:
    python = bin_dir / name
    python.write_text(body, encoding="utf-8")
    python.chmod(0o755)
    return python


def _path_with_shell_bins(path: str | None) -> str:
    parts = [part for part in (path or os.defpath).split(os.pathsep) if part]
    for required in ("/usr/bin", "/bin"):
        if required not in parts:
            parts.append(required)
    return os.pathsep.join(parts)


def _wrapper_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = (
        f"{_fake_docker_bin(tmp_path)}{os.pathsep}"
        f"{_path_with_shell_bins(env.get('PATH'))}"
    )
    env.pop("ENABLED_MARKETS", None)
    env.pop("COMPOSE_PROFILES", None)
    env.pop("STOCKSCREEN_PYTHON", None)
    return env


def test_enabled_market_compose_wrapper_uses_last_env_file_values(tmp_path):
    base_env = tmp_path / "base.env"
    override_env = tmp_path / "override.env"
    base_env.write_text("ENABLED_MARKETS=US\nCOMPOSE_PROFILES=assistant\n", encoding="utf-8")
    override_env.write_text("ENABLED_MARKETS=HK,CN\nCOMPOSE_PROFILES=debug\n", encoding="utf-8")

    result = subprocess.run(
        [
            str(ROOT / "scripts" / "docker-compose-enabled-markets.sh"),
            "--env-file",
            str(base_env),
            "--env-file",
            str(override_env),
            "config",
        ],
        cwd=ROOT,
        env=_wrapper_env(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "ENABLED_MARKETS=HK,CN" in result.stdout
    assert "COMPOSE_PROFILES=debug,market-hk,market-cn" in result.stdout
    assert (
        f"DOCKER_ARGS|compose|--env-file|{base_env}|--env-file|{override_env}|config"
        in result.stdout
    )


def test_enabled_market_compose_wrapper_down_enables_all_market_profiles(tmp_path):
    expected_market_profiles = ",".join(f"market-{market.lower()}" for market in SUPPORTED_MARKETS)
    env_file = tmp_path / "empty.env"
    env_file.write_text("", encoding="utf-8")

    result = subprocess.run(
        [
            str(ROOT / "scripts" / "docker-compose-enabled-markets.sh"),
            "--env-file",
            str(env_file),
            "down",
        ],
        cwd=ROOT,
        env={**_wrapper_env(tmp_path), "ENABLED_MARKETS": "US"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert f"COMPOSE_PROFILES={expected_market_profiles}" in result.stdout
    assert f"DOCKER_ARGS|compose|--env-file|{env_file}|down|--remove-orphans" in result.stdout


def test_enabled_market_compose_wrapper_prefers_python311_over_old_python3(tmp_path):
    bin_dir = _fake_docker_bin(tmp_path)
    marker = tmp_path / "python311.invocations"
    _write_fake_python(
        bin_dir,
        "python3",
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"-\" ]]; then exit 1; fi\n"
        "printf 'old python3 should not run backend helpers\\n' >&2\n"
        "exit 42\n",
    )
    _write_fake_python(
        bin_dir,
        "python3.11",
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f"printf '%s\\n' \"$*\" >> {marker}",
                f"exec {shlex.quote(sys.executable)} \"$@\"",
            ]
        ),
    )

    result = subprocess.run(
        [
            str(ROOT / "scripts" / "docker-compose-enabled-markets.sh"),
            "config",
        ],
        cwd=ROOT,
        env={**_wrapper_env(tmp_path), "ENABLED_MARKETS": "US,HK"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "ENABLED_MARKETS=US,HK" in result.stdout
    assert "COMPOSE_PROFILES=market-us,market-hk" in result.stdout
    assert "DOCKER_ARGS|compose|" in result.stdout
    assert result.stdout.splitlines()[2].endswith("|config")
    assert str(ROOT / "backend" / "scripts" / "compose_enabled_markets.py") in marker.read_text(
        encoding="utf-8"
    )
