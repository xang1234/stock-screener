"""Deployment worker queue coverage for supported markets."""

from __future__ import annotations

import os
import subprocess
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


def test_docker_compose_forwards_opendart_api_key_to_app_env():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "OPENDART_API_KEY: ${OPENDART_API_KEY:-}" in compose


def test_local_celery_script_consumes_every_supported_market_queue():
    script = (ROOT / "backend" / "start_celery.sh").read_text(encoding="utf-8")
    supported_csv = ",".join(SUPPORTED_MARKETS)
    supported_case = "|".join(SUPPORTED_MARKETS)

    assert f'ENABLED_MARKETS="${{ENABLED_MARKETS:-{supported_csv}}}"' in script
    assert f"data_fetch_shared,{','.join(f'data_fetch_{market.lower()}' for market in SUPPORTED_MARKETS)}" in script
    assert f"{supported_case}) ;;" in script


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
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "printf 'DOCKER_ARGS'",
                "for arg in \"$@\"; do printf '|%s' \"$arg\"; done",
                "printf '\\n'",
                "printf 'DOCKER_ENABLED_MARKETS=%s\\n' \"$ENABLED_MARKETS\"",
                "printf 'DOCKER_COMPOSE_PROFILES=%s\\n' \"$COMPOSE_PROFILES\"",
            ]
        ),
        encoding="utf-8",
    )
    docker.chmod(0o755)
    return bin_dir


def _wrapper_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{_fake_docker_bin(tmp_path)}{os.pathsep}{env['PATH']}"
    env.pop("ENABLED_MARKETS", None)
    env.pop("COMPOSE_PROFILES", None)
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
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"COMPOSE_PROFILES={expected_market_profiles}" in result.stdout
    assert f"DOCKER_ARGS|compose|--env-file|{env_file}|down|--remove-orphans" in result.stdout
