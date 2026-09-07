"""Security and topology contract for the optional Social worker."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "backend" / "Dockerfile"
BASE_COMPOSE = ROOT / "docker-compose.yml"
SOCIAL_COMPOSE = ROOT / "docker-compose.social.yml"
XUI_COMPOSE = ROOT / "docker-compose.social-xui.yml"


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _environment_map(service: dict) -> dict[str, str]:
    environment = service.get("environment") or {}
    if isinstance(environment, dict):
        return environment
    return dict(item.split("=", 1) for item in environment)


def test_public_social_worker_is_opt_in_fail_closed_and_queue_is_isolated():
    base = _yaml(BASE_COMPOSE)
    overlay = _yaml(SOCIAL_COMPOSE)
    base_worker = base["services"]["celery-social"]
    worker = {**base_worker, **overlay["services"]["celery-social"]}

    assert worker["profiles"] == ["social"]
    assert str(worker["user"]) not in {"root", "0", "0:0"}
    assert worker["build"]["target"] == "public"
    environment = _environment_map(base_worker)
    assert environment["SOCIAL_SIGNALS_MODE"] == "${SOCIAL_SIGNALS_MODE:-off}"
    assert environment["SOCIAL_INGEST_PROVIDER"] == "${SOCIAL_INGEST_PROVIDER:-disabled}"
    assert not any(key.startswith("SOCIAL_XUI_") for key in environment)
    assert "-Q social_ingestion" in worker["command"]

    services = base["services"]
    consumers = [
        name
        for name, service in services.items()
        if "social_ingestion" in str(service.get("command", ""))
    ]
    assert consumers == ["celery-social"]
    assert all("xui-reader" not in str(service.get("volumes", [])) for service in base["services"].values())


def test_private_overlay_reuses_the_one_worker_and_alone_mounts_profile_writable():
    overlay = _yaml(XUI_COMPOSE)

    assert list(overlay["services"]) == ["celery-social"]
    worker = overlay["services"]["celery-social"]
    assert worker["build"]["target"] == "social-xui"
    assert worker["image"] == "${SOCIAL_WORKER_IMAGE:-stock-screener-social-xui:dev}"
    profile_mounts = [
        volume for volume in worker["volumes"] if "/app/data/xui-reader" in volume
    ]
    assert profile_mounts == ["${XUI_PROFILE_DIR:?Set XUI_PROFILE_DIR}:/app/data/xui-reader"]
    assert not profile_mounts[0].endswith(":ro")
    assert worker["build"]["ssh"] == ["default"]
    assert "github_known_hosts" in worker["build"]["secrets"]
    assert worker["environment"] == {
        "SOCIAL_XUI_CONFIG_PATH": "/app/data/xui-reader/config.toml",
        "SOCIAL_XUI_PROFILE": "${SOCIAL_XUI_PROFILE:-automation}",
    }


def test_dockerfile_keeps_private_install_out_of_default_public_target():
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert dockerfile.startswith("# syntax=docker/dockerfile:1.7\n")
    assert "FROM python:3.11-slim AS runtime-base" in dockerfile
    assert "FROM builder AS social-xui-builder" in dockerfile
    assert "FROM runtime-base AS social-xui" in dockerfile
    assert dockerfile.rstrip().endswith("FROM runtime-base AS public")
    assert "git+ssh://git@github.com/xang1234/xui.git@${XUI_READER_REF}" in dockerfile
    assert 'test "${#XUI_READER_REF}" -eq 40' in dockerfile
    assert '*[!0-9a-fA-F]*' in dockerfile
    assert "--mount=type=ssh,required=true" in dockerfile
    assert "--mount=type=secret,id=github_known_hosts" in dockerfile
    assert "python -m playwright install --with-deps chromium" in dockerfile
    assert "ENV PLAYWRIGHT_BROWSERS_PATH=/opt/stockscanner/playwright" in dockerfile

    public_build = dockerfile.split("FROM builder AS social-xui-builder", 1)[0]
    assert "xui-reader" not in public_build.lower()
    assert re.search(r"FROM runtime-base AS social-xui[\s\S]*USER stockscanner", dockerfile)
    assert not re.search(r"(?:ARG|ENV)\s+.*(?:SSH_PRIVATE|GITHUB_TOKEN|SESSION_TOKEN)", dockerfile, re.I)
    assert not re.search(r"COPY\s+.*(?:\.ssh|xui-reader|config\.toml)", dockerfile, re.I)


def test_docker_context_excludes_private_checkout_credentials_and_profiles():
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert ".ssh" in ignored
    assert "**/.ssh" in ignored
    assert "xui-reader" in ignored
    assert "data/xui-reader" in ignored
