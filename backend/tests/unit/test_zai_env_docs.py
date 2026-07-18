"""Regression checks for Z.AI env examples and Compose propagation."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_backend_env_example_documents_zai_settings() -> None:
    content = (ROOT / "backend" / ".env.example").read_text()

    assert "ZAI_API_KEY=" in content
    assert "ZAI_API_KEYS=" in content
    assert "ZAI_API_BASE=https://api.z.ai/api/paas/v4" in content


def test_docker_env_example_documents_zai_settings() -> None:
    content = (ROOT / ".env.docker.example").read_text()

    assert "ZAI_API_KEY=" in content
    assert "ZAI_API_KEYS=" in content
    assert "ZAI_API_BASE=https://api.z.ai/api/paas/v4" in content
    assert "not from backend/.env" in content


def test_docker_compose_forwards_zai_settings_to_backend_and_workers() -> None:
    content = (ROOT / "docker-compose.yml").read_text()
    app_env = content.split("x-app-env: &app-env", maxsplit=1)[1].split(
        "x-celery-env: &celery-env", maxsplit=1
    )[0]
    celery_env = content.split("x-celery-env: &celery-env", maxsplit=1)[1].split(
        "x-worker-common: &worker-common", maxsplit=1
    )[0]
    worker_common = content.split("x-worker-common: &worker-common", maxsplit=1)[
        1
    ].split("services:", maxsplit=1)[0]
    backend_service = content.split("  backend:", maxsplit=1)[1].split(
        "  celery-general:", maxsplit=1
    )[0]

    assert "ZAI_API_KEY: ${ZAI_API_KEY:-}" in app_env
    assert "ZAI_API_KEYS: ${ZAI_API_KEYS:-}" in app_env
    assert "ZAI_API_BASE: ${ZAI_API_BASE:-https://api.z.ai/api/paas/v4}" in app_env
    assert "<<: *app-env" in celery_env
    assert "<<: *celery-env" in worker_common
    assert "<<: *app-env" in backend_service
