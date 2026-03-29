"""Regression checks for Z.AI env examples and Compose propagation."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_backend_env_example_documents_zai_settings() -> None:
    content = (ROOT / "backend" / ".env.example").read_text()

    assert "ZAI_API_KEY=" in content
    assert "ZAI_API_BASE=https://api.z.ai/api/paas/v4" in content


def test_docker_env_example_documents_zai_settings() -> None:
    content = (ROOT / ".env.docker.example").read_text()

    assert "ZAI_API_KEY=" in content
    assert "ZAI_API_BASE=https://api.z.ai/api/paas/v4" in content
    assert "not from backend/.env" in content


def test_docker_compose_forwards_zai_settings_to_theme_services() -> None:
    content = (ROOT / "docker-compose.yml").read_text()

    assert content.count("ZAI_API_KEY=${ZAI_API_KEY:-}") >= 4
    assert content.count("ZAI_API_BASE=${ZAI_API_BASE:-https://api.z.ai/api/paas/v4}") >= 4
