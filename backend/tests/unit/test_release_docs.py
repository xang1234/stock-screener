"""Regression checks for release workflow notes and deployment docs."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_release_workflow_uses_curated_release_notes() -> None:
    content = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert "body_path: .github/release-notes.md" in content
    assert "generate_release_notes: true" not in content


def test_readme_documents_v1_release_compose_flow() -> None:
    content = (ROOT / "README.md").read_text()

    assert "APP_IMAGE_TAG=v1.0.0" in content
    assert "docker-compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.release.yml pull" in content
    assert "docker-compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.release.yml up -d --no-build" in content


def test_install_docker_uses_v1_release_example() -> None:
    content = (ROOT / "docs" / "INSTALL_DOCKER.md").read_text()

    assert "APP_IMAGE_TAG=v1.0.0" in content
    assert "Push a git tag like `v1.0.0`" in content
    assert "**Deploy:** Set `APP_IMAGE_TAG=v1.0.0`" in content


def test_curated_release_notes_capture_v1_capabilities() -> None:
    content = (ROOT / ".github" / "release-notes.md").read_text()

    assert "# Stock Scanner v1.0.0" in content
    assert "multi-market" in content.lower()
    assert "first-run bootstrap" in content.lower()
    assert "GHCR" in content
