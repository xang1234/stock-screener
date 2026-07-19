"""Regression checks for grouped scan filter feature-flag deployment wiring."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_docker_compose_forwards_grouped_scan_filter_flag() -> None:
    content = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert (
        "FEATURE_GROUPED_SCAN_FILTERS: "
        "${FEATURE_GROUPED_SCAN_FILTERS:-false}"
    ) in content


def test_docker_env_example_documents_grouped_scan_filter_flag() -> None:
    content = (ROOT / ".env.docker.example").read_text(encoding="utf-8")

    assert "FEATURE_GROUPED_SCAN_FILTERS=false" in content


def test_backend_env_example_documents_grouped_scan_filter_flag() -> None:
    content = (ROOT / "backend" / ".env.example").read_text(encoding="utf-8")

    assert "FEATURE_GROUPED_SCAN_FILTERS=false" in content
