"""Shared helpers for CLI scripts that need the app runtime."""

from __future__ import annotations

from pathlib import Path

from app.main import initialize_runtime


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def prepare_runtime() -> None:
    initialize_runtime()
