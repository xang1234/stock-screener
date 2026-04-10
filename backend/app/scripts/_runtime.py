"""Shared helpers for CLI scripts that need the app runtime."""

from __future__ import annotations

from pathlib import Path

from app.database import SessionLocal
from app.main import initialize_runtime
from app.wiring.bootstrap import build_runtime_services, set_runtime_services


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def prepare_runtime() -> None:
    initialize_runtime()
    runtime_services = build_runtime_services(session_factory=SessionLocal)
    set_runtime_services(runtime_services)
