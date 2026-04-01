"""Shared helpers for CLI scripts that need the app runtime."""

from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.infra.db.portability import is_sqlite
from app.main import initialize_runtime


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_database_path_ready() -> None:
    if not is_sqlite(settings.database_url):
        return

    database_path = Path(settings.database_url.removeprefix("sqlite:///"))
    database_path.parent.mkdir(parents=True, exist_ok=True)


def prepare_runtime() -> None:
    ensure_database_path_ready()
    initialize_runtime()
