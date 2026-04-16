"""Regression tests for backend pytest database bootstrap behavior."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]


def _run_bootstrap(database_url: str | None) -> tuple[int, str]:
    env = os.environ.copy()
    if database_url is None:
        env.pop("DATABASE_URL", None)
    else:
        env["DATABASE_URL"] = database_url
    env.pop("STOCKSCANNER_TEST_ALLOW_SQLITE", None)

    result = subprocess.run(
        [sys.executable, "-c", "import tests.conftest as c; print(c.engine.url)"],
        cwd=BACKEND_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout.strip()


def test_backend_test_bootstrap_defaults_to_sqlite_when_database_url_missing():
    code, stdout = _run_bootstrap(None)

    assert code == 0
    assert stdout == "sqlite://"


def test_backend_test_bootstrap_preserves_explicit_postgres_database_url():
    code, stdout = _run_bootstrap("postgresql://user:pass@localhost/testdb")

    assert code == 0
    assert stdout == "postgresql://user:***@localhost/testdb"
