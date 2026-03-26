"""Unit tests for the SQLite -> PostgreSQL migration utility."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "migrate_sqlite_to_postgres.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "migrate_sqlite_to_postgres",
    MODULE_PATH,
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
migrate_sqlite_to_postgres = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(migrate_sqlite_to_postgres)


def test_finalize_skipped_rows_raises_by_default():
    with pytest.raises(RuntimeError, match="ui_view_snapshots=6"):
        migrate_sqlite_to_postgres._finalize_skipped_rows(
            {
                "document_chunks": 88,
                "ui_view_snapshots": 6,
            },
            allow_skipped_rows=False,
        )


def test_finalize_skipped_rows_allows_explicit_override():
    migrate_sqlite_to_postgres._finalize_skipped_rows(
        {
            "document_chunks": 88,
            "ui_view_snapshots": 6,
        },
        allow_skipped_rows=True,
    )

