"""Focused tests for startup migration runner behavior."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from app.main import run_theme_aliases_migration, run_theme_match_decision_migration


def test_run_theme_aliases_migration_is_fatal_on_error():
    with patch(
        "app.db_migrations.theme_aliases_migration.migrate_theme_aliases",
        side_effect=RuntimeError("migration failed"),
    ):
        with pytest.raises(RuntimeError, match="migration failed"):
            run_theme_aliases_migration()


def test_run_theme_match_decision_migration_is_fatal_on_error():
    with patch(
        "app.db_migrations.theme_match_decision_migration.migrate_theme_match_decision",
        side_effect=RuntimeError("migration failed"),
    ):
        with pytest.raises(RuntimeError, match="migration failed"):
            run_theme_match_decision_migration()
