"""Focused tests for startup migration runner behavior."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from app.main import (
    run_theme_relationships_migration,
    run_universe_lifecycle_migration,
)


def test_run_theme_relationships_migration_is_fatal_on_error():
    with patch(
        "app.db_migrations.theme_relationships_migration.migrate_theme_relationships",
        side_effect=RuntimeError("migration failed"),
    ):
        with pytest.raises(RuntimeError, match="migration failed"):
            run_theme_relationships_migration()


def test_run_universe_lifecycle_migration_is_fatal_on_error():
    with patch(
        "app.db_migrations.universe_lifecycle_migration.migrate_universe_lifecycle",
        side_effect=RuntimeError("migration failed"),
    ):
        with pytest.raises(RuntimeError, match="migration failed"):
            run_universe_lifecycle_migration()
