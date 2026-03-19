"""Focused tests for startup migration runner behavior."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from app.main import (
    run_provider_snapshot_migration,
    run_theme_aliases_migration,
    run_theme_match_decision_migration,
    run_theme_pipeline_run_migration,
    run_theme_relationships_migration,
    run_ui_view_snapshot_migration,
    run_universe_lifecycle_migration,
)


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


def test_run_theme_relationships_migration_is_fatal_on_error():
    with patch(
        "app.db_migrations.theme_relationships_migration.migrate_theme_relationships",
        side_effect=RuntimeError("migration failed"),
    ):
        with pytest.raises(RuntimeError, match="migration failed"):
            run_theme_relationships_migration()


def test_run_theme_pipeline_run_migration_is_fatal_on_error():
    with patch(
        "app.db_migrations.theme_pipeline_run_migration.migrate_theme_pipeline_run_schema",
        side_effect=RuntimeError("migration failed"),
    ):
        with pytest.raises(RuntimeError, match="migration failed"):
            run_theme_pipeline_run_migration()


def test_run_ui_view_snapshot_migration_is_fatal_on_error():
    with patch(
        "app.db_migrations.ui_view_snapshot_migration.migrate_ui_view_snapshot_tables",
        side_effect=RuntimeError("migration failed"),
    ):
        with pytest.raises(RuntimeError, match="migration failed"):
            run_ui_view_snapshot_migration()


def test_run_universe_lifecycle_migration_is_fatal_on_error():
    with patch(
        "app.db_migrations.universe_lifecycle_migration.migrate_universe_lifecycle",
        side_effect=RuntimeError("migration failed"),
    ):
        with pytest.raises(RuntimeError, match="migration failed"):
            run_universe_lifecycle_migration()


def test_run_provider_snapshot_migration_is_fatal_on_error():
    with patch(
        "app.db_migrations.provider_snapshot_migration.migrate_provider_snapshot_tables",
        side_effect=RuntimeError("migration failed"),
    ):
        with pytest.raises(RuntimeError, match="migration failed"):
            run_provider_snapshot_migration()
