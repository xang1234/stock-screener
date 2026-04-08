"""Focused tests for runtime Alembic bootstrap behavior."""

from __future__ import annotations

from unittest.mock import patch

from app.infra.db.migrations import migrate_database_to_head


def test_migrate_database_to_head_stamps_existing_schema_without_alembic_version():
    class _Engine:
        url = "postgresql://test"

    engine = _Engine()
    with patch("app.infra.db.migrations._has_alembic_version_table", return_value=False), patch(
        "app.infra.db.migrations._has_user_tables", return_value=True
    ), patch("app.infra.db.migrations._alembic_config", return_value=object()), patch(
        "app.infra.db.migrations.command.stamp"
    ) as stamp, patch("app.infra.db.migrations.command.upgrade") as upgrade:
        action = migrate_database_to_head(engine)

    assert action == "stamped"
    stamp.assert_called_once()
    upgrade.assert_not_called()


def test_migrate_database_to_head_upgrades_new_schema():
    class _Engine:
        url = "postgresql://test"

    engine = _Engine()
    with patch("app.infra.db.migrations._has_alembic_version_table", return_value=False), patch(
        "app.infra.db.migrations._has_user_tables", return_value=False
    ), patch("app.infra.db.migrations._alembic_config", return_value=object()), patch(
        "app.infra.db.migrations.command.stamp"
    ) as stamp, patch("app.infra.db.migrations.command.upgrade") as upgrade:
        action = migrate_database_to_head(engine)

    assert action == "upgraded"
    stamp.assert_not_called()
    upgrade.assert_called_once()
