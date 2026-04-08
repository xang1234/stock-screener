"""Focused tests for runtime Alembic bootstrap behavior."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from unittest.mock import patch

from app.infra.db.migrations import _alembic_config, _engine_database_url, migrate_database_to_head


def test_engine_database_url_preserves_password_for_alembic_config():
    engine = create_engine("postgresql://stockscanner:secret@localhost/stockscanner")
    database_url = _engine_database_url(engine)
    assert database_url == "postgresql://stockscanner:secret@localhost/stockscanner"
    assert _alembic_config(database_url).get_main_option("sqlalchemy.url") == database_url
    engine.dispose()


def test_migrate_database_to_head_reconciles_existing_schema_without_alembic_version():
    class _Engine:
        url = make_url("postgresql://stockscanner:secret@localhost/stockscanner")

    engine = _Engine()
    config = object()
    calls: list[str] = []

    def _record_reconcile(_engine):
        calls.append("reconcile")

    def _record_stamp(_config, _revision):
        calls.append("stamp")

    def _record_upgrade(_config, _revision):
        calls.append("upgrade")

    with patch("app.infra.db.migrations._has_alembic_version_table", return_value=False), patch(
        "app.infra.db.migrations._has_user_tables", return_value=True
    ), patch("app.infra.db.migrations._alembic_config", return_value=config), patch(
        "app.infra.db.migrations.reconcile_legacy_runtime_schema", side_effect=_record_reconcile
    ) as reconcile, patch(
        "app.infra.db.migrations.command.stamp", side_effect=_record_stamp
    ) as stamp, patch("app.infra.db.migrations.command.upgrade", side_effect=_record_upgrade) as upgrade:
        action = migrate_database_to_head(engine)

    assert action == "reconciled"
    assert calls == ["reconcile", "stamp", "upgrade"]
    reconcile.assert_called_once_with(engine)
    stamp.assert_called_once_with(config, "20260408_0001")
    upgrade.assert_called_once()


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
