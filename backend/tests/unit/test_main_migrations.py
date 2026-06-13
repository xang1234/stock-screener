"""Focused tests for runtime Alembic bootstrap behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine

from app.infra.db.migrations import _alembic_config, _engine_database_url, migrate_database_to_head


def test_engine_database_url_preserves_password_for_alembic_config():
    engine = create_engine("postgresql://stockscanner:secret@localhost/stockscanner")
    database_url = _engine_database_url(engine)
    assert database_url == "postgresql://stockscanner:secret@localhost/stockscanner"
    assert _alembic_config(database_url).get_main_option("sqlalchemy.url") == database_url
    engine.dispose()


def _stub_engine(dialect_name: str, calls: list[str]):
    """Engine stub whose single connection records advisory lock/unlock.

    Mirrors the runtime flow: engine.connect() yields one connection,
    execution_options() returns the same connection, and all lock SQL
    goes through it.
    """

    def _record_execute(statement, params=None):
        sql = str(statement)
        if "pg_advisory_lock" in sql:
            calls.append("lock")
        elif "pg_advisory_unlock" in sql:
            calls.append("unlock")
        return MagicMock()

    conn = MagicMock()
    conn.dialect = SimpleNamespace(name=dialect_name)
    conn.execute.side_effect = _record_execute
    conn.execution_options.return_value = conn
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)

    return SimpleNamespace(
        url=f"{dialect_name}://stockscanner:secret@localhost/stockscanner",
        connect=MagicMock(return_value=conn),
    )


def test_migrate_database_to_head_reconciles_existing_schema_without_alembic_version():
    calls: list[str] = []
    engine = _stub_engine("postgresql", calls)
    config = object()

    with patch("app.infra.db.migrations._has_alembic_version_table", return_value=False), patch(
        "app.infra.db.migrations._has_user_tables", return_value=True
    ), patch("app.infra.db.migrations._alembic_config", return_value=config), patch(
        "app.infra.db.migrations.reconcile_legacy_runtime_schema",
        side_effect=lambda _engine: calls.append("reconcile"),
    ) as reconcile, patch(
        "app.infra.db.migrations.command.stamp",
        side_effect=lambda _config, _revision: calls.append("stamp"),
    ) as stamp, patch(
        "app.infra.db.migrations.command.upgrade",
        side_effect=lambda _config, _revision: calls.append("upgrade"),
    ) as upgrade:
        action = migrate_database_to_head(engine)

    assert action == "reconciled"
    # Advisory lock must bracket the entire reconcile/stamp/upgrade sequence
    # so concurrent uvicorn workers cannot interleave DDL.
    assert calls == ["lock", "reconcile", "stamp", "upgrade", "unlock"]
    reconcile.assert_called_once_with(engine)
    stamp.assert_called_once_with(config, "20260408_0001")
    upgrade.assert_called_once()


def test_migrate_database_to_head_upgrades_new_schema():
    # Non-Postgres dialect (test harness SQLite) skips the advisory lock path.
    calls: list[str] = []
    engine = _stub_engine("sqlite", calls)
    with patch("app.infra.db.migrations._has_alembic_version_table", return_value=False), patch(
        "app.infra.db.migrations._has_user_tables", return_value=False
    ), patch("app.infra.db.migrations._alembic_config", return_value=object()), patch(
        "app.infra.db.migrations.command.stamp"
    ) as stamp, patch("app.infra.db.migrations.command.upgrade") as upgrade:
        action = migrate_database_to_head(engine)

    assert action == "upgraded"
    assert calls == []
    stamp.assert_not_called()
    upgrade.assert_called_once()


def test_migrate_database_to_head_releases_lock_when_upgrade_fails():
    calls: list[str] = []
    engine = _stub_engine("postgresql", calls)

    with patch("app.infra.db.migrations._has_alembic_version_table", return_value=True), patch(
        "app.infra.db.migrations._has_user_tables", return_value=True
    ), patch("app.infra.db.migrations._alembic_config", return_value=object()), patch(
        "app.infra.db.migrations.command.upgrade", side_effect=RuntimeError("boom")
    ), pytest.raises(RuntimeError):
        migrate_database_to_head(engine)

    assert calls == ["lock", "unlock"]


def test_single_connection_serves_lock_and_inspection():
    """The lock window must not open extra connections (one conn for
    lock + schema inspection; Alembic connects separately)."""
    calls: list[str] = []
    engine = _stub_engine("postgresql", calls)
    seen_conns = []

    with patch(
        "app.infra.db.migrations._has_alembic_version_table",
        side_effect=lambda conn: seen_conns.append(conn) or True,
    ), patch(
        "app.infra.db.migrations._has_user_tables",
        side_effect=lambda conn: seen_conns.append(conn) or True,
    ), patch("app.infra.db.migrations._alembic_config", return_value=object()), patch(
        "app.infra.db.migrations.command.upgrade"
    ):
        migrate_database_to_head(engine)

    engine.connect.assert_called_once()
    lock_conn = engine.connect.return_value
    assert seen_conns == [lock_conn, lock_conn]
