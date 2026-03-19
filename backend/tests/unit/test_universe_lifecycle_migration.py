"""Regression tests for universe lifecycle startup migration."""

from __future__ import annotations

import sqlite3

from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from app.db_migrations.universe_lifecycle_migration import (
    _backfill_stock_universe_rows,
    _ensure_index,
    _get_columns,
    migrate_universe_lifecycle,
)


def _make_legacy_engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE stock_universe (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL UNIQUE,
                    exchange TEXT,
                    is_active BOOLEAN,
                    added_at DATETIME,
                    updated_at DATETIME
                )
                """
            )
        )
    return engine


def test_migration_backfills_status_fields_from_legacy_is_active():
    engine = _make_legacy_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO stock_universe(symbol, exchange, is_active, added_at, updated_at)
                VALUES
                    ('AAPL', 'NASDAQ', 1, '2025-01-01 00:00:00', '2025-01-02 00:00:00'),
                    ('OLD', 'NYSE', 0, '2024-01-01 00:00:00', '2024-02-01 00:00:00')
                """
            )
        )

    migrate_universe_lifecycle(engine)

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT symbol, is_active, status, status_reason, first_seen_at,
                       last_seen_in_source_at, deactivated_at
                FROM stock_universe
                ORDER BY symbol
                """
            )
        ).mappings().all()
        events_table = conn.execute(
            text(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table' AND name='stock_universe_status_events'
                """
            )
        ).scalar()

    assert events_table == "stock_universe_status_events"
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["is_active"] == 1
    assert rows[0]["status"] == "active"
    assert rows[0]["status_reason"] == "Existing active symbol"
    assert rows[0]["first_seen_at"] == "2025-01-01 00:00:00"
    assert rows[0]["last_seen_in_source_at"] == "2025-01-02 00:00:00"
    assert rows[0]["deactivated_at"] is None

    assert rows[1]["symbol"] == "OLD"
    assert rows[1]["is_active"] == 0
    assert rows[1]["status"] == "inactive_manual"
    assert rows[1]["status_reason"] == "Backfilled from legacy inactive flag"
    assert rows[1]["first_seen_at"] == "2024-01-01 00:00:00"
    assert rows[1]["last_seen_in_source_at"] is None
    assert rows[1]["deactivated_at"] == "2024-02-01 00:00:00"


def test_migration_normalizes_blank_status_values_without_sql_trim():
    engine = _make_legacy_engine()
    migrate_universe_lifecycle(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO stock_universe(
                    symbol, exchange, is_active, added_at, updated_at,
                    status, status_reason, first_seen_at, last_seen_in_source_at, deactivated_at
                )
                VALUES (
                    'MSFT', 'NASDAQ', 1, '2025-03-01 00:00:00', '2025-03-02 00:00:00',
                    '   ', '   ', NULL, NULL, NULL
                )
                """
            )
        )

    migrate_universe_lifecycle(engine)

    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT is_active, status, status_reason, first_seen_at, last_seen_in_source_at, deactivated_at
                FROM stock_universe
                WHERE symbol = 'MSFT'
                """
            )
        ).mappings().one()

    assert row["is_active"] == 1
    assert row["status"] == "active"
    assert row["status_reason"] == "Existing active symbol"
    assert row["first_seen_at"] == "2025-03-01 00:00:00"
    assert row["last_seen_in_source_at"] == "2025-03-02 00:00:00"
    assert row["deactivated_at"] is None


def test_migration_preserves_inactive_last_seen_in_source_timestamp():
    engine = _make_legacy_engine()
    migrate_universe_lifecycle(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO stock_universe(
                    symbol, exchange, is_active, added_at, updated_at,
                    status, status_reason, first_seen_at, last_seen_in_source_at, deactivated_at
                )
                VALUES (
                    'DEAD', 'NASDAQ', 0, '2024-03-01 00:00:00', '2024-06-01 00:00:00',
                    'inactive_missing_source', 'Missing from Finviz universe sync',
                    '2024-03-01 00:00:00', '2024-05-15 00:00:00', '2024-06-01 00:00:00'
                )
                """
            )
        )

    migrate_universe_lifecycle(engine)

    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT is_active, status, status_reason, first_seen_at, last_seen_in_source_at, deactivated_at
                FROM stock_universe
                WHERE symbol = 'DEAD'
                """
            )
        ).mappings().one()

    assert row["is_active"] == 0
    assert row["status"] == "inactive_missing_source"
    assert row["status_reason"] == "Missing from Finviz universe sync"
    assert row["first_seen_at"] == "2024-03-01 00:00:00"
    assert row["last_seen_in_source_at"] == "2024-05-15 00:00:00"
    assert row["deactivated_at"] == "2024-06-01 00:00:00"


def test_backfill_is_noop_for_already_migrated_rows():
    engine = _make_legacy_engine()
    migrate_universe_lifecycle(engine)

    with engine.connect() as raw_conn:
        columns = _get_columns(raw_conn, "stock_universe")
        raw_conn.execute(
            text(
                """
                INSERT INTO stock_universe(
                    symbol, exchange, is_active, added_at, updated_at,
                    status, status_reason, first_seen_at, last_seen_in_source_at, deactivated_at
                )
                VALUES (
                    'AAPL', 'NASDAQ', 1, '2025-01-01 00:00:00', '2025-01-02 00:00:00',
                    'active', 'Existing active symbol',
                    '2025-01-01 00:00:00', '2025-01-02 00:00:00', NULL
                )
                """
            )
        )
        raw_conn.commit()

        class RecordingConnection:
            def __init__(self, conn):
                self.conn = conn
                self.update_calls = 0

            def begin_nested(self):
                return self.conn.begin_nested()

            def execute(self, statement, params=None):
                sql = str(statement)
                if sql.lstrip().startswith("UPDATE stock_universe"):
                    self.update_calls += 1
                if params is None:
                    return self.conn.execute(statement)
                return self.conn.execute(statement, params)

        recording_conn = RecordingConnection(raw_conn)
        _backfill_stock_universe_rows(recording_conn, columns)

        assert recording_conn.update_calls == 0


def test_backfill_retries_row_by_row_and_skips_corrupt_rows():
    engine = _make_legacy_engine()
    migrate_universe_lifecycle(engine)

    with engine.connect() as raw_conn:
        columns = _get_columns(raw_conn, "stock_universe")
        raw_conn.execute(
            text(
                """
                INSERT INTO stock_universe(
                    id, symbol, exchange, is_active, added_at, updated_at,
                    status, status_reason, first_seen_at, last_seen_in_source_at, deactivated_at
                )
                VALUES
                    (101, 'BAD', 'NASDAQ', 1, '2025-01-01 00:00:00', '2025-01-02 00:00:00',
                     '   ', NULL, NULL, NULL, NULL),
                    (102, 'GOOD', 'NASDAQ', 0, '2024-01-01 00:00:00', '2024-02-01 00:00:00',
                     '   ', NULL, NULL, NULL, NULL)
                """
            )
        )
        raw_conn.commit()

        class CorruptingConnection:
            def __init__(self, conn):
                self.conn = conn
                self.batch_calls = 0

            def begin_nested(self):
                return self.conn.begin_nested()

            def execute(self, statement, params=None):
                sql = str(statement)
                if sql.lstrip().startswith("UPDATE stock_universe"):
                    if isinstance(params, list):
                        self.batch_calls += 1
                        raise sqlite3.DatabaseError("database disk image is malformed")
                    if params["id"] == 101:
                        raise sqlite3.DatabaseError("database disk image is malformed")
                if params is None:
                    return self.conn.execute(statement)
                return self.conn.execute(statement, params)

        corrupting_conn = CorruptingConnection(raw_conn)
        _backfill_stock_universe_rows(corrupting_conn, columns)

        rows = raw_conn.execute(
            text(
                """
                SELECT id, status, status_reason, first_seen_at, last_seen_in_source_at, deactivated_at, is_active
                FROM stock_universe
                WHERE id IN (101, 102)
                ORDER BY id
                """
            )
        ).mappings().all()

    assert corrupting_conn.batch_calls == 1
    assert rows[0]["id"] == 101
    assert rows[0]["status"] == "   "
    assert rows[0]["status_reason"] is None
    assert rows[0]["is_active"] == 1

    assert rows[1]["id"] == 102
    assert rows[1]["status"] == "inactive_manual"
    assert rows[1]["status_reason"] == "Backfilled from legacy inactive flag"
    assert rows[1]["first_seen_at"] == "2024-01-01 00:00:00"
    assert rows[1]["last_seen_in_source_at"] is None
    assert rows[1]["deactivated_at"] == "2024-02-01 00:00:00"
    assert rows[1]["is_active"] == 0


def test_backfill_skips_metadata_only_updates_when_lifecycle_columns_already_exist():
    engine = _make_legacy_engine()
    migrate_universe_lifecycle(engine)

    with engine.connect() as raw_conn:
        columns = _get_columns(raw_conn, "stock_universe")
        raw_conn.execute(
            text(
                """
                INSERT INTO stock_universe(
                    symbol, exchange, is_active, added_at, updated_at,
                    status, status_reason, first_seen_at, last_seen_in_source_at, deactivated_at
                )
                VALUES (
                    'AAPL', 'NASDAQ', 1, '2025-01-01 00:00:00', '2025-01-02 00:00:00',
                    'active', NULL, NULL, NULL, NULL
                )
                """
            )
        )
        raw_conn.commit()

        class RecordingConnection:
            def __init__(self, conn):
                self.conn = conn
                self.update_calls = 0

            def begin_nested(self):
                return self.conn.begin_nested()

            def execute(self, statement, params=None):
                sql = str(statement)
                if sql.lstrip().startswith("UPDATE stock_universe"):
                    self.update_calls += 1
                if params is None:
                    return self.conn.execute(statement)
                return self.conn.execute(statement, params)

        recording_conn = RecordingConnection(raw_conn)
        _backfill_stock_universe_rows(recording_conn, columns)

        row = raw_conn.execute(
            text(
                """
                SELECT status, status_reason, first_seen_at, last_seen_in_source_at, deactivated_at, is_active
                FROM stock_universe
                WHERE symbol = 'AAPL'
                """
            )
        ).mappings().one()

    assert recording_conn.update_calls == 0
    assert row["status"] == "active"
    assert row["status_reason"] is None
    assert row["first_seen_at"] is None
    assert row["last_seen_in_source_at"] is None
    assert row["deactivated_at"] is None
    assert row["is_active"] == 1


def test_migration_skips_row_rewrites_after_partial_column_add():
    engine = _make_legacy_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO stock_universe(symbol, exchange, is_active, added_at, updated_at)
                VALUES ('OLD', 'NYSE', 0, '2024-01-01 00:00:00', '2024-02-01 00:00:00')
                """
            )
        )
        conn.execute(
            text(
                "ALTER TABLE stock_universe ADD COLUMN status TEXT NOT NULL DEFAULT 'active'"
            )
        )
        conn.execute(text("ALTER TABLE stock_universe ADD COLUMN status_reason TEXT"))
        conn.execute(text("ALTER TABLE stock_universe ADD COLUMN first_seen_at DATETIME"))
        conn.execute(text("ALTER TABLE stock_universe ADD COLUMN last_seen_in_source_at DATETIME"))
        conn.execute(text("ALTER TABLE stock_universe ADD COLUMN deactivated_at DATETIME"))
        conn.execute(
            text(
                """
                ALTER TABLE stock_universe
                ADD COLUMN consecutive_fetch_failures INTEGER NOT NULL DEFAULT 0
                """
            )
        )
        conn.execute(text("ALTER TABLE stock_universe ADD COLUMN last_fetch_success_at DATETIME"))
        conn.execute(text("ALTER TABLE stock_universe ADD COLUMN last_fetch_failure_at DATETIME"))

    migrate_universe_lifecycle(engine)

    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT is_active, status, status_reason, first_seen_at, deactivated_at
                FROM stock_universe
                WHERE symbol = 'OLD'
                """
            )
        ).mappings().one()

    assert row["is_active"] == 0
    assert row["status"] == "active"
    assert row["status_reason"] is None
    assert row["first_seen_at"] is None
    assert row["deactivated_at"] is None


def test_migration_reuses_equivalent_existing_status_event_indexes():
    engine = _make_legacy_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE stock_universe_status_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    old_status TEXT,
                    new_status TEXT NOT NULL,
                    trigger_source TEXT NOT NULL,
                    reason TEXT,
                    payload_json TEXT,
                    created_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX idx_universe_status_events_symbol_created
                ON stock_universe_status_events(symbol, created_at)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX idx_universe_status_events_status_created
                ON stock_universe_status_events(new_status, created_at)
                """
            )
        )

    migrate_universe_lifecycle(engine)

    with engine.connect() as conn:
        indexes = {
            row[0]
            for row in conn.execute(
                text(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'index' AND tbl_name = 'stock_universe_status_events'
                    """
                )
            ).fetchall()
        }

    assert "idx_universe_status_events_symbol_created" in indexes
    assert "idx_universe_status_events_status_created" in indexes
    assert "idx_stock_universe_status_events_symbol_created" not in indexes
    assert "idx_stock_universe_status_events_status_created" not in indexes


def test_ensure_index_skips_optional_index_creation_on_corruption():
    engine = _make_legacy_engine()

    with engine.connect() as raw_conn:
        class CorruptingConnection:
            def __init__(self, conn):
                self.conn = conn

            def execute(self, statement, params=None):
                sql = str(statement)
                if "CREATE INDEX IF NOT EXISTS idx_test_corrupt" in sql:
                    raise sqlite3.DatabaseError("database disk image is malformed")
                if params is None:
                    return self.conn.execute(statement)
                return self.conn.execute(statement, params)

        corrupting_conn = CorruptingConnection(raw_conn)
        _ensure_index(
            corrupting_conn,
            table_name="stock_universe",
            index_name="idx_test_corrupt",
            columns=("exchange",),
            ddl="""
                CREATE INDEX IF NOT EXISTS idx_test_corrupt
                ON stock_universe(exchange)
            """,
        )

        indexes = {
            row[1]
            for row in raw_conn.execute(
                text("PRAGMA index_list('stock_universe')")
            ).fetchall()
        }

    assert "idx_test_corrupt" not in indexes
