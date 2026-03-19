"""Regression tests for universe lifecycle startup migration."""

from __future__ import annotations

import sqlite3

from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from app.db_migrations.universe_lifecycle_migration import (
    _backfill_stock_universe_rows,
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
