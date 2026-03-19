"""Regression tests for universe lifecycle startup migration."""

from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from app.db_migrations.universe_lifecycle_migration import migrate_universe_lifecycle


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
