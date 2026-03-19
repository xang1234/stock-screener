"""Idempotent migration for stock universe lifecycle fields and audit tables."""

from __future__ import annotations

import logging

from sqlalchemy import text

logger = logging.getLogger(__name__)


def _get_columns(conn, table_name: str) -> set[str]:
    rows = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
    return {row[1] for row in rows}


def migrate_universe_lifecycle(engine) -> None:
    """Add lifecycle columns to stock_universe and create audit tables."""
    with engine.connect() as conn:
        existing_tables = {
            row[0]
            for row in conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
        }

        if "stock_universe" not in existing_tables:
            logger.info("stock_universe table not present yet; lifecycle migration skipped")
            return

        columns = _get_columns(conn, "stock_universe")

        add_columns = {
            "status": "ALTER TABLE stock_universe ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",
            "status_reason": "ALTER TABLE stock_universe ADD COLUMN status_reason TEXT",
            "first_seen_at": "ALTER TABLE stock_universe ADD COLUMN first_seen_at DATETIME",
            "last_seen_in_source_at": "ALTER TABLE stock_universe ADD COLUMN last_seen_in_source_at DATETIME",
            "deactivated_at": "ALTER TABLE stock_universe ADD COLUMN deactivated_at DATETIME",
            "consecutive_fetch_failures": (
                "ALTER TABLE stock_universe ADD COLUMN consecutive_fetch_failures INTEGER NOT NULL DEFAULT 0"
            ),
            "last_fetch_success_at": "ALTER TABLE stock_universe ADD COLUMN last_fetch_success_at DATETIME",
            "last_fetch_failure_at": "ALTER TABLE stock_universe ADD COLUMN last_fetch_failure_at DATETIME",
        }

        for name, ddl in add_columns.items():
            if name not in columns:
                conn.execute(text(ddl))

        conn.execute(
            text(
                """
                UPDATE stock_universe
                SET status = CASE
                    WHEN COALESCE(is_active, 1) = 1 THEN 'active'
                    ELSE 'inactive_manual'
                END
                WHERE status IS NULL OR TRIM(status) = ''
                """
            )
        )
        conn.execute(
            text(
                """
                UPDATE stock_universe
                SET status_reason = CASE
                    WHEN status = 'active' THEN COALESCE(status_reason, 'Existing active symbol')
                    ELSE COALESCE(status_reason, 'Backfilled from legacy inactive flag')
                END
                WHERE status_reason IS NULL OR TRIM(status_reason) = ''
                """
            )
        )
        conn.execute(
            text(
                """
                UPDATE stock_universe
                SET first_seen_at = COALESCE(first_seen_at, added_at, updated_at, CURRENT_TIMESTAMP)
                WHERE first_seen_at IS NULL
                """
            )
        )
        conn.execute(
            text(
                """
                UPDATE stock_universe
                SET last_seen_in_source_at = COALESCE(last_seen_in_source_at, updated_at, added_at, CURRENT_TIMESTAMP)
                WHERE status = 'active' AND last_seen_in_source_at IS NULL
                """
            )
        )
        conn.execute(
            text(
                """
                UPDATE stock_universe
                SET deactivated_at = COALESCE(deactivated_at, updated_at, CURRENT_TIMESTAMP)
                WHERE status <> 'active' AND deactivated_at IS NULL
                """
            )
        )
        conn.execute(
            text(
                """
                UPDATE stock_universe
                SET is_active = CASE WHEN status = 'active' THEN 1 ELSE 0 END
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS stock_universe_status_events (
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
                CREATE INDEX IF NOT EXISTS idx_stock_universe_status_events_symbol_created
                ON stock_universe_status_events(symbol, created_at)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_stock_universe_status_events_status_created
                ON stock_universe_status_events(new_status, created_at)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_stock_universe_exchange_status
                ON stock_universe(exchange, status)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_stock_universe_status_active
                ON stock_universe(status, is_active)
                """
            )
        )
        conn.commit()

    logger.info("Universe lifecycle migration completed")
