"""Idempotent migration for stock universe lifecycle fields and audit tables."""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import text

logger = logging.getLogger(__name__)


def _get_columns(conn, table_name: str) -> set[str]:
    rows = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
    return {row[1] for row in rows}


def _normalize_text(value) -> str | None:
    """Normalize legacy text values without using SQLite string functions."""
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _build_select_expr(columns: set[str], column_name: str) -> str:
    if column_name in columns:
        return column_name
    return f"NULL AS {column_name}"


def _backfill_stock_universe_rows(conn, columns: set[str]) -> None:
    """Backfill lifecycle data row-by-row to avoid brittle full-table text scans."""
    select_columns = [
        "id",
        "COALESCE(is_active, 1) AS legacy_is_active",
        _build_select_expr(columns, "status"),
        _build_select_expr(columns, "status_reason"),
        _build_select_expr(columns, "first_seen_at"),
        _build_select_expr(columns, "last_seen_in_source_at"),
        _build_select_expr(columns, "deactivated_at"),
        _build_select_expr(columns, "added_at"),
        _build_select_expr(columns, "updated_at"),
    ]
    rows = conn.execute(
        text(f"SELECT {', '.join(select_columns)} FROM stock_universe")
    ).mappings().all()

    now = datetime.utcnow()
    updates: list[dict[str, object]] = []
    for row in rows:
        status = _normalize_text(row["status"])
        if status is None:
            status = "active" if row["legacy_is_active"] else "inactive_manual"

        status_reason = _normalize_text(row["status_reason"])
        if status_reason is None:
            status_reason = (
                "Existing active symbol"
                if status == "active"
                else "Backfilled from legacy inactive flag"
            )

        first_seen_at = row["first_seen_at"] or row["added_at"] or row["updated_at"] or now
        last_seen_in_source_at = row["last_seen_in_source_at"]
        if status == "active" and last_seen_in_source_at is None:
            last_seen_in_source_at = row["updated_at"] or row["added_at"] or now

        deactivated_at = row["deactivated_at"]
        if status != "active" and deactivated_at is None:
            deactivated_at = row["updated_at"] or now
        elif status == "active":
            deactivated_at = None

        updates.append(
            {
                "id": row["id"],
                "status": status,
                "status_reason": status_reason,
                "first_seen_at": first_seen_at,
                "last_seen_in_source_at": last_seen_in_source_at,
                "deactivated_at": deactivated_at,
                "is_active": 1 if status == "active" else 0,
            }
        )

    if not updates:
        return

    conn.execute(
        text(
            """
            UPDATE stock_universe
            SET status = :status,
                status_reason = :status_reason,
                first_seen_at = :first_seen_at,
                last_seen_in_source_at = :last_seen_in_source_at,
                deactivated_at = :deactivated_at,
                is_active = :is_active
            WHERE id = :id
            """
        ),
        updates,
    )


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

        legacy_columns = _get_columns(conn, "stock_universe")

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
            if name not in legacy_columns:
                conn.execute(text(ddl))

        _backfill_stock_universe_rows(conn, legacy_columns)

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
