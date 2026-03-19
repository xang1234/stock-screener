"""Idempotent migration for provider snapshot tables and snapshot hydration columns."""

from __future__ import annotations

import logging

from sqlalchemy import text

logger = logging.getLogger(__name__)


def _get_columns(conn, table_name: str) -> set[str]:
    rows = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
    return {row[1] for row in rows}


def migrate_provider_snapshot_tables(engine) -> None:
    """Create provider snapshot tables and add stock_fundamentals provenance columns."""
    with engine.connect() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS provider_snapshot_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_key TEXT NOT NULL,
                    run_mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_revision TEXT NOT NULL,
                    coverage_stats_json TEXT,
                    parity_stats_json TEXT,
                    warnings_json TEXT,
                    symbols_total INTEGER NOT NULL DEFAULT 0,
                    symbols_published INTEGER NOT NULL DEFAULT 0,
                    created_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP),
                    published_at DATETIME,
                    CONSTRAINT uq_provider_snapshot_revision
                        UNIQUE (snapshot_key, source_revision)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS provider_snapshot_rows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL REFERENCES provider_snapshot_runs(id) ON DELETE CASCADE,
                    symbol TEXT NOT NULL,
                    exchange TEXT,
                    row_hash TEXT NOT NULL,
                    normalized_payload_json TEXT NOT NULL,
                    raw_payload_json TEXT,
                    created_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP),
                    CONSTRAINT uq_provider_snapshot_row_run_symbol
                        UNIQUE (run_id, symbol)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS provider_snapshot_pointers (
                    snapshot_key TEXT PRIMARY KEY,
                    run_id INTEGER NOT NULL REFERENCES provider_snapshot_runs(id) ON DELETE CASCADE,
                    updated_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_provider_snapshot_runs_key_created
                ON provider_snapshot_runs(snapshot_key, created_at)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_provider_snapshot_runs_key_status
                ON provider_snapshot_runs(snapshot_key, status)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_provider_snapshot_rows_run_exchange
                ON provider_snapshot_rows(run_id, exchange)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_provider_snapshot_rows_symbol
                ON provider_snapshot_rows(symbol)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_provider_snapshot_pointers_run_id
                ON provider_snapshot_pointers(run_id)
                """
            )
        )

        tables = {
            row[0]
            for row in conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
        }
        if "stock_fundamentals" in tables:
            columns = _get_columns(conn, "stock_fundamentals")
            add_columns = {
                "finviz_snapshot_revision": (
                    "ALTER TABLE stock_fundamentals ADD COLUMN finviz_snapshot_revision TEXT"
                ),
                "finviz_snapshot_at": (
                    "ALTER TABLE stock_fundamentals ADD COLUMN finviz_snapshot_at DATETIME"
                ),
                "yahoo_profile_refreshed_at": (
                    "ALTER TABLE stock_fundamentals ADD COLUMN yahoo_profile_refreshed_at DATETIME"
                ),
                "yahoo_statements_refreshed_at": (
                    "ALTER TABLE stock_fundamentals ADD COLUMN yahoo_statements_refreshed_at DATETIME"
                ),
                "technicals_refreshed_at": (
                    "ALTER TABLE stock_fundamentals ADD COLUMN technicals_refreshed_at DATETIME"
                ),
            }
            for name, ddl in add_columns.items():
                if name not in columns:
                    conn.execute(text(ddl))

            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_stock_fundamentals_snapshot_revision
                    ON stock_fundamentals(finviz_snapshot_revision)
                    """
                )
            )

        conn.commit()

    logger.info("Provider snapshot migration completed")
