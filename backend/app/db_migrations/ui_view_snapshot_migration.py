"""Idempotent migration for UI bootstrap snapshot tables."""

from __future__ import annotations

import logging

from sqlalchemy import text

logger = logging.getLogger(__name__)


def migrate_ui_view_snapshot_tables(engine) -> None:
    """Create UI snapshot tables and indexes idempotently."""
    with engine.connect() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS ui_view_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    view_key TEXT NOT NULL,
                    variant_key TEXT NOT NULL,
                    source_revision TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    published_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP),
                    created_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP),
                    CONSTRAINT uq_ui_view_snapshots_revision
                        UNIQUE (view_key, variant_key, source_revision)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS ui_view_snapshot_pointers (
                    view_key TEXT NOT NULL,
                    variant_key TEXT NOT NULL,
                    snapshot_id INTEGER NOT NULL REFERENCES ui_view_snapshots(id) ON DELETE CASCADE,
                    updated_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP),
                    PRIMARY KEY (view_key, variant_key)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_ui_view_snapshots_view_variant
                ON ui_view_snapshots(view_key, variant_key, published_at DESC)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_ui_view_snapshots_source_revision
                ON ui_view_snapshots(source_revision)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_ui_view_snapshot_pointers_snapshot_id
                ON ui_view_snapshot_pointers(snapshot_id)
                """
            )
        )
        conn.commit()

    logger.info("UI view snapshot migration completed")
