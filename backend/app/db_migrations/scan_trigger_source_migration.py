"""
Idempotent migration: add trigger_source column to scans table.

Backfills legacy rows to ``manual`` so older scans render with the new
labeling scheme without requiring destructive data changes.
"""

from __future__ import annotations

import logging

from sqlalchemy import text

from ..infra.db.portability import column_names

logger = logging.getLogger(__name__)


def migrate_scan_trigger_source(engine) -> None:
    """Add ``trigger_source`` to scans and backfill existing rows."""
    with engine.connect() as conn:
        existing = column_names(conn, "scans")
        if "trigger_source" not in existing:
            conn.execute(
                text(
                    "ALTER TABLE scans ADD COLUMN trigger_source "
                    "VARCHAR(20) NOT NULL DEFAULT 'manual'"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_scans_trigger_source "
                    "ON scans(trigger_source)"
                )
            )
            conn.commit()
            logger.info("Added scans.trigger_source column with default 'manual'")

        updated = conn.execute(
            text(
                "UPDATE scans SET trigger_source = 'manual' "
                "WHERE trigger_source IS NULL OR trigger_source = ''"
            )
        ).rowcount or 0
        if updated:
            conn.commit()
            logger.info("Backfilled %d scans rows with trigger_source='manual'", updated)
