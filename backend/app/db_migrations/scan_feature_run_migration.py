"""
Idempotent migration: add feature_run_id FK column to scans table.

Binds each scan to the Feature Store snapshot (feature_run) that was
the latest published run at scan creation time.  Legacy scans keep
feature_run_id = NULL — no backfill.

Safe to run on every startup — checks prerequisites and skips if present.
"""
import logging
from sqlalchemy import text

logger = logging.getLogger(__name__)


def migrate_scan_feature_run_id(engine) -> None:
    """
    Add feature_run_id column + index to the scans table.

    Prerequisites:
      - The ``feature_runs`` table must already exist (created by
        ``feature_store_migration``).  If it doesn't, the migration
        is silently skipped so that startup isn't blocked.

    Idempotent: skips if the column already exists.
    """
    with engine.connect() as conn:
        # Prerequisite: feature_runs table must exist for FK
        tables = {
            r[0]
            for r in conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
        }
        if "feature_runs" not in tables:
            logger.warning(
                "feature_runs table not found — skipping scan.feature_run_id migration"
            )
            return

        # Detect existing columns
        existing = {
            r[1] for r in conn.execute(text("PRAGMA table_info(scans)")).fetchall()
        }
        if "feature_run_id" in existing:
            return  # Already migrated

        # Add column + index
        conn.execute(
            text(
                "ALTER TABLE scans ADD COLUMN feature_run_id INTEGER "
                "REFERENCES feature_runs(id)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_scans_feature_run_id "
                "ON scans(feature_run_id)"
            )
        )
        conn.commit()
        logger.info("Added scans.feature_run_id column with index")
