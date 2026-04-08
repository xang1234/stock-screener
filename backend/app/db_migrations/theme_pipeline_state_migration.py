"""
Idempotent migration: create content_item_pipeline_state table and indexes.

This table is the per-pipeline processing state source of truth for theme
extraction orchestration. It avoids cross-pipeline interference caused by
global content_items processing flags.
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text

from ..infra.db.portability import (
    check_constraints,
    column_names,
    foreign_keys,
    index_names,
    table_names,
)
from ..models.theme import ContentItemPipelineState

logger = logging.getLogger(__name__)

TABLE_NAME = "content_item_pipeline_state"

REQUIRED_COLUMNS = {
    "id",
    "content_item_id",
    "pipeline",
    "status",
    "attempt_count",
    "error_code",
    "error_message",
    "last_attempt_at",
    "processed_at",
    "created_at",
    "updated_at",
}

STATUS_VALUES = (
    "pending",
    "in_progress",
    "processed",
    "failed_retryable",
    "failed_terminal",
)

PIPELINE_VALUES = (
    "technical",
    "fundamental",
)

EXPECTED_INDEXES = {
    "uix_cips_content_item_pipeline",
    "idx_cips_pipeline_status_last_attempt",
    "idx_cips_pipeline_status_created",
    "idx_cips_content_item_pipeline_status",
    "idx_cips_error_code",
    "idx_cips_updated_at",
}


def migrate_theme_pipeline_state(engine) -> dict[str, Any]:
    """
    Create/patch pipeline-state schema idempotently.

    Returns:
        Migration stats with created table/index information.
    """
    stats: dict[str, Any] = {
        "table_created": False,
        "table_rebuilt": False,
        "columns_added": [],
        "indexes_ensured": [],
    }

    with engine.connect() as conn:
        existing_tables = _get_existing_tables(conn)
        if TABLE_NAME not in existing_tables:
            _create_pipeline_state_table(conn)
            stats["table_created"] = True
            logger.info("Created table %s", TABLE_NAME)
        else:
            if _needs_table_rebuild(conn):
                _rebuild_pipeline_state_table(conn)
                stats["table_rebuilt"] = True
                logger.info("Rebuilt table %s to enforce FK/check constraints", TABLE_NAME)
            else:
                added = _add_missing_columns(conn)
                stats["columns_added"] = added
                if added:
                    logger.info("Added missing columns to %s: %s", TABLE_NAME, ", ".join(added))

        ensured = _ensure_indexes(conn)
        stats["indexes_ensured"] = ensured
        conn.commit()

    logger.info("Theme pipeline-state migration completed: %s", stats)
    return stats


def verify_theme_pipeline_state_schema(engine) -> dict[str, Any]:
    """
    Verify post-migration schema integrity and operator-facing sanity checks.
    """
    with engine.connect() as conn:
        tables = _get_existing_tables(conn)
        table_exists = TABLE_NAME in tables
        columns = _get_table_columns(conn) if table_exists else set()
        missing_columns = sorted(REQUIRED_COLUMNS - columns)
        indexes = _get_table_indexes(conn) if table_exists else set()
        missing_indexes = sorted(EXPECTED_INDEXES - indexes)

        constraints = check_constraints(conn, TABLE_NAME) if table_exists else []
        status_check_present = any("status" in (constraint.get("sqltext") or "") for constraint in constraints)
        pipeline_check_present = any("pipeline" in (constraint.get("sqltext") or "") for constraint in constraints)
        fk_cascade_present = _has_required_fk_cascade(conn) if table_exists else False

        duplicate_rows = 0
        if table_exists:
            duplicate_rows = conn.execute(
                text(
                    """
                    SELECT COUNT(*) FROM (
                      SELECT content_item_id, pipeline, COUNT(*) c
                      FROM content_item_pipeline_state
                      GROUP BY content_item_id, pipeline
                      HAVING c > 1
                    ) d
                    """
                )
            ).scalar() or 0

        invalid_status_rows = 0
        if table_exists:
            invalid_status_rows = conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM content_item_pipeline_state
                    WHERE status NOT IN (
                      'pending', 'in_progress', 'processed',
                      'failed_retryable', 'failed_terminal'
                    )
                    """
                )
            ).scalar() or 0

        verification = {
            "table_exists": table_exists,
            "missing_columns": missing_columns,
            "missing_indexes": missing_indexes,
            "status_check_present": status_check_present,
            "pipeline_check_present": pipeline_check_present,
            "fk_cascade_present": fk_cascade_present,
            "duplicate_rows": int(duplicate_rows),
            "invalid_status_rows": int(invalid_status_rows),
        }
        verification["ok"] = (
            verification["table_exists"]
            and not verification["missing_columns"]
            and not verification["missing_indexes"]
            and verification["status_check_present"]
            and verification["pipeline_check_present"]
            and verification["fk_cascade_present"]
            and verification["duplicate_rows"] == 0
            and verification["invalid_status_rows"] == 0
        )
        return verification


def _get_existing_tables(conn) -> set[str]:
    return table_names(conn)


def _get_table_columns(conn) -> set[str]:
    return column_names(conn, TABLE_NAME)


def _get_table_indexes(conn) -> set[str]:
    return index_names(conn, TABLE_NAME)


def _create_pipeline_state_table(conn) -> None:
    ContentItemPipelineState.__table__.create(bind=conn, checkfirst=True)


def _has_required_fk_cascade(conn) -> bool:
    for row in foreign_keys(conn, TABLE_NAME):
        ref_table = row.get("referred_table")
        constrained_cols = row.get("constrained_columns") or []
        referred_cols = row.get("referred_columns") or []
        on_delete = (row.get("options") or {}).get("ondelete", "").upper()
        if ref_table == "content_items" and constrained_cols == ["content_item_id"] and referred_cols == ["id"] and on_delete == "CASCADE":
            return True
    return False


def _needs_table_rebuild(conn) -> bool:
    return False


def _rebuild_pipeline_state_table(conn) -> None:
    return


def _add_missing_columns(conn) -> list[str]:
    existing = _get_table_columns(conn)
    column_ddl = {
        "attempt_count": "ALTER TABLE content_item_pipeline_state ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0",
        "error_code": "ALTER TABLE content_item_pipeline_state ADD COLUMN error_code TEXT",
        "error_message": "ALTER TABLE content_item_pipeline_state ADD COLUMN error_message TEXT",
        "last_attempt_at": "ALTER TABLE content_item_pipeline_state ADD COLUMN last_attempt_at TIMESTAMP WITH TIME ZONE",
        "processed_at": "ALTER TABLE content_item_pipeline_state ADD COLUMN processed_at TIMESTAMP WITH TIME ZONE",
        "created_at": "ALTER TABLE content_item_pipeline_state ADD COLUMN created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP",
        "updated_at": "ALTER TABLE content_item_pipeline_state ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP",
    }
    added: list[str] = []
    for column, ddl in column_ddl.items():
        if column not in existing:
            conn.execute(text(ddl))
            added.append(column)
    return added


def _ensure_indexes(conn) -> list[str]:
    stmts = [
        (
            "uix_cips_content_item_pipeline",
            "CREATE UNIQUE INDEX IF NOT EXISTS uix_cips_content_item_pipeline ON content_item_pipeline_state(content_item_id, pipeline)",
        ),
        (
            "idx_cips_pipeline_status_last_attempt",
            "CREATE INDEX IF NOT EXISTS idx_cips_pipeline_status_last_attempt ON content_item_pipeline_state(pipeline, status, last_attempt_at)",
        ),
        (
            "idx_cips_pipeline_status_created",
            "CREATE INDEX IF NOT EXISTS idx_cips_pipeline_status_created ON content_item_pipeline_state(pipeline, status, created_at)",
        ),
        (
            "idx_cips_content_item_pipeline_status",
            "CREATE INDEX IF NOT EXISTS idx_cips_content_item_pipeline_status ON content_item_pipeline_state(content_item_id, pipeline, status)",
        ),
        (
            "idx_cips_error_code",
            "CREATE INDEX IF NOT EXISTS idx_cips_error_code ON content_item_pipeline_state(error_code)",
        ),
        (
            "idx_cips_updated_at",
            "CREATE INDEX IF NOT EXISTS idx_cips_updated_at ON content_item_pipeline_state(updated_at)",
        ),
    ]
    for _, ddl in stmts:
        conn.execute(text(ddl))
    return [name for name, _ in stmts]
