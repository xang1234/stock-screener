"""Idempotent migration for L1/L2 theme taxonomy columns on theme_clusters."""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

TABLE_NAME = "theme_clusters"

TAXONOMY_COLUMNS = {
    "parent_cluster_id": "ALTER TABLE theme_clusters ADD COLUMN parent_cluster_id INTEGER REFERENCES theme_clusters(id)",
    "is_l1": "ALTER TABLE theme_clusters ADD COLUMN is_l1 BOOLEAN NOT NULL DEFAULT 0",
    "taxonomy_level": "ALTER TABLE theme_clusters ADD COLUMN taxonomy_level INTEGER NOT NULL DEFAULT 2",
    "l1_assignment_method": "ALTER TABLE theme_clusters ADD COLUMN l1_assignment_method TEXT",
    "l1_assignment_confidence": "ALTER TABLE theme_clusters ADD COLUMN l1_assignment_confidence REAL",
    "l1_assigned_at": "ALTER TABLE theme_clusters ADD COLUMN l1_assigned_at DATETIME",
}

EXPECTED_INDEXES = {
    "idx_tc_parent_cluster_id",
    "idx_tc_is_l1_pipeline_active",
    "idx_tc_taxonomy_level",
}


def migrate_theme_taxonomy(engine) -> dict[str, Any]:
    """Add L1/L2 taxonomy columns and indexes to theme_clusters idempotently."""
    stats: dict[str, Any] = {
        "columns_added": [],
        "indexes_ensured": [],
    }

    with engine.connect() as conn:
        existing_columns = _get_table_columns(conn)

        for column, ddl in TAXONOMY_COLUMNS.items():
            if column in existing_columns:
                continue
            conn.execute(text(ddl))
            stats["columns_added"].append(column)

        stats["indexes_ensured"] = _ensure_indexes(conn)
        conn.commit()

    if stats["columns_added"]:
        logger.info("Theme taxonomy migration: added columns %s", stats["columns_added"])
    else:
        logger.debug("Theme taxonomy migration: all columns already present")

    return stats


def verify_theme_taxonomy_schema(engine) -> dict[str, Any]:
    """Verify post-migration schema for theme taxonomy columns."""
    with engine.connect() as conn:
        columns = _get_table_columns(conn)
        indexes = _get_table_indexes(conn)
        missing_columns = sorted(set(TAXONOMY_COLUMNS.keys()) - columns)
        missing_indexes = sorted(EXPECTED_INDEXES - indexes)
        return {
            "missing_columns": missing_columns,
            "missing_indexes": missing_indexes,
            "ok": not missing_columns and not missing_indexes,
        }


def _get_table_columns(conn) -> set[str]:
    rows = conn.execute(text(f"PRAGMA table_info({TABLE_NAME})")).fetchall()
    return {row[1] for row in rows}


def _get_table_indexes(conn) -> set[str]:
    rows = conn.execute(text(f"PRAGMA index_list({TABLE_NAME})")).fetchall()
    return {row[1] for row in rows}


def _ensure_indexes(conn) -> list[str]:
    ensured: list[str] = []

    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_tc_parent_cluster_id
            ON theme_clusters(parent_cluster_id)
            """
        )
    )
    ensured.append("idx_tc_parent_cluster_id")

    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_tc_is_l1_pipeline_active
            ON theme_clusters(is_l1, pipeline, is_active)
            """
        )
    )
    ensured.append("idx_tc_is_l1_pipeline_active")

    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_tc_taxonomy_level
            ON theme_clusters(taxonomy_level)
            """
        )
    )
    ensured.append("idx_tc_taxonomy_level")

    return ensured
