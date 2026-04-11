"""Idempotent migration for theme taxonomy schema on theme_clusters."""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from ..infra.db.portability import column_names, dialect_name, index_names

logger = logging.getLogger(__name__)

THEME_TABLE = "theme_clusters"
PARENT_CLUSTER_INDEX = "ix_theme_clusters_parent_cluster_id"
LEGACY_PARENT_CLUSTER_INDEX = "idx_theme_clusters_parent_cluster_id"
IS_L1_INDEX = "idx_theme_clusters_is_l1"


def migrate_theme_taxonomy(engine) -> dict[str, Any]:
    """Ensure L1/L2 taxonomy columns exist on ``theme_clusters``."""
    stats: dict[str, Any] = {
        "columns_added": [],
        "indexes_ensured": [],
    }

    with engine.connect() as conn:
        columns = column_names(conn, THEME_TABLE)
        if not columns:
            logger.warning("Table %s not found; skipping taxonomy migration", THEME_TABLE)
            return stats

        is_postgres = dialect_name(conn) == "postgresql"
        timestamp_type = "TIMESTAMP WITH TIME ZONE" if is_postgres else "DATETIME"
        true_literal = "TRUE" if is_postgres else "1"
        false_literal = "FALSE" if is_postgres else "0"

        add_statements = {
            "parent_cluster_id": f"ALTER TABLE {THEME_TABLE} ADD COLUMN parent_cluster_id INTEGER",
            "is_l1": f"ALTER TABLE {THEME_TABLE} ADD COLUMN is_l1 BOOLEAN NOT NULL DEFAULT FALSE",
            "taxonomy_level": f"ALTER TABLE {THEME_TABLE} ADD COLUMN taxonomy_level INTEGER NOT NULL DEFAULT 2",
            "l1_assignment_method": f"ALTER TABLE {THEME_TABLE} ADD COLUMN l1_assignment_method VARCHAR(20)",
            "l1_assignment_confidence": f"ALTER TABLE {THEME_TABLE} ADD COLUMN l1_assignment_confidence FLOAT",
            "l1_assigned_at": f"ALTER TABLE {THEME_TABLE} ADD COLUMN l1_assigned_at {timestamp_type}",
        }

        for column, statement in add_statements.items():
            if column in columns:
                continue
            try:
                with conn.begin_nested():
                    conn.execute(text(statement))
                    stats["columns_added"].append(column)
            except SQLAlchemyError:
                # Another process may have applied the same ALTER between check and execute.
                if column not in column_names(conn, THEME_TABLE):
                    raise

        if "taxonomy_level" in column_names(conn, THEME_TABLE):
            conn.execute(
                text(
                    f"""
                    UPDATE {THEME_TABLE}
                    SET taxonomy_level = CASE
                        WHEN COALESCE(is_l1, {false_literal}) = {true_literal} THEN 1
                        ELSE 2
                    END
                    WHERE (
                        COALESCE(is_l1, {false_literal}) = {true_literal}
                        AND (taxonomy_level IS NULL OR taxonomy_level != 1)
                    )
                    OR (
                        COALESCE(is_l1, {false_literal}) != {true_literal}
                        AND (taxonomy_level IS NULL OR taxonomy_level != 2)
                    )
                    """
                )
            )

        existing_indexes = index_names(conn, THEME_TABLE)
        has_parent_cluster_index = (
            PARENT_CLUSTER_INDEX in existing_indexes
            or LEGACY_PARENT_CLUSTER_INDEX in existing_indexes
        )
        if not has_parent_cluster_index:
            conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS {PARENT_CLUSTER_INDEX}
                    ON {THEME_TABLE}(parent_cluster_id)
                    """
                )
            )
        if IS_L1_INDEX not in existing_indexes:
            conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS {IS_L1_INDEX}
                    ON {THEME_TABLE}(is_l1)
                    """
                )
            )

        stats["indexes_ensured"] = [PARENT_CLUSTER_INDEX, IS_L1_INDEX]
        conn.commit()

    logger.info("Theme taxonomy migration completed: %s", stats)
    return stats
