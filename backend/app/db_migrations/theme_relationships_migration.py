"""Idempotent migration for theme_relationships semantic edge table."""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

TABLE_NAME = "theme_relationships"

REQUIRED_COLUMNS = {
    "id",
    "source_cluster_id",
    "target_cluster_id",
    "pipeline",
    "relationship_type",
    "confidence",
    "provenance",
    "evidence",
    "is_active",
    "created_at",
    "updated_at",
}

EXPECTED_INDEXES = {
    "idx_theme_relationship_source_active",
    "idx_theme_relationship_target_active",
    "idx_theme_relationship_pipeline_type",
    "idx_theme_relationships_pair_type",
}


def migrate_theme_relationships(engine) -> dict[str, Any]:
    """Create/patch theme_relationships schema idempotently."""
    stats: dict[str, Any] = {
        "table_created": False,
        "columns_added": [],
        "indexes_ensured": [],
        "rows_normalized": 0,
        "self_edges_removed": 0,
    }

    with engine.connect() as conn:
        tables = _get_existing_tables(conn)
        if TABLE_NAME not in tables:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS theme_relationships (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_cluster_id INTEGER NOT NULL,
                        target_cluster_id INTEGER NOT NULL,
                        pipeline TEXT NOT NULL DEFAULT 'technical',
                        relationship_type TEXT NOT NULL,
                        confidence FLOAT NOT NULL DEFAULT 0.5,
                        provenance TEXT NOT NULL DEFAULT 'rule_inference',
                        evidence JSON,
                        is_active BOOLEAN NOT NULL DEFAULT 1,
                        created_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP),
                        updated_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP),
                        FOREIGN KEY (source_cluster_id) REFERENCES theme_clusters(id) ON DELETE CASCADE,
                        FOREIGN KEY (target_cluster_id) REFERENCES theme_clusters(id) ON DELETE CASCADE
                    )
                    """
                )
            )
            stats["table_created"] = True
        else:
            stats["columns_added"] = _add_missing_columns(conn)

        # Normalize any invalid relationship_type values for compatibility with older ad-hoc rows.
        normalized = conn.execute(
            text(
                """
                UPDATE theme_relationships
                SET relationship_type = 'related'
                WHERE relationship_type NOT IN ('subset', 'related', 'distinct')
                """
            )
        )
        stats["rows_normalized"] = int(normalized.rowcount or 0)

        removed = conn.execute(
            text("DELETE FROM theme_relationships WHERE source_cluster_id = target_cluster_id")
        )
        stats["self_edges_removed"] = int(removed.rowcount or 0)

        stats["indexes_ensured"] = _ensure_indexes(conn)
        conn.commit()

    logger.info("Theme relationships migration completed: %s", stats)
    return stats


def verify_theme_relationships_schema(engine) -> dict[str, Any]:
    """Verify post-migration schema and indexes for theme_relationships."""
    with engine.connect() as conn:
        tables = _get_existing_tables(conn)
        table_exists = TABLE_NAME in tables
        columns = _get_table_columns(conn) if table_exists else set()
        indexes = _get_table_indexes(conn) if table_exists else set()
        missing_columns = sorted(REQUIRED_COLUMNS - columns)
        missing_indexes = sorted(EXPECTED_INDEXES - indexes)
        return {
            "table_exists": table_exists,
            "missing_columns": missing_columns,
            "missing_indexes": missing_indexes,
            "ok": table_exists and not missing_columns and not missing_indexes,
        }


def _get_existing_tables(conn) -> set[str]:
    rows = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
    return {row[0] for row in rows}


def _get_table_columns(conn) -> set[str]:
    rows = conn.execute(text(f"PRAGMA table_info({TABLE_NAME})")).fetchall()
    return {row[1] for row in rows}


def _get_table_indexes(conn) -> set[str]:
    rows = conn.execute(text(f"PRAGMA index_list({TABLE_NAME})")).fetchall()
    return {row[1] for row in rows}


def _add_missing_columns(conn) -> list[str]:
    existing = _get_table_columns(conn)
    add_statements = {
        "source_cluster_id": "ALTER TABLE theme_relationships ADD COLUMN source_cluster_id INTEGER",
        "target_cluster_id": "ALTER TABLE theme_relationships ADD COLUMN target_cluster_id INTEGER",
        "pipeline": "ALTER TABLE theme_relationships ADD COLUMN pipeline TEXT NOT NULL DEFAULT 'technical'",
        "relationship_type": "ALTER TABLE theme_relationships ADD COLUMN relationship_type TEXT NOT NULL DEFAULT 'related'",
        "confidence": "ALTER TABLE theme_relationships ADD COLUMN confidence FLOAT NOT NULL DEFAULT 0.5",
        "provenance": "ALTER TABLE theme_relationships ADD COLUMN provenance TEXT NOT NULL DEFAULT 'rule_inference'",
        "evidence": "ALTER TABLE theme_relationships ADD COLUMN evidence JSON",
        "is_active": "ALTER TABLE theme_relationships ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT 1",
        "created_at": "ALTER TABLE theme_relationships ADD COLUMN created_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP)",
        "updated_at": "ALTER TABLE theme_relationships ADD COLUMN updated_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP)",
    }
    added: list[str] = []
    for column, ddl in add_statements.items():
        if column in existing:
            continue
        conn.execute(text(ddl))
        added.append(column)
    return added


def _ensure_indexes(conn) -> list[str]:
    ensured: list[str] = []
    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_theme_relationship_source_active
            ON theme_relationships(source_cluster_id, is_active)
            """
        )
    )
    ensured.append("idx_theme_relationship_source_active")

    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_theme_relationship_target_active
            ON theme_relationships(target_cluster_id, is_active)
            """
        )
    )
    ensured.append("idx_theme_relationship_target_active")

    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_theme_relationship_pipeline_type
            ON theme_relationships(pipeline, relationship_type, confidence)
            """
        )
    )
    ensured.append("idx_theme_relationship_pipeline_type")

    conn.execute(
        text(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_theme_relationships_pair_type
            ON theme_relationships(source_cluster_id, target_cluster_id, relationship_type, pipeline)
            """
        )
    )
    ensured.append("idx_theme_relationships_pair_type")

    return ensured
