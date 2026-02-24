"""Idempotent migration for theme_aliases table and indexes."""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

TABLE_NAME = "theme_aliases"

REQUIRED_COLUMNS = {
    "id",
    "theme_cluster_id",
    "pipeline",
    "alias_text",
    "alias_key",
    "source",
    "confidence",
    "evidence_count",
    "first_seen_at",
    "last_seen_at",
    "is_active",
    "created_at",
    "updated_at",
}

EXPECTED_INDEXES = {
    "uix_theme_alias_pipeline_alias_key",
    "idx_theme_alias_cluster_active",
    "idx_theme_alias_source_confidence",
    "idx_theme_aliases_alias_key",
    "idx_theme_aliases_pipeline",
}


def migrate_theme_aliases(engine) -> dict[str, Any]:
    """Create/patch theme_aliases schema idempotently."""
    stats: dict[str, Any] = {
        "table_created": False,
        "columns_added": [],
        "indexes_ensured": [],
    }

    with engine.connect() as conn:
        existing_tables = _get_existing_tables(conn)
        if TABLE_NAME not in existing_tables:
            _create_theme_aliases_table(conn)
            stats["table_created"] = True
        else:
            stats["columns_added"] = _add_missing_columns(conn)

        stats["indexes_ensured"] = _ensure_indexes(conn)
        conn.commit()

    logger.info("Theme aliases migration completed: %s", stats)
    return stats


def verify_theme_aliases_schema(engine) -> dict[str, Any]:
    """Verify post-migration schema and uniqueness for alias lookups."""
    with engine.connect() as conn:
        tables = _get_existing_tables(conn)
        table_exists = TABLE_NAME in tables
        columns = _get_table_columns(conn) if table_exists else set()
        missing_columns = sorted(REQUIRED_COLUMNS - columns)
        indexes = _get_table_indexes(conn) if table_exists else set()
        missing_indexes = sorted(EXPECTED_INDEXES - indexes)

        duplicate_alias_keys = 0
        null_identity_rows = 0
        if table_exists:
            duplicate_alias_keys = conn.execute(
                text(
                    """
                    SELECT COUNT(*) FROM (
                      SELECT pipeline, alias_key, COUNT(*) c
                      FROM theme_aliases
                      GROUP BY pipeline, alias_key
                      HAVING c > 1
                    ) d
                    """
                )
            ).scalar() or 0
            null_identity_rows = conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM theme_aliases
                    WHERE pipeline IS NULL OR pipeline = ''
                       OR alias_key IS NULL OR alias_key = ''
                       OR alias_text IS NULL OR alias_text = ''
                    """
                )
            ).scalar() or 0

        verification = {
            "table_exists": table_exists,
            "missing_columns": missing_columns,
            "missing_indexes": missing_indexes,
            "duplicate_alias_keys": int(duplicate_alias_keys),
            "null_identity_rows": int(null_identity_rows),
        }
        verification["ok"] = (
            verification["table_exists"]
            and not verification["missing_columns"]
            and not verification["missing_indexes"]
            and verification["duplicate_alias_keys"] == 0
            and verification["null_identity_rows"] == 0
        )
        return verification


def _get_existing_tables(conn) -> set[str]:
    rows = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
    return {row[0] for row in rows}


def _get_table_columns(conn) -> set[str]:
    rows = conn.execute(text(f"PRAGMA table_info({TABLE_NAME})")).fetchall()
    return {row[1] for row in rows}


def _get_table_indexes(conn) -> set[str]:
    rows = conn.execute(text(f"PRAGMA index_list({TABLE_NAME})")).fetchall()
    return {row[1] for row in rows}


def _create_theme_aliases_table(conn) -> None:
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS theme_aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                theme_cluster_id INTEGER NOT NULL,
                pipeline TEXT NOT NULL,
                alias_text TEXT NOT NULL,
                alias_key TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'llm_extraction',
                confidence FLOAT NOT NULL DEFAULT 0.5,
                evidence_count INTEGER NOT NULL DEFAULT 1,
                first_seen_at DATETIME,
                last_seen_at DATETIME,
                is_active BOOLEAN NOT NULL DEFAULT 1,
                created_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP),
                updated_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP),
                FOREIGN KEY (theme_cluster_id) REFERENCES theme_clusters(id) ON DELETE CASCADE
            )
            """
        )
    )


def _add_missing_columns(conn) -> list[str]:
    existing = _get_table_columns(conn)
    column_ddl = {
        "source": "ALTER TABLE theme_aliases ADD COLUMN source TEXT NOT NULL DEFAULT 'llm_extraction'",
        "confidence": "ALTER TABLE theme_aliases ADD COLUMN confidence FLOAT NOT NULL DEFAULT 0.5",
        "evidence_count": "ALTER TABLE theme_aliases ADD COLUMN evidence_count INTEGER NOT NULL DEFAULT 1",
        "first_seen_at": "ALTER TABLE theme_aliases ADD COLUMN first_seen_at DATETIME",
        "last_seen_at": "ALTER TABLE theme_aliases ADD COLUMN last_seen_at DATETIME",
        "is_active": "ALTER TABLE theme_aliases ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT 1",
        "created_at": "ALTER TABLE theme_aliases ADD COLUMN created_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP)",
        "updated_at": "ALTER TABLE theme_aliases ADD COLUMN updated_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP)",
    }
    added: list[str] = []
    for column, ddl in column_ddl.items():
        if column not in existing:
            conn.execute(text(ddl))
            added.append(column)
    return added


def _ensure_indexes(conn) -> list[str]:
    ensured = []

    conn.execute(
        text(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uix_theme_alias_pipeline_alias_key
            ON theme_aliases(pipeline, alias_key)
            """
        )
    )
    ensured.append("uix_theme_alias_pipeline_alias_key")

    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_theme_alias_cluster_active
            ON theme_aliases(theme_cluster_id, is_active)
            """
        )
    )
    ensured.append("idx_theme_alias_cluster_active")

    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_theme_alias_source_confidence
            ON theme_aliases(source, confidence)
            """
        )
    )
    ensured.append("idx_theme_alias_source_confidence")

    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_theme_aliases_alias_key
            ON theme_aliases(alias_key)
            """
        )
    )
    ensured.append("idx_theme_aliases_alias_key")

    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_theme_aliases_pipeline
            ON theme_aliases(pipeline)
            """
        )
    )
    ensured.append("idx_theme_aliases_pipeline")

    return ensured
