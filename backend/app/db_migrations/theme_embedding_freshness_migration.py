"""Idempotent migration for theme_embeddings freshness metadata."""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

TABLE_NAME = "theme_embeddings"

REQUIRED_COLUMNS = {
    "id",
    "theme_cluster_id",
    "embedding",
    "embedding_model",
    "embedding_text",
    "content_hash",
    "model_version",
    "is_stale",
    "created_at",
    "updated_at",
}

EXPECTED_INDEXES = {
    "ix_theme_embeddings_content_hash",
    "ix_theme_embeddings_model_version",
    "ix_theme_embeddings_is_stale",
}


def migrate_theme_embedding_freshness(engine) -> dict[str, Any]:
    """Add freshness metadata columns/indexes and backfill stale markers."""
    stats: dict[str, Any] = {
        "table_exists": False,
        "columns_added": [],
        "rows_marked_stale": 0,
        "indexes_ensured": [],
    }

    with engine.connect() as conn:
        tables = _get_existing_tables(conn)
        if TABLE_NAME not in tables:
            conn.commit()
            return stats

        stats["table_exists"] = True
        stats["columns_added"] = _add_missing_columns(conn)
        stats["rows_marked_stale"] = _backfill_stale_flags(conn)
        stats["indexes_ensured"] = _ensure_indexes(conn)
        conn.commit()

    logger.info("Theme embedding freshness migration completed: %s", stats)
    return stats


def verify_theme_embedding_freshness_schema(engine) -> dict[str, Any]:
    """Verify post-migration schema shape for embedding freshness metadata."""
    with engine.connect() as conn:
        tables = _get_existing_tables(conn)
        table_exists = TABLE_NAME in tables
        columns = _get_table_columns(conn) if table_exists else set()
        missing_columns = sorted(REQUIRED_COLUMNS - columns)
        indexes = _get_table_indexes(conn) if table_exists else set()
        missing_indexes = sorted(EXPECTED_INDEXES - indexes)

        verification = {
            "table_exists": table_exists,
            "missing_columns": missing_columns,
            "missing_indexes": missing_indexes,
        }
        verification["ok"] = (
            verification["table_exists"]
            and not verification["missing_columns"]
            and not verification["missing_indexes"]
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


def _add_missing_columns(conn) -> list[str]:
    existing = _get_table_columns(conn)
    column_ddl = {
        "content_hash": "ALTER TABLE theme_embeddings ADD COLUMN content_hash TEXT",
        "model_version": "ALTER TABLE theme_embeddings ADD COLUMN model_version TEXT NOT NULL DEFAULT 'embedding-v1'",
        "is_stale": "ALTER TABLE theme_embeddings ADD COLUMN is_stale BOOLEAN NOT NULL DEFAULT 0",
    }
    added: list[str] = []
    for column, ddl in column_ddl.items():
        if column not in existing:
            conn.execute(text(ddl))
            added.append(column)
    return added


def _backfill_stale_flags(conn) -> int:
    conn.execute(
        text(
            """
            UPDATE theme_embeddings
            SET model_version = 'embedding-v1'
            WHERE model_version IS NULL OR TRIM(model_version) = ''
            """
        )
    )
    result = conn.execute(
        text(
            """
            UPDATE theme_embeddings
            SET is_stale = 1
            WHERE (content_hash IS NULL OR TRIM(content_hash) = '')
              AND (is_stale IS NULL OR is_stale = 0)
            """
        )
    )
    return int(result.rowcount or 0)


def _ensure_indexes(conn) -> list[str]:
    ensured: list[str] = []
    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS ix_theme_embeddings_content_hash
            ON theme_embeddings(content_hash)
            """
        )
    )
    ensured.append("ix_theme_embeddings_content_hash")

    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS ix_theme_embeddings_model_version
            ON theme_embeddings(model_version)
            """
        )
    )
    ensured.append("ix_theme_embeddings_model_version")

    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS ix_theme_embeddings_is_stale
            ON theme_embeddings(is_stale)
            """
        )
    )
    ensured.append("ix_theme_embeddings_is_stale")
    return ensured
