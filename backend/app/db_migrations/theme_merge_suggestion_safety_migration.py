"""Idempotent migration for merge-suggestion canonical pair + idempotency fields."""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

TABLE_NAME = "theme_merge_suggestions"


def migrate_theme_merge_suggestion_safety(engine) -> dict[str, Any]:
    """Add canonical pair and approval idempotency fields for merge suggestions."""
    stats: dict[str, Any] = {
        "table_exists": False,
        "columns_added": [],
        "rows_backfilled": 0,
        "duplicates_removed": 0,
        "indexes_ensured": [],
    }
    with engine.connect() as conn:
        tables = _get_existing_tables(conn)
        if TABLE_NAME not in tables:
            conn.commit()
            return stats
        stats["table_exists"] = True
        stats["columns_added"] = _add_missing_columns(conn)
        stats["rows_backfilled"] = _backfill_canonical_pairs(conn)
        stats["duplicates_removed"] = _dedupe_canonical_pairs(conn)
        stats["indexes_ensured"] = _ensure_indexes(conn)
        conn.commit()
    logger.info("Theme merge suggestion safety migration completed: %s", stats)
    return stats


def _get_existing_tables(conn) -> set[str]:
    rows = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
    return {row[0] for row in rows}


def _get_table_columns(conn) -> set[str]:
    rows = conn.execute(text(f"PRAGMA table_info({TABLE_NAME})")).fetchall()
    return {row[1] for row in rows}


def _add_missing_columns(conn) -> list[str]:
    existing = _get_table_columns(conn)
    ddl_by_column = {
        "pair_min_cluster_id": "ALTER TABLE theme_merge_suggestions ADD COLUMN pair_min_cluster_id INTEGER",
        "pair_max_cluster_id": "ALTER TABLE theme_merge_suggestions ADD COLUMN pair_max_cluster_id INTEGER",
        "approval_idempotency_key": "ALTER TABLE theme_merge_suggestions ADD COLUMN approval_idempotency_key VARCHAR(128)",
        "approval_result_json": "ALTER TABLE theme_merge_suggestions ADD COLUMN approval_result_json TEXT",
    }
    added: list[str] = []
    for column, ddl in ddl_by_column.items():
        if column not in existing:
            conn.execute(text(ddl))
            added.append(column)
    return added


def _backfill_canonical_pairs(conn) -> int:
    result = conn.execute(
        text(
            """
            UPDATE theme_merge_suggestions
            SET pair_min_cluster_id = CASE
                    WHEN source_cluster_id <= target_cluster_id THEN source_cluster_id
                    ELSE target_cluster_id
                END,
                pair_max_cluster_id = CASE
                    WHEN source_cluster_id >= target_cluster_id THEN source_cluster_id
                    ELSE target_cluster_id
                END
            WHERE pair_min_cluster_id IS NULL
               OR pair_max_cluster_id IS NULL
            """
        )
    )
    return int(result.rowcount or 0)


def _ensure_indexes(conn) -> list[str]:
    ensured: list[str] = []
    conn.execute(
        text(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uix_merge_suggestion_pair_canonical
            ON theme_merge_suggestions(pair_min_cluster_id, pair_max_cluster_id)
            """
        )
    )
    ensured.append("uix_merge_suggestion_pair_canonical")
    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS ix_theme_merge_suggestions_approval_idempotency_key
            ON theme_merge_suggestions(approval_idempotency_key)
            """
        )
    )
    ensured.append("ix_theme_merge_suggestions_approval_idempotency_key")
    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS ix_theme_merge_suggestions_pair_min_cluster_id
            ON theme_merge_suggestions(pair_min_cluster_id)
            """
        )
    )
    ensured.append("ix_theme_merge_suggestions_pair_min_cluster_id")
    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS ix_theme_merge_suggestions_pair_max_cluster_id
            ON theme_merge_suggestions(pair_max_cluster_id)
            """
        )
    )
    ensured.append("ix_theme_merge_suggestions_pair_max_cluster_id")
    return ensured


def _dedupe_canonical_pairs(conn) -> int:
    result = conn.execute(
        text(
            """
            DELETE FROM theme_merge_suggestions
            WHERE id IN (
                WITH ranked AS (
                    SELECT
                        id,
                        ROW_NUMBER() OVER (
                            PARTITION BY pair_min_cluster_id, pair_max_cluster_id
                            ORDER BY
                                CASE status
                                    WHEN 'approved' THEN 5
                                    WHEN 'auto_merged' THEN 4
                                    WHEN 'pending' THEN 3
                                    WHEN 'rejected' THEN 2
                                    ELSE 1
                                END DESC,
                                CASE WHEN approval_idempotency_key IS NOT NULL THEN 1 ELSE 0 END DESC,
                                COALESCE(reviewed_at, created_at) DESC,
                                id DESC
                        ) AS rn
                    FROM theme_merge_suggestions
                    WHERE pair_min_cluster_id IS NOT NULL
                      AND pair_max_cluster_id IS NOT NULL
                )
                SELECT id FROM ranked WHERE rn > 1
            )
            """
        )
    )
    return int(result.rowcount or 0)
