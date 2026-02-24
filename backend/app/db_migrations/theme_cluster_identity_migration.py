"""Idempotent migration for theme_clusters canonical identity schema."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from sqlalchemy import text

from ..services.theme_identity_normalization import UNKNOWN_THEME_KEY, canonical_theme_key, display_theme_name

logger = logging.getLogger(__name__)

TABLE_NAME = "theme_clusters"

REQUIRED_COLUMNS = {
    "id",
    "name",
    "canonical_key",
    "display_name",
    "aliases",
    "description",
    "pipeline",
    "category",
    "is_emerging",
    "first_seen_at",
    "last_seen_at",
    "discovery_source",
    "is_active",
    "is_validated",
    "created_at",
    "updated_at",
}

UNIQUE_INDEX_NAME = "uix_theme_clusters_pipeline_canonical_key"


def migrate_theme_cluster_identity(engine) -> dict[str, Any]:
    """Create/patch theme_clusters schema idempotently."""
    stats: dict[str, Any] = {
        "table_created": False,
        "table_rebuilt": False,
        "rows_backfilled": 0,
        "indexes_ensured": [],
    }

    with engine.connect() as conn:
        existing_tables = _get_existing_tables(conn)
        if TABLE_NAME not in existing_tables:
            _create_theme_clusters_table(conn, TABLE_NAME)
            stats["table_created"] = True
        elif _needs_table_rebuild(conn):
            rows = _build_migrated_rows(conn)
            _assert_no_pipeline_key_duplicates(rows)
            _rebuild_theme_clusters_table(conn, rows)
            stats["table_rebuilt"] = True
            stats["rows_backfilled"] = len(rows)
        else:
            rows = _build_migrated_rows(conn)
            _assert_no_pipeline_key_duplicates(rows)
            stats["rows_backfilled"] = _backfill_identity_columns(conn, rows)

        _ensure_unique_index(conn)
        stats["indexes_ensured"] = [UNIQUE_INDEX_NAME]
        conn.commit()

    logger.info("Theme cluster identity migration completed: %s", stats)
    return stats


def verify_theme_cluster_identity_schema(engine) -> dict[str, Any]:
    """Verify post-migration identity schema and uniqueness semantics."""
    with engine.connect() as conn:
        tables = _get_existing_tables(conn)
        table_exists = TABLE_NAME in tables
        columns = _get_table_columns(conn) if table_exists else set()
        missing_columns = sorted(REQUIRED_COLUMNS - columns)

        indexes = _get_index_defs(conn) if table_exists else []
        has_pipeline_key_unique = any(
            idx["unique"] and idx["columns"] == ["pipeline", "canonical_key"] for idx in indexes
        )
        has_global_name_unique = any(idx["unique"] and idx["columns"] == ["name"] for idx in indexes)

        duplicate_pipeline_keys = 0
        null_identity_rows = 0
        if table_exists:
            duplicate_pipeline_keys = conn.execute(
                text(
                    """
                    SELECT COUNT(*) FROM (
                      SELECT pipeline, canonical_key, COUNT(*) c
                      FROM theme_clusters
                      GROUP BY pipeline, canonical_key
                      HAVING c > 1
                    ) d
                    """
                )
            ).scalar() or 0
            null_identity_rows = conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM theme_clusters
                    WHERE canonical_key IS NULL OR canonical_key = ''
                       OR display_name IS NULL OR display_name = ''
                    """
                )
            ).scalar() or 0

        verification = {
            "table_exists": table_exists,
            "missing_columns": missing_columns,
            "has_pipeline_key_unique": has_pipeline_key_unique,
            "has_global_name_unique": has_global_name_unique,
            "duplicate_pipeline_keys": int(duplicate_pipeline_keys),
            "null_identity_rows": int(null_identity_rows),
        }
        verification["ok"] = (
            verification["table_exists"]
            and not verification["missing_columns"]
            and verification["has_pipeline_key_unique"]
            and not verification["has_global_name_unique"]
            and verification["duplicate_pipeline_keys"] == 0
            and verification["null_identity_rows"] == 0
        )
        return verification


def _get_existing_tables(conn) -> set[str]:
    rows = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
    return {row[0] for row in rows}


def _get_table_columns(conn) -> set[str]:
    rows = conn.execute(text(f"PRAGMA table_info({TABLE_NAME})")).fetchall()
    return {row[1] for row in rows}


def _get_index_defs(conn) -> list[dict[str, Any]]:
    rows = conn.execute(text(f"PRAGMA index_list({TABLE_NAME})")).fetchall()
    indexes: list[dict[str, Any]] = []
    for row in rows:
        name = row[1]
        unique = bool(row[2])
        cols = conn.execute(text(f"PRAGMA index_info({name})")).fetchall()
        indexes.append(
            {
                "name": name,
                "unique": unique,
                "columns": [c[2] for c in cols],
            }
        )
    return indexes


def _needs_table_rebuild(conn) -> bool:
    columns = _get_table_columns(conn)
    if "canonical_key" not in columns or "display_name" not in columns:
        return True

    indexes = _get_index_defs(conn)
    has_global_name_unique = any(idx["unique"] and idx["columns"] == ["name"] for idx in indexes)
    return has_global_name_unique


def _build_migrated_rows(conn) -> list[dict[str, Any]]:
    columns = _get_table_columns(conn)
    selected = [col for col in REQUIRED_COLUMNS if col in columns]
    rows = conn.execute(text(f"SELECT {', '.join(sorted(selected))} FROM {TABLE_NAME}")).mappings().all()

    migrated: list[dict[str, Any]] = []
    for row in rows:
        pipeline = row.get("pipeline") or "technical"
        seed = row.get("display_name") or row.get("name") or row.get("canonical_key") or ""
        canonical_key = (row.get("canonical_key") or canonical_theme_key(seed) or UNKNOWN_THEME_KEY).strip()
        display_name = (row.get("display_name") or display_theme_name(seed) or "Unknown Theme").strip()
        legacy_name = (row.get("name") or display_name).strip()

        migrated.append(
            {
                "id": row.get("id"),
                "name": legacy_name,
                "canonical_key": canonical_key,
                "display_name": display_name,
                "aliases": row.get("aliases"),
                "description": row.get("description"),
                "pipeline": pipeline,
                "category": row.get("category"),
                "is_emerging": row.get("is_emerging", True),
                "first_seen_at": row.get("first_seen_at"),
                "last_seen_at": row.get("last_seen_at"),
                "discovery_source": row.get("discovery_source"),
                "is_active": row.get("is_active", True),
                "is_validated": row.get("is_validated", False),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
            }
        )
    return migrated


def _assert_no_pipeline_key_duplicates(rows: list[dict[str, Any]]) -> None:
    by_key: dict[tuple[str, str], list[int | None]] = defaultdict(list)
    for row in rows:
        by_key[(row["pipeline"], row["canonical_key"])].append(row["id"])

    duplicates = {
        key: ids
        for key, ids in by_key.items()
        if len(ids) > 1
    }
    if duplicates:
        sample = list(duplicates.items())[:5]
        raise ValueError(
            "theme_clusters migration blocked by duplicate canonical_key per pipeline: "
            f"{sample}"
        )


def _create_theme_clusters_table(conn, table_name: str) -> None:
    conn.execute(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                canonical_key TEXT NOT NULL,
                display_name TEXT NOT NULL,
                aliases JSON,
                description TEXT,
                pipeline TEXT DEFAULT 'technical',
                category TEXT,
                is_emerging BOOLEAN DEFAULT 1,
                first_seen_at DATETIME,
                last_seen_at DATETIME,
                discovery_source TEXT,
                is_active BOOLEAN DEFAULT 1,
                is_validated BOOLEAN DEFAULT 0,
                created_at DATETIME DEFAULT (CURRENT_TIMESTAMP),
                updated_at DATETIME DEFAULT (CURRENT_TIMESTAMP)
            )
            """
        )
    )


def _rebuild_theme_clusters_table(conn, rows: list[dict[str, Any]]) -> None:
    old_table = TABLE_NAME
    new_table = f"{TABLE_NAME}__new"

    conn.execute(text(f"DROP TABLE IF EXISTS {new_table}"))
    _create_theme_clusters_table(conn, new_table)

    if rows:
        conn.execute(
            text(
                f"""
                INSERT INTO {new_table} (
                    id, name, canonical_key, display_name, aliases, description, pipeline, category,
                    is_emerging, first_seen_at, last_seen_at, discovery_source, is_active,
                    is_validated, created_at, updated_at
                ) VALUES (
                    :id, :name, :canonical_key, :display_name, :aliases, :description, :pipeline, :category,
                    :is_emerging, :first_seen_at, :last_seen_at, :discovery_source, :is_active,
                    :is_validated, :created_at, :updated_at
                )
                """
            ),
            rows,
        )

    conn.execute(text("PRAGMA foreign_keys=OFF"))
    try:
        conn.execute(text(f"DROP TABLE {old_table}"))
        conn.execute(text(f"ALTER TABLE {new_table} RENAME TO {old_table}"))
    finally:
        conn.execute(text("PRAGMA foreign_keys=ON"))


def _backfill_identity_columns(conn, rows: list[dict[str, Any]]) -> int:
    updated = 0
    for row in rows:
        result = conn.execute(
            text(
                """
                UPDATE theme_clusters
                SET canonical_key = :canonical_key,
                    display_name = :display_name,
                    name = :name
                WHERE id = :id
                  AND (
                    canonical_key IS NULL OR canonical_key = ''
                    OR display_name IS NULL OR display_name = ''
                    OR name IS NULL OR name = ''
                  )
                """
            ),
            {
                "id": row["id"],
                "canonical_key": row["canonical_key"],
                "display_name": row["display_name"],
                "name": row["name"],
            },
        )
        updated += int(result.rowcount or 0)
    return updated


def _ensure_unique_index(conn) -> None:
    conn.execute(text(f"DROP INDEX IF EXISTS {UNIQUE_INDEX_NAME}"))
    conn.execute(
        text(
            f"""
            CREATE UNIQUE INDEX IF NOT EXISTS {UNIQUE_INDEX_NAME}
            ON {TABLE_NAME}(pipeline, canonical_key)
            """
        )
    )

