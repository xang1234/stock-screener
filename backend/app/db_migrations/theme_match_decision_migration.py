"""Idempotent migration for theme_mentions match decision telemetry columns."""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

TABLE_NAME = "theme_mentions"

REQUIRED_COLUMNS = {
    "match_method",
    "match_score",
    "match_threshold",
    "threshold_version",
    "match_score_model",
    "match_score_model_version",
    "match_fallback_reason",
    "best_alternative_cluster_id",
    "best_alternative_score",
    "match_score_margin",
}

EXPECTED_INDEXES = {
    "idx_theme_mentions_match_method",
    "idx_theme_mentions_threshold_version",
    "idx_theme_mentions_match_score_model",
    "idx_theme_mentions_match_score_model_version",
    "idx_theme_mentions_best_alternative_cluster_id",
}


def migrate_theme_match_decision(engine) -> dict[str, Any]:
    """Add match decision telemetry columns/indexes to theme_mentions idempotently."""
    stats: dict[str, Any] = {
        "columns_added": [],
        "indexes_ensured": [],
        "table_exists": False,
    }

    with engine.connect() as conn:
        tables = _get_existing_tables(conn)
        if TABLE_NAME not in tables:
            conn.commit()
            return stats

        stats["table_exists"] = True
        stats["columns_added"] = _add_missing_columns(conn)
        stats["indexes_ensured"] = _ensure_indexes(conn)
        conn.commit()

    logger.info("Theme match decision migration completed: %s", stats)
    return stats


def verify_theme_match_decision_schema(engine) -> dict[str, Any]:
    """Verify post-migration match decision schema on theme_mentions."""
    with engine.connect() as conn:
        tables = _get_existing_tables(conn)
        table_exists = TABLE_NAME in tables
        columns = _get_table_columns(conn) if table_exists else set()
        missing_columns = sorted(REQUIRED_COLUMNS - columns)
        indexes = _get_table_indexes(conn) if table_exists else set()
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
    column_ddl = {
        "match_method": "ALTER TABLE theme_mentions ADD COLUMN match_method TEXT",
        "match_score": "ALTER TABLE theme_mentions ADD COLUMN match_score FLOAT",
        "match_threshold": "ALTER TABLE theme_mentions ADD COLUMN match_threshold FLOAT",
        "threshold_version": "ALTER TABLE theme_mentions ADD COLUMN threshold_version TEXT",
        "match_score_model": "ALTER TABLE theme_mentions ADD COLUMN match_score_model TEXT",
        "match_score_model_version": "ALTER TABLE theme_mentions ADD COLUMN match_score_model_version TEXT",
        "match_fallback_reason": "ALTER TABLE theme_mentions ADD COLUMN match_fallback_reason TEXT",
        "best_alternative_cluster_id": "ALTER TABLE theme_mentions ADD COLUMN best_alternative_cluster_id INTEGER",
        "best_alternative_score": "ALTER TABLE theme_mentions ADD COLUMN best_alternative_score FLOAT",
        "match_score_margin": "ALTER TABLE theme_mentions ADD COLUMN match_score_margin FLOAT",
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
            CREATE INDEX IF NOT EXISTS idx_theme_mentions_match_method
            ON theme_mentions(match_method)
            """
        )
    )
    ensured.append("idx_theme_mentions_match_method")

    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_theme_mentions_threshold_version
            ON theme_mentions(threshold_version)
            """
        )
    )
    ensured.append("idx_theme_mentions_threshold_version")

    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_theme_mentions_match_score_model
            ON theme_mentions(match_score_model)
            """
        )
    )
    ensured.append("idx_theme_mentions_match_score_model")

    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_theme_mentions_match_score_model_version
            ON theme_mentions(match_score_model_version)
            """
        )
    )
    ensured.append("idx_theme_mentions_match_score_model_version")

    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_theme_mentions_best_alternative_cluster_id
            ON theme_mentions(best_alternative_cluster_id)
            """
        )
    )
    ensured.append("idx_theme_mentions_best_alternative_cluster_id")

    return ensured
