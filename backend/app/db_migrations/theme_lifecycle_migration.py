"""Idempotent migration for theme lifecycle schema and transition audit table."""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text

from ..infra.db.portability import column_names, sql_timestamp_type, table_names
from ..models.theme import ThemeLifecycleTransition

logger = logging.getLogger(__name__)

THEME_TABLE = "theme_clusters"
TRANSITION_TABLE = "theme_lifecycle_transitions"
LIFECYCLE_INDEX = "idx_theme_clusters_lifecycle_state"
TRANSITION_IDX = "idx_theme_lifecycle_transitions_theme_time"


def migrate_theme_lifecycle(engine) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "columns_added": [],
        "clusters_backfilled": 0,
        "transition_table_created": False,
        "indexes_ensured": [],
    }
    with engine.connect() as conn:
        columns = _get_table_columns(conn, THEME_TABLE)
        timestamp_type = sql_timestamp_type(conn)
        add_statements = {
            "lifecycle_state": "ALTER TABLE theme_clusters ADD COLUMN lifecycle_state TEXT",
            "lifecycle_state_updated_at": f"ALTER TABLE theme_clusters ADD COLUMN lifecycle_state_updated_at {timestamp_type}",
            "lifecycle_state_metadata": "ALTER TABLE theme_clusters ADD COLUMN lifecycle_state_metadata JSON",
            "candidate_since_at": f"ALTER TABLE theme_clusters ADD COLUMN candidate_since_at {timestamp_type}",
            "activated_at": f"ALTER TABLE theme_clusters ADD COLUMN activated_at {timestamp_type}",
            "dormant_at": f"ALTER TABLE theme_clusters ADD COLUMN dormant_at {timestamp_type}",
            "reactivated_at": f"ALTER TABLE theme_clusters ADD COLUMN reactivated_at {timestamp_type}",
            "retired_at": f"ALTER TABLE theme_clusters ADD COLUMN retired_at {timestamp_type}",
        }

        for name, statement in add_statements.items():
            if name in columns:
                continue
            conn.execute(text(statement))
            stats["columns_added"].append(name)

        result = conn.execute(
            text(
                """
                UPDATE theme_clusters
                SET lifecycle_state = CASE
                    WHEN is_active = 0 THEN 'retired'
                    ELSE 'active'
                END,
                    lifecycle_state_updated_at = COALESCE(lifecycle_state_updated_at, updated_at, created_at, CURRENT_TIMESTAMP),
                    candidate_since_at = COALESCE(candidate_since_at, first_seen_at, created_at, CURRENT_TIMESTAMP),
                    activated_at = CASE
                        WHEN is_active = 1 THEN COALESCE(activated_at, first_seen_at, created_at, CURRENT_TIMESTAMP)
                        ELSE activated_at
                    END,
                    retired_at = CASE
                        WHEN is_active = 0 THEN COALESCE(retired_at, updated_at, last_seen_at, CURRENT_TIMESTAMP)
                        ELSE retired_at
                    END
                WHERE lifecycle_state IS NULL OR lifecycle_state = ''
                """
            )
        )
        stats["clusters_backfilled"] = int(result.rowcount or 0)

        invalid_result = conn.execute(
            text(
                """
                UPDATE theme_clusters
                SET lifecycle_state = CASE
                    WHEN is_active = 0 THEN 'retired'
                    ELSE 'active'
                END,
                    lifecycle_state_updated_at = COALESCE(lifecycle_state_updated_at, updated_at, created_at, CURRENT_TIMESTAMP),
                    candidate_since_at = COALESCE(candidate_since_at, first_seen_at, created_at, CURRENT_TIMESTAMP),
                    activated_at = CASE
                        WHEN is_active = 1 THEN COALESCE(activated_at, first_seen_at, created_at, CURRENT_TIMESTAMP)
                        ELSE activated_at
                    END,
                    retired_at = CASE
                        WHEN is_active = 0 THEN COALESCE(retired_at, updated_at, last_seen_at, CURRENT_TIMESTAMP)
                        ELSE retired_at
                    END
                WHERE lifecycle_state NOT IN ('candidate', 'active', 'dormant', 'reactivated', 'retired')
                """
            )
        )
        stats["clusters_backfilled"] += int(invalid_result.rowcount or 0)

        ThemeLifecycleTransition.__table__.create(bind=conn, checkfirst=True)
        if TRANSITION_TABLE in _get_existing_tables(conn):
            stats["transition_table_created"] = True

        conn.execute(text(f"CREATE INDEX IF NOT EXISTS {LIFECYCLE_INDEX} ON theme_clusters(lifecycle_state)"))
        conn.execute(
            text(
                f"""
                CREATE INDEX IF NOT EXISTS {TRANSITION_IDX}
                ON theme_lifecycle_transitions(theme_cluster_id, transitioned_at DESC)
                """
            )
        )
        stats["indexes_ensured"] = [LIFECYCLE_INDEX, TRANSITION_IDX]

        conn.commit()
    logger.info("Theme lifecycle migration completed: %s", stats)
    return stats


def _get_existing_tables(conn) -> set[str]:
    return table_names(conn)


def _get_table_columns(conn, table_name: str) -> set[str]:
    return column_names(conn, table_name)
