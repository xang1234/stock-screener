"""Idempotent migration for pipeline metadata on theme_pipeline_runs."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text

from ..infra.db.portability import column_names

logger = logging.getLogger(__name__)

TABLE_NAME = "theme_pipeline_runs"
COLUMN_NAME = "pipeline"
INDEX_NAME = "idx_theme_pipeline_runs_pipeline"


def migrate_theme_pipeline_run_schema(engine) -> dict[str, Any]:
    """Add the pipeline column/index to theme_pipeline_runs idempotently."""
    stats: dict[str, Any] = {
        "column_added": False,
        "index_ensured": False,
    }

    with engine.connect() as conn:
        columns = _get_table_columns(conn)
        if COLUMN_NAME not in columns:
            conn.execute(
                text(
                    """
                    ALTER TABLE theme_pipeline_runs
                    ADD COLUMN pipeline TEXT
                    """
                )
            )
            stats["column_added"] = True

        conn.execute(
            text(
                f"""
                CREATE INDEX IF NOT EXISTS {INDEX_NAME}
                ON {TABLE_NAME}(pipeline)
                """
            )
        )
        stats["index_ensured"] = True
        conn.commit()

    return stats


def _get_table_columns(conn) -> set[str]:
    return column_names(conn, TABLE_NAME)
