"""Run the pre-Alembic schema reconciliation steps exactly once."""

from __future__ import annotations

import logging

from app.database import engine
from app.db_migrations.theme_cluster_identity_migration import migrate_theme_cluster_identity
from app.db_migrations.theme_lifecycle_migration import migrate_theme_lifecycle
from app.db_migrations.theme_merge_suggestion_safety_migration import migrate_theme_merge_suggestion_safety
from app.db_migrations.theme_pipeline_state_migration import migrate_theme_pipeline_state
from app.db_migrations.theme_relationships_migration import migrate_theme_relationships
from app.db_migrations.universe_lifecycle_migration import migrate_universe_lifecycle
from app.db_migrations.universe_migration import migrate_scan_universe_schema_and_backfill

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    migrations = [
        ("universe schema backfill", migrate_scan_universe_schema_and_backfill),
        ("theme pipeline state", migrate_theme_pipeline_state),
        ("theme cluster identity", migrate_theme_cluster_identity),
        ("theme lifecycle", migrate_theme_lifecycle),
        ("theme relationships", migrate_theme_relationships),
        ("theme merge suggestion safety", migrate_theme_merge_suggestion_safety),
        ("universe lifecycle", migrate_universe_lifecycle),
    ]
    for name, fn in migrations:
        logger.info("Running legacy migration: %s", name)
        fn(engine)
    logger.info("Legacy runtime migrations completed")


if __name__ == "__main__":
    main()
