"""Run the pre-Alembic schema reconciliation steps exactly once."""

from __future__ import annotations

import logging

from app.database import engine
from app.infra.db.legacy_runtime_migrations import reconcile_legacy_runtime_schema

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    results = reconcile_legacy_runtime_schema(engine)
    ran = [result["name"] for result in results if not result["skipped"]]
    skipped = [result["name"] for result in results if result["skipped"]]
    logger.info("Legacy runtime migrations completed")
    logger.info("Applied steps: %s", ", ".join(ran) if ran else "none")
    logger.info("Skipped steps: %s", ", ".join(skipped) if skipped else "none")


if __name__ == "__main__":
    main()
