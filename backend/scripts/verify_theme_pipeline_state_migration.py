#!/usr/bin/env python3
"""Verify content_item_pipeline_state migration invariants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure backend module root is importable when script is run directly.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.database import engine
from app.db_migrations.theme_pipeline_state_migration import (
    migrate_theme_pipeline_state,
    verify_theme_pipeline_state_schema,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify content_item_pipeline_state schema, indexes, and basic invariants."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit verification result as JSON",
    )
    parser.add_argument(
        "--apply-migration",
        action="store_true",
        help="Run migration before verification",
    )
    args = parser.parse_args()

    if args.apply_migration:
        migrate_theme_pipeline_state(engine)

    result = verify_theme_pipeline_state_schema(engine)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("Theme pipeline-state migration verification")
        print(f"ok: {result['ok']}")
        print(f"table_exists: {result['table_exists']}")
        print(f"missing_columns: {result['missing_columns']}")
        print(f"missing_indexes: {result['missing_indexes']}")
        print(f"status_check_present: {result['status_check_present']}")
        print(f"pipeline_check_present: {result['pipeline_check_present']}")
        print(f"duplicate_rows: {result['duplicate_rows']}")
        print(f"invalid_status_rows: {result['invalid_status_rows']}")

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
