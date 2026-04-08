"""Shared legacy runtime schema reconciliation helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from sqlalchemy.engine import Engine

from ...db_migrations.theme_cluster_identity_migration import (
    migrate_theme_cluster_identity,
    verify_theme_cluster_identity_schema,
)
from ...db_migrations.theme_lifecycle_migration import (
    LIFECYCLE_INDEX,
    THEME_TABLE,
    TRANSITION_IDX,
    TRANSITION_TABLE,
    migrate_theme_lifecycle,
)
from ...db_migrations.theme_merge_suggestion_safety_migration import (
    TABLE_NAME as THEME_MERGE_SUGGESTIONS_TABLE,
    migrate_theme_merge_suggestion_safety,
)
from ...db_migrations.theme_pipeline_state_migration import (
    migrate_theme_pipeline_state,
    verify_theme_pipeline_state_schema,
)
from ...db_migrations.theme_relationships_migration import (
    TABLE_NAME as THEME_RELATIONSHIPS_TABLE,
    migrate_theme_relationships,
    verify_theme_relationships_schema,
)
from ...db_migrations.universe_lifecycle_migration import migrate_universe_lifecycle
from ...db_migrations.universe_migration import migrate_scan_universe_schema_and_backfill
from .portability import column_names, index_names, table_names

logger = logging.getLogger(__name__)

_SCANS_TABLE = "scans"
_SCANS_REQUIRED_COLUMNS = {
    "universe_key",
    "universe_type",
    "universe_exchange",
    "universe_index",
    "universe_symbols",
}
_SCANS_REQUIRED_INDEXES = {
    "idx_scans_universe_key",
    "idx_scans_universe_type",
    "idx_scans_universe_exchange",
    "idx_scans_universe_index",
}

_THEME_LIFECYCLE_REQUIRED_COLUMNS = {
    "lifecycle_state",
    "lifecycle_state_updated_at",
    "lifecycle_state_metadata",
    "candidate_since_at",
    "activated_at",
    "dormant_at",
    "reactivated_at",
    "retired_at",
}

_THEME_MERGE_SAFETY_REQUIRED_COLUMNS = {
    "pair_min_cluster_id",
    "pair_max_cluster_id",
    "approval_idempotency_key",
    "approval_result_json",
}
_THEME_MERGE_SAFETY_REQUIRED_INDEXES = {
    "uix_merge_suggestion_pair_canonical",
    "ix_theme_merge_suggestions_approval_idempotency_key",
    "ix_theme_merge_suggestions_pair_min_cluster_id",
    "ix_theme_merge_suggestions_pair_max_cluster_id",
}

_UNIVERSE_LIFECYCLE_TABLE = "stock_universe"
_UNIVERSE_LIFECYCLE_REQUIRED_COLUMNS = {
    "status",
    "status_reason",
    "first_seen_at",
    "last_seen_in_source_at",
    "deactivated_at",
    "consecutive_fetch_failures",
    "last_fetch_success_at",
    "last_fetch_failure_at",
}
_UNIVERSE_LIFECYCLE_REQUIRED_INDEXES = {
    "idx_stock_universe_exchange_status",
    "idx_stock_universe_status_active",
}
_UNIVERSE_STATUS_EVENT_INDEXES = {
    "idx_stock_universe_status_events_symbol_created",
    "idx_stock_universe_status_events_status_created",
}


class LegacySchemaVerificationError(RuntimeError):
    """Raised when a legacy schema reconciliation step still fails verification."""

    def __init__(self, step_name: str, verification: dict[str, Any]):
        super().__init__(f"Legacy schema verification failed for '{step_name}': {verification}")
        self.step_name = step_name
        self.verification = verification


@dataclass(frozen=True)
class LegacyMigrationStep:
    """One idempotent legacy migration with an applicability predicate."""

    name: str
    applies_when: Callable[[set[str]], bool]
    migrate: Callable[[Engine], Any]
    verify: Callable[[Engine], dict[str, Any]]


_LEGACY_MIGRATION_STEPS = (
    LegacyMigrationStep(
        name="universe schema backfill",
        applies_when=lambda tables: _SCANS_TABLE in tables,
        migrate=migrate_scan_universe_schema_and_backfill,
        verify=lambda engine: _verify_columns_and_indexes(
            engine,
            table_name=_SCANS_TABLE,
            required_columns=_SCANS_REQUIRED_COLUMNS,
            required_indexes=_SCANS_REQUIRED_INDEXES,
        ),
    ),
    LegacyMigrationStep(
        name="theme pipeline state",
        applies_when=lambda tables: "content_items" in tables or "content_item_pipeline_state" in tables,
        migrate=migrate_theme_pipeline_state,
        verify=verify_theme_pipeline_state_schema,
    ),
    LegacyMigrationStep(
        name="theme cluster identity",
        applies_when=lambda tables: THEME_TABLE in tables,
        migrate=migrate_theme_cluster_identity,
        verify=verify_theme_cluster_identity_schema,
    ),
    LegacyMigrationStep(
        name="theme lifecycle",
        applies_when=lambda tables: THEME_TABLE in tables or TRANSITION_TABLE in tables,
        migrate=migrate_theme_lifecycle,
        verify=lambda engine: _verify_theme_lifecycle_schema(engine),
    ),
    LegacyMigrationStep(
        name="theme relationships",
        applies_when=lambda tables: THEME_TABLE in tables or THEME_RELATIONSHIPS_TABLE in tables,
        migrate=migrate_theme_relationships,
        verify=verify_theme_relationships_schema,
    ),
    LegacyMigrationStep(
        name="theme merge suggestion safety",
        applies_when=lambda tables: THEME_MERGE_SUGGESTIONS_TABLE in tables,
        migrate=migrate_theme_merge_suggestion_safety,
        verify=lambda engine: _verify_columns_and_indexes(
            engine,
            table_name=THEME_MERGE_SUGGESTIONS_TABLE,
            required_columns=_THEME_MERGE_SAFETY_REQUIRED_COLUMNS,
            required_indexes=_THEME_MERGE_SAFETY_REQUIRED_INDEXES,
        ),
    ),
    LegacyMigrationStep(
        name="universe lifecycle",
        applies_when=lambda tables: _UNIVERSE_LIFECYCLE_TABLE in tables,
        migrate=migrate_universe_lifecycle,
        verify=lambda engine: _verify_universe_lifecycle_schema(engine),
    ),
)


def reconcile_legacy_runtime_schema(engine: Engine) -> list[dict[str, Any]]:
    """Run and verify the legacy schema deltas for pre-Alembic deployments."""
    with engine.connect() as conn:
        existing_tables = table_names(conn)

    results: list[dict[str, Any]] = []
    for step in _LEGACY_MIGRATION_STEPS:
        if not step.applies_when(existing_tables):
            logger.info("Skipping legacy migration %s; supporting tables are absent", step.name)
            results.append({"name": step.name, "skipped": True})
            continue

        logger.info("Running legacy migration: %s", step.name)
        migration_result = step.migrate(engine)
        verification = step.verify(engine)
        if not verification.get("ok", False):
            raise LegacySchemaVerificationError(step.name, verification)

        results.append(
            {
                "name": step.name,
                "skipped": False,
                "migration_result": migration_result,
                "verification": verification,
            }
        )

    return results


def _verify_columns_and_indexes(
    engine: Engine,
    *,
    table_name: str,
    required_columns: set[str],
    required_indexes: set[str],
) -> dict[str, Any]:
    with engine.connect() as conn:
        tables = table_names(conn)
        table_exists = table_name in tables
        columns = column_names(conn, table_name) if table_exists else set()
        indexes = index_names(conn, table_name) if table_exists else set()

    missing_columns = sorted(required_columns - columns)
    missing_indexes = sorted(required_indexes - indexes)
    return {
        "table_exists": table_exists,
        "missing_columns": missing_columns,
        "missing_indexes": missing_indexes,
        "ok": table_exists and not missing_columns and not missing_indexes,
    }


def _verify_theme_lifecycle_schema(engine: Engine) -> dict[str, Any]:
    base = _verify_columns_and_indexes(
        engine,
        table_name=THEME_TABLE,
        required_columns=_THEME_LIFECYCLE_REQUIRED_COLUMNS,
        required_indexes={LIFECYCLE_INDEX},
    )
    transition = _verify_columns_and_indexes(
        engine,
        table_name=TRANSITION_TABLE,
        required_columns=set(),
        required_indexes={TRANSITION_IDX},
    )
    verification = {
        "theme_clusters": base,
        "theme_lifecycle_transitions": transition,
    }
    verification["ok"] = bool(base["ok"] and transition["ok"])
    return verification


def _verify_universe_lifecycle_schema(engine: Engine) -> dict[str, Any]:
    universe = _verify_columns_and_indexes(
        engine,
        table_name=_UNIVERSE_LIFECYCLE_TABLE,
        required_columns=_UNIVERSE_LIFECYCLE_REQUIRED_COLUMNS,
        required_indexes=_UNIVERSE_LIFECYCLE_REQUIRED_INDEXES,
    )
    events = _verify_columns_and_indexes(
        engine,
        table_name="stock_universe_status_events",
        required_columns=set(),
        required_indexes=_UNIVERSE_STATUS_EVENT_INDEXES,
    )
    verification = {
        "stock_universe": universe,
        "stock_universe_status_events": events,
    }
    verification["ok"] = bool(universe["ok"] and events["ok"])
    return verification
