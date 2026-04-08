"""Focused tests for legacy runtime schema reconciliation helpers."""

from __future__ import annotations

from sqlalchemy import create_engine, text

from app.db_migrations.universe_migration import _ensure_indexes, migrate_scan_universe_schema_and_backfill
from app.infra.db.legacy_runtime_migrations import (
    _THEME_MERGE_SAFETY_REQUIRED_COLUMNS,
    _THEME_MERGE_SAFETY_REQUIRED_INDEXES,
    _verify_columns_and_indexes,
)
from app.infra.db.portability import index_defs


def test_verify_columns_and_indexes_accepts_equivalent_index_shapes_with_custom_names():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE theme_merge_suggestions (
                    id INTEGER PRIMARY KEY,
                    pair_min_cluster_id INTEGER,
                    pair_max_cluster_id INTEGER,
                    approval_idempotency_key TEXT,
                    approval_result_json TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE UNIQUE INDEX custom_pair_lookup
                ON theme_merge_suggestions(pair_min_cluster_id, pair_max_cluster_id)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX custom_approval_lookup
                ON theme_merge_suggestions(approval_idempotency_key)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX custom_pair_min_lookup
                ON theme_merge_suggestions(pair_min_cluster_id)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX custom_pair_max_lookup
                ON theme_merge_suggestions(pair_max_cluster_id)
                """
            )
        )

    verification = _verify_columns_and_indexes(
        engine,
        table_name="theme_merge_suggestions",
        required_columns=_THEME_MERGE_SAFETY_REQUIRED_COLUMNS,
        required_indexes=_THEME_MERGE_SAFETY_REQUIRED_INDEXES,
    )

    assert verification["ok"] is True
    assert verification["missing_indexes"] == []
    engine.dispose()


def test_universe_migration_ensures_indexes_for_partially_migrated_scans_schema():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE scans (
                    id INTEGER PRIMARY KEY,
                    scan_id TEXT,
                    universe TEXT,
                    universe_key TEXT,
                    universe_type TEXT,
                    universe_exchange TEXT,
                    universe_index TEXT,
                    universe_symbols TEXT
                )
                """
            )
        )

    migrate_scan_universe_schema_and_backfill(engine)

    with engine.connect() as conn:
        indexes = {
            index["name"]
            for index in index_defs(conn, "scans")
            if index.get("name")
        }

    assert {
        "idx_scans_universe_key",
        "idx_scans_universe_type",
        "idx_scans_universe_exchange",
        "idx_scans_universe_index",
    }.issubset(indexes)
    engine.dispose()


def test_universe_index_ensure_reuses_equivalent_existing_indexes():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE scans (
                    id INTEGER PRIMARY KEY,
                    scan_id TEXT,
                    universe TEXT,
                    universe_key TEXT,
                    universe_type TEXT,
                    universe_exchange TEXT,
                    universe_index TEXT,
                    universe_symbols TEXT
                )
                """
            )
        )
        conn.execute(text("CREATE INDEX custom_key_idx ON scans(universe_key)"))
        conn.execute(text("CREATE INDEX custom_type_idx ON scans(universe_type)"))
        conn.execute(text("CREATE INDEX custom_exchange_idx ON scans(universe_exchange)"))
        conn.execute(text("CREATE INDEX custom_index_idx ON scans(universe_index)"))

        created = _ensure_indexes(conn)

    with engine.connect() as conn:
        indexes = {
            index["name"]
            for index in index_defs(conn, "scans")
            if index.get("name")
        }

    assert created == 0
    assert "idx_scans_universe_key" not in indexes
    assert "idx_scans_universe_type" not in indexes
    assert "idx_scans_universe_exchange" not in indexes
    assert "idx_scans_universe_index" not in indexes
    engine.dispose()
