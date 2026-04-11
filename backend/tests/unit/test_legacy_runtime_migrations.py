"""Focused tests for legacy runtime schema reconciliation helpers."""

from __future__ import annotations

from sqlalchemy import create_engine, text

from app.db_migrations.theme_cluster_identity_migration import verify_theme_cluster_identity_schema
from app.db_migrations.theme_pipeline_state_migration import verify_theme_pipeline_state_schema
from app.db_migrations.universe_migration import _ensure_indexes, migrate_scan_universe_schema_and_backfill
from app.infra.db.legacy_runtime_migrations import (
    _THEME_MERGE_SAFETY_REQUIRED_COLUMNS,
    _THEME_MERGE_SAFETY_REQUIRED_INDEXES,
    _verify_columns_and_indexes,
    _verify_theme_lifecycle_schema,
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
                    universe_market TEXT,
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
        "idx_scans_universe_market",
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
                    universe_market TEXT,
                    universe_exchange TEXT,
                    universe_index TEXT,
                    universe_symbols TEXT
                )
                """
            )
        )
        conn.execute(text("CREATE INDEX custom_key_idx ON scans(universe_key)"))
        conn.execute(text("CREATE INDEX custom_type_idx ON scans(universe_type)"))
        conn.execute(text("CREATE INDEX custom_market_idx ON scans(universe_market)"))
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
    assert "idx_scans_universe_market" not in indexes
    assert "idx_scans_universe_exchange" not in indexes
    assert "idx_scans_universe_index" not in indexes
    engine.dispose()


def test_verify_theme_lifecycle_schema_accepts_equivalent_index_shapes():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE theme_clusters (
                    id INTEGER PRIMARY KEY,
                    lifecycle_state TEXT,
                    lifecycle_state_updated_at TEXT,
                    lifecycle_state_metadata TEXT,
                    candidate_since_at TEXT,
                    activated_at TEXT,
                    dormant_at TEXT,
                    reactivated_at TEXT,
                    retired_at TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX custom_theme_lifecycle_state
                ON theme_clusters(lifecycle_state)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE theme_lifecycle_transitions (
                    id INTEGER PRIMARY KEY,
                    theme_cluster_id INTEGER,
                    transitioned_at TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX custom_theme_transition_lookup
                ON theme_lifecycle_transitions(theme_cluster_id, transitioned_at)
                """
            )
        )

    verification = _verify_theme_lifecycle_schema(engine)

    assert verification["ok"] is True
    assert verification["theme_clusters"]["missing_indexes"] == []
    assert verification["theme_lifecycle_transitions"]["missing_indexes"] == []
    engine.dispose()


def test_verify_theme_pipeline_state_schema_counts_duplicate_rows():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE content_item_pipeline_state (
                    id INTEGER PRIMARY KEY,
                    content_item_id INTEGER NOT NULL,
                    pipeline TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    error_code TEXT,
                    error_message TEXT,
                    last_attempt_at TEXT,
                    processed_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX idx_cips_pipeline_status_last_attempt
                ON content_item_pipeline_state(pipeline, status, last_attempt_at)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX idx_cips_pipeline_status_created
                ON content_item_pipeline_state(pipeline, status, created_at)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX idx_cips_content_item_pipeline_status
                ON content_item_pipeline_state(content_item_id, pipeline, status)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX idx_cips_error_code
                ON content_item_pipeline_state(error_code)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX idx_cips_updated_at
                ON content_item_pipeline_state(updated_at)
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO content_item_pipeline_state
                    (id, content_item_id, pipeline, status, attempt_count, created_at, updated_at)
                VALUES
                    (1, 42, 'technical', 'pending', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
                    (2, 42, 'fundamental', 'pending', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
                    (3, 42, 'fundamental', 'pending', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """
            )
        )

    verification = verify_theme_pipeline_state_schema(engine)

    assert verification["duplicate_rows"] == 1
    engine.dispose()


def test_verify_theme_cluster_identity_schema_counts_duplicate_pipeline_keys():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE theme_clusters (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    canonical_key TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    aliases TEXT,
                    description TEXT,
                    pipeline TEXT NOT NULL,
                    category TEXT,
                    is_emerging INTEGER,
                    first_seen_at TEXT,
                    last_seen_at TEXT,
                    discovery_source TEXT,
                    is_active INTEGER,
                    is_validated INTEGER,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO theme_clusters
                    (id, name, canonical_key, display_name, pipeline, created_at, updated_at)
                VALUES
                    (1, 'AI Infra', 'ai-infra', 'AI Infra', 'technical', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
                    (2, 'AI Infra 2', 'ai-infra', 'AI Infra', 'fundamental', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
                    (3, 'AI Infra 3', 'ai-infra', 'AI Infra', 'fundamental', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """
            )
        )

    verification = verify_theme_cluster_identity_schema(engine)

    assert verification["duplicate_pipeline_keys"] == 1
    engine.dispose()
