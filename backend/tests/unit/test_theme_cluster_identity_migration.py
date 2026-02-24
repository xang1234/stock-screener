"""Tests for theme_clusters canonical identity migration."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from app.db_migrations.theme_cluster_identity_migration import (
    migrate_theme_cluster_identity,
    verify_theme_cluster_identity_schema,
)


def _create_legacy_theme_clusters(conn) -> None:
    conn.execute(
        text(
            """
            CREATE TABLE theme_clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
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
    conn.commit()


def test_migration_creates_theme_clusters_with_pipeline_scoped_identity():
    engine = create_engine("sqlite:///:memory:")

    result = migrate_theme_cluster_identity(engine)
    assert result["table_created"] is True

    verification = verify_theme_cluster_identity_schema(engine)
    assert verification["ok"] is True
    assert verification["has_pipeline_key_unique"] is True
    assert verification["has_global_name_unique"] is False


def test_migration_rebuilds_legacy_table_and_backfills_identity_columns():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        _create_legacy_theme_clusters(conn)
        conn.execute(
            text(
                """
                INSERT INTO theme_clusters(name, pipeline, description)
                VALUES ('AI Infrastructure', 'technical', 'legacy row')
                """
            )
        )
        conn.commit()

    result = migrate_theme_cluster_identity(engine)
    assert result["table_rebuilt"] is True
    assert result["rows_backfilled"] == 1

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT name, canonical_key, display_name, pipeline FROM theme_clusters")
        ).fetchone()
        assert row[0] == "AI Infrastructure"
        assert row[1] == "ai_infrastructure"
        assert row[2] == "AI Infrastructure"
        assert row[3] == "technical"


def test_migration_is_idempotent():
    engine = create_engine("sqlite:///:memory:")

    first = migrate_theme_cluster_identity(engine)
    second = migrate_theme_cluster_identity(engine)

    assert first["table_created"] is True
    assert second["table_created"] is False
    assert second["table_rebuilt"] is False

    verification = verify_theme_cluster_identity_schema(engine)
    assert verification["ok"] is True


def test_unique_enforced_for_pipeline_and_canonical_key_only():
    engine = create_engine("sqlite:///:memory:")
    migrate_theme_cluster_identity(engine)

    with engine.connect() as conn:
        conn.execute(
            text(
                """
                INSERT INTO theme_clusters(name, canonical_key, display_name, pipeline)
                VALUES
                ('AI Infrastructure', 'ai_infrastructure', 'AI Infrastructure', 'technical'),
                ('AI Infrastructure', 'ai_infrastructure', 'AI Infrastructure', 'fundamental')
                """
            )
        )
        conn.commit()

        try:
            conn.execute(
                text(
                    """
                    INSERT INTO theme_clusters(name, canonical_key, display_name, pipeline)
                    VALUES ('AI Infra', 'ai_infrastructure', 'AI Infra', 'technical')
                    """
                )
            )
            conn.commit()
            assert False, "Expected unique index violation for duplicate pipeline + canonical_key"
        except IntegrityError:
            conn.rollback()


def test_migration_fails_on_legacy_duplicates_that_collapse_to_same_canonical_key():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        _create_legacy_theme_clusters(conn)
        conn.execute(
            text(
                """
                INSERT INTO theme_clusters(name, pipeline)
                VALUES ('GLP-1', 'technical'),
                       ('GLP1', 'technical')
                """
            )
        )
        conn.commit()

    with pytest.raises(ValueError, match="duplicate canonical_key per pipeline"):
        migrate_theme_cluster_identity(engine)
