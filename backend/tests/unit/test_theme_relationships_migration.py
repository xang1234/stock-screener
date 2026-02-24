"""Tests for theme_relationships idempotent migration and verification."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from app.db_migrations.theme_relationships_migration import (
    migrate_theme_relationships,
    verify_theme_relationships_schema,
)


def _create_theme_clusters(conn) -> None:
    conn.execute(
        text(
            """
            CREATE TABLE theme_clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                canonical_key TEXT NOT NULL,
                display_name TEXT NOT NULL,
                pipeline TEXT NOT NULL DEFAULT 'technical',
                is_active BOOLEAN DEFAULT 1
            )
            """
        )
    )
    conn.execute(
        text(
            """
            INSERT INTO theme_clusters(name, canonical_key, display_name, pipeline, is_active)
            VALUES
            ('AI Infrastructure', 'ai_infrastructure', 'AI Infrastructure', 'technical', 1),
            ('Datacenter Power', 'datacenter_power', 'Datacenter Power', 'technical', 1)
            """
        )
    )
    conn.commit()


def test_theme_relationships_migration_creates_table_and_indexes():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        _create_theme_clusters(conn)

    result = migrate_theme_relationships(engine)
    assert result["table_created"] is True
    assert "idx_theme_relationship_source_active" in result["indexes_ensured"]
    assert "idx_theme_relationship_target_active" in result["indexes_ensured"]

    verification = verify_theme_relationships_schema(engine)
    assert verification["ok"] is True


def test_theme_relationships_migration_is_idempotent():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        _create_theme_clusters(conn)

    first = migrate_theme_relationships(engine)
    second = migrate_theme_relationships(engine)
    assert first["table_created"] is True
    assert second["table_created"] is False
    assert second["columns_added"] == []


def test_theme_relationships_migration_removes_self_edges_and_normalizes_type():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        _create_theme_clusters(conn)
        conn.execute(
            text(
                """
                CREATE TABLE theme_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_cluster_id INTEGER NOT NULL,
                    target_cluster_id INTEGER NOT NULL,
                    pipeline TEXT NOT NULL DEFAULT 'technical',
                    relationship_type TEXT NOT NULL,
                    confidence FLOAT NOT NULL DEFAULT 0.5,
                    provenance TEXT NOT NULL DEFAULT 'legacy',
                    evidence JSON,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    created_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP),
                    updated_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO theme_relationships(
                    source_cluster_id, target_cluster_id, pipeline, relationship_type, confidence
                )
                VALUES
                (1, 1, 'technical', 'subset', 0.9),
                (1, 2, 'technical', 'unexpected', 0.7)
                """
            )
        )
        conn.commit()

    result = migrate_theme_relationships(engine)
    assert result["self_edges_removed"] == 1
    assert result["rows_normalized"] >= 1

    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT source_cluster_id, target_cluster_id, relationship_type FROM theme_relationships")
        ).fetchall()
    assert rows == [(1, 2, "related")]


def test_theme_relationships_migration_enforces_constraints_with_triggers():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        _create_theme_clusters(conn)

    migrate_theme_relationships(engine)

    with engine.connect() as conn:
        with pytest.raises(Exception):
            conn.execute(
                text(
                    """
                    INSERT INTO theme_relationships(
                        source_cluster_id, target_cluster_id, pipeline, relationship_type, confidence
                    )
                    VALUES (1, 1, 'technical', 'subset', 0.9)
                    """
                )
            )

        with pytest.raises(Exception):
            conn.execute(
                text(
                    """
                    INSERT INTO theme_relationships(
                        source_cluster_id, target_cluster_id, pipeline, relationship_type, confidence
                    )
                    VALUES (1, 2, 'technical', 'invalid_type', 0.9)
                    """
                )
            )
