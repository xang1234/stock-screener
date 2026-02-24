"""Tests for theme_embeddings freshness migration and verification."""

from __future__ import annotations

from sqlalchemy import create_engine, text

from app.db_migrations.theme_embedding_freshness_migration import (
    EXPECTED_INDEXES,
    migrate_theme_embedding_freshness,
    verify_theme_embedding_freshness_schema,
)


def _bootstrap_legacy_theme_embeddings(conn) -> None:
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS theme_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                theme_cluster_id INTEGER NOT NULL UNIQUE,
                embedding TEXT NOT NULL,
                embedding_model TEXT DEFAULT 'all-MiniLM-L6-v2',
                embedding_text TEXT,
                created_at DATETIME DEFAULT (CURRENT_TIMESTAMP),
                updated_at DATETIME DEFAULT (CURRENT_TIMESTAMP)
            )
            """
        )
    )
    conn.commit()


def test_migration_adds_freshness_columns_indexes_and_marks_legacy_rows_stale():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        _bootstrap_legacy_theme_embeddings(conn)
        conn.execute(
            text(
                """
                INSERT INTO theme_embeddings(theme_cluster_id, embedding, embedding_model, embedding_text)
                VALUES (101, '[0.1, 0.2]', 'all-MiniLM-L6-v2', 'AI Infrastructure')
                """
            )
        )
        conn.commit()

    result = migrate_theme_embedding_freshness(engine)
    assert result["table_exists"] is True
    assert set(result["columns_added"]) == {"content_hash", "model_version", "is_stale"}
    assert set(result["indexes_ensured"]) == EXPECTED_INDEXES
    assert result["rows_marked_stale"] == 1

    verification = verify_theme_embedding_freshness_schema(engine)
    assert verification["ok"] is True
    assert verification["missing_columns"] == []
    assert verification["missing_indexes"] == []

    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT model_version, is_stale, content_hash
                FROM theme_embeddings
                WHERE theme_cluster_id = 101
                """
            )
        ).first()
    assert row is not None
    assert row[0] == "embedding-v1"
    assert row[1] == 1
    assert row[2] is None


def test_migration_is_idempotent():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        _bootstrap_legacy_theme_embeddings(conn)
        conn.execute(
            text(
                """
                INSERT INTO theme_embeddings(theme_cluster_id, embedding, embedding_model, embedding_text)
                VALUES (202, '[0.3, 0.4]', 'all-MiniLM-L6-v2', 'Nuclear Energy')
                """
            )
        )
        conn.commit()

    first = migrate_theme_embedding_freshness(engine)
    second = migrate_theme_embedding_freshness(engine)

    assert first["columns_added"] == ["content_hash", "model_version", "is_stale"]
    assert second["columns_added"] == []
    assert second["rows_marked_stale"] == 0
