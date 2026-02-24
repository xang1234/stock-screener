"""Tests for theme_aliases idempotent migration and verification."""

from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from app.db_migrations.theme_aliases_migration import (
    EXPECTED_INDEXES,
    migrate_theme_aliases,
    verify_theme_aliases_schema,
)


def _bootstrap_theme_clusters(conn) -> None:
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS theme_clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT
            )
            """
        )
    )
    conn.commit()


def test_migration_creates_table_columns_and_indexes():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        _bootstrap_theme_clusters(conn)

    result = migrate_theme_aliases(engine)
    assert result["table_created"] is True
    assert set(result["indexes_ensured"]) == EXPECTED_INDEXES

    verification = verify_theme_aliases_schema(engine)
    assert verification["ok"] is True
    assert verification["missing_columns"] == []
    assert verification["missing_indexes"] == []


def test_migration_is_idempotent_on_rerun():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        _bootstrap_theme_clusters(conn)

    first = migrate_theme_aliases(engine)
    second = migrate_theme_aliases(engine)

    assert first["table_created"] is True
    assert second["table_created"] is False
    assert second["columns_added"] == []

    verification = verify_theme_aliases_schema(engine)
    assert verification["ok"] is True


def test_unique_pipeline_alias_key_index_enforced():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        _bootstrap_theme_clusters(conn)
        conn.execute(text("INSERT INTO theme_clusters DEFAULT VALUES"))
        conn.commit()

    migrate_theme_aliases(engine)

    with engine.connect() as conn:
        conn.execute(
            text(
                """
                INSERT INTO theme_aliases(theme_cluster_id, pipeline, alias_text, alias_key)
                VALUES (1, 'technical', 'AI Infrastructure', 'ai_infrastructure')
                """
            )
        )
        conn.commit()

        try:
            conn.execute(
                text(
                    """
                    INSERT INTO theme_aliases(theme_cluster_id, pipeline, alias_text, alias_key)
                    VALUES (1, 'technical', 'A.I. Infrastructure', 'ai_infrastructure')
                    """
                )
            )
            conn.commit()
            assert False, "Expected unique index violation for duplicate pipeline + alias_key"
        except IntegrityError:
            conn.rollback()

