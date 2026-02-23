"""Tests for content_item_pipeline_state idempotent migration and verification."""

from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from app.db_migrations.theme_pipeline_state_migration import (
    EXPECTED_INDEXES,
    migrate_theme_pipeline_state,
    verify_theme_pipeline_state_schema,
)


def _bootstrap_content_items(conn) -> None:
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS content_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT
            )
            """
        )
    )
    conn.commit()


def test_migration_creates_table_columns_and_indexes():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        _bootstrap_content_items(conn)

    result = migrate_theme_pipeline_state(engine)
    assert result["table_created"] is True
    assert set(result["indexes_ensured"]) == EXPECTED_INDEXES

    verification = verify_theme_pipeline_state_schema(engine)
    assert verification["ok"] is True
    assert verification["missing_columns"] == []
    assert verification["missing_indexes"] == []
    assert verification["duplicate_rows"] == 0
    assert verification["invalid_status_rows"] == 0


def test_migration_is_idempotent_on_rerun():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        _bootstrap_content_items(conn)

    first = migrate_theme_pipeline_state(engine)
    second = migrate_theme_pipeline_state(engine)

    assert first["table_created"] is True
    assert second["table_created"] is False
    assert second["columns_added"] == []

    verification = verify_theme_pipeline_state_schema(engine)
    assert verification["ok"] is True


def test_unique_content_item_pipeline_index_enforced():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        _bootstrap_content_items(conn)
        conn.execute(text("INSERT INTO content_items DEFAULT VALUES"))
        conn.commit()

    migrate_theme_pipeline_state(engine)

    with engine.connect() as conn:
        conn.execute(
            text(
                """
                INSERT INTO content_item_pipeline_state(content_item_id, pipeline, status)
                VALUES (1, 'technical', 'pending')
                """
            )
        )
        conn.commit()

        try:
            conn.execute(
                text(
                    """
                    INSERT INTO content_item_pipeline_state(content_item_id, pipeline, status)
                    VALUES (1, 'technical', 'pending')
                    """
                )
            )
            conn.commit()
            assert False, "Expected unique index violation for duplicate content_item_id + pipeline"
        except IntegrityError:
            conn.rollback()


def test_verification_fails_when_table_missing():
    engine = create_engine("sqlite:///:memory:")
    verification = verify_theme_pipeline_state_schema(engine)
    assert verification["ok"] is False
    assert verification["table_exists"] is False

