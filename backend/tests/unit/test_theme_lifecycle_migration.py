"""Tests for theme lifecycle migration."""

from __future__ import annotations

from sqlalchemy import create_engine, text

from app.db_migrations.theme_lifecycle_migration import migrate_theme_lifecycle


def _create_pre_lifecycle_theme_clusters(conn) -> None:
    conn.execute(
        text(
            """
            CREATE TABLE theme_clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                canonical_key TEXT NOT NULL,
                display_name TEXT NOT NULL,
                aliases JSON,
                description TEXT,
                pipeline TEXT NOT NULL DEFAULT 'technical',
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


def test_theme_lifecycle_migration_adds_columns_and_backfills_states():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        _create_pre_lifecycle_theme_clusters(conn)
        conn.execute(
            text(
                """
                INSERT INTO theme_clusters(name, canonical_key, display_name, is_active)
                VALUES
                ('AI Infrastructure', 'ai_infrastructure', 'AI Infrastructure', 1),
                ('Legacy Theme', 'legacy_theme', 'Legacy Theme', 0)
                """
            )
        )
        conn.commit()

    result = migrate_theme_lifecycle(engine)
    assert "lifecycle_state" in result["columns_added"]
    assert result["clusters_backfilled"] == 2
    assert result["transition_table_created"] is True

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT canonical_key, lifecycle_state, candidate_since_at, lifecycle_state_updated_at, retired_at
                FROM theme_clusters
                ORDER BY id
                """
            )
        ).fetchall()

    assert rows[0][0] == "ai_infrastructure"
    assert rows[0][1] == "active"
    assert rows[0][2] is not None
    assert rows[0][3] is not None
    assert rows[1][0] == "legacy_theme"
    assert rows[1][1] == "retired"
    assert rows[1][4] is not None


def test_theme_lifecycle_migration_is_idempotent():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        _create_pre_lifecycle_theme_clusters(conn)
    first = migrate_theme_lifecycle(engine)
    second = migrate_theme_lifecycle(engine)

    assert first["transition_table_created"] is True
    assert second["clusters_backfilled"] == 0


def test_theme_lifecycle_migration_normalizes_invalid_existing_states():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        _create_pre_lifecycle_theme_clusters(conn)
        conn.execute(
            text(
                """
                ALTER TABLE theme_clusters ADD COLUMN lifecycle_state TEXT
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO theme_clusters(name, canonical_key, display_name, is_active, lifecycle_state)
                VALUES
                ('Bad Active', 'bad_active', 'Bad Active', 1, 'weird_state'),
                ('Bad Retired', 'bad_retired', 'Bad Retired', 0, 'unknown_state')
                """
            )
        )
        conn.commit()

    result = migrate_theme_lifecycle(engine)
    assert result["clusters_backfilled"] >= 2

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT canonical_key, lifecycle_state
                FROM theme_clusters
                ORDER BY id
                """
            )
        ).fetchall()

    assert rows[0][1] == "active"
    assert rows[1][1] == "retired"
