"""Tests for theme_mentions match decision migration and verification."""

from __future__ import annotations

from sqlalchemy import create_engine, text

from app.db_migrations.theme_match_decision_migration import (
    EXPECTED_INDEXES,
    migrate_theme_match_decision,
    verify_theme_match_decision_schema,
)


def _bootstrap_theme_mentions(conn) -> None:
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS theme_mentions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_item_id INTEGER,
                source_type TEXT NOT NULL,
                source_name TEXT,
                raw_theme TEXT NOT NULL,
                canonical_theme TEXT,
                theme_cluster_id INTEGER,
                pipeline TEXT,
                tickers JSON,
                ticker_count INTEGER DEFAULT 0,
                sentiment TEXT,
                confidence FLOAT,
                excerpt TEXT,
                mentioned_at DATETIME,
                extracted_at DATETIME
            )
            """
        )
    )
    conn.commit()


def test_migration_adds_columns_and_indexes():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        _bootstrap_theme_mentions(conn)

    result = migrate_theme_match_decision(engine)
    assert result["table_exists"] is True
    assert set(result["indexes_ensured"]) == EXPECTED_INDEXES
    assert set(result["columns_added"]) == {
        "match_method",
        "match_score",
        "match_threshold",
        "threshold_version",
        "match_score_model",
        "match_score_model_version",
        "match_fallback_reason",
        "best_alternative_cluster_id",
        "best_alternative_score",
        "match_score_margin",
    }

    verification = verify_theme_match_decision_schema(engine)
    assert verification["ok"] is True
    assert verification["missing_columns"] == []
    assert verification["missing_indexes"] == []


def test_migration_is_idempotent():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        _bootstrap_theme_mentions(conn)

    first = migrate_theme_match_decision(engine)
    second = migrate_theme_match_decision(engine)

    assert first["table_exists"] is True
    assert second["table_exists"] is True
    assert second["columns_added"] == []
