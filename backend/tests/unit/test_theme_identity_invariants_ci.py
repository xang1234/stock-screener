"""Fast identity invariants for CI quality gates."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.db_migrations.theme_aliases_migration import migrate_theme_aliases
from app.db_migrations.theme_cluster_identity_migration import migrate_theme_cluster_identity
from app.infra.db.repositories.theme_alias_repo import SqlThemeAliasRepository
from app.models.theme import ThemeCluster
from app.schemas.theme import ThemeClusterResponse


def test_db_enforces_pipeline_scoped_uniqueness_for_cluster_keys():
    engine = create_engine("sqlite:///:memory:")
    migrate_theme_cluster_identity(engine)

    with engine.connect() as conn:
        conn.execute(
            text(
                """
                INSERT INTO theme_clusters(name, canonical_key, display_name, pipeline)
                VALUES ('AI Infrastructure', 'ai_infrastructure', 'AI Infrastructure', 'technical')
                """
            )
        )
        conn.commit()

        with pytest.raises(IntegrityError):
            conn.execute(
                text(
                    """
                    INSERT INTO theme_clusters(name, canonical_key, display_name, pipeline)
                    VALUES ('AI Infra Duplicate', 'ai_infrastructure', 'AI Infra Duplicate', 'technical')
                    """
                )
            )
            conn.commit()


def test_db_enforces_pipeline_scoped_uniqueness_for_alias_keys():
    engine = create_engine("sqlite:///:memory:")
    migrate_theme_cluster_identity(engine)
    migrate_theme_aliases(engine)

    with engine.connect() as conn:
        conn.execute(
            text(
                """
                INSERT INTO theme_clusters(name, canonical_key, display_name, pipeline)
                VALUES ('AI Infrastructure', 'ai_infrastructure', 'AI Infrastructure', 'technical')
                """
            )
        )
        conn.commit()

        cluster_id = conn.execute(text("SELECT id FROM theme_clusters LIMIT 1")).scalar_one()
        conn.execute(
            text(
                """
                INSERT INTO theme_aliases(theme_cluster_id, pipeline, alias_text, alias_key)
                VALUES (:cluster_id, 'technical', 'AI Infrastructure', 'ai_infrastructure')
                """
            ),
            {"cluster_id": int(cluster_id)},
        )
        conn.commit()

        with pytest.raises(IntegrityError):
            conn.execute(
                text(
                    """
                    INSERT INTO theme_aliases(theme_cluster_id, pipeline, alias_text, alias_key)
                    VALUES (:cluster_id, 'technical', 'A.I. Infrastructure', 'ai_infrastructure')
                    """
                ),
                {"cluster_id": int(cluster_id)},
            )
            conn.commit()


def test_repository_rejects_invalid_pipeline_or_empty_alias_transitions():
    engine = create_engine("sqlite:///:memory:")
    SessionFactory = sessionmaker(bind=engine)
    from app.database import Base

    Base.metadata.create_all(engine)

    session: Session = SessionFactory()
    try:
        cluster = ThemeCluster(
            canonical_key="ai_infrastructure",
            display_name="AI Infrastructure",
            name="AI Infrastructure",
            pipeline="technical",
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
            is_active=True,
        )
        session.add(cluster)
        session.flush()
        repo = SqlThemeAliasRepository(session)

        with pytest.raises(ValueError, match="Invalid pipeline"):
            repo.record_observation(
                theme_cluster_id=cluster.id,
                pipeline="macro",
                alias_text="AI Infrastructure",
            )

        with pytest.raises(ValueError, match="empty alias_text"):
            repo.record_observation(
                theme_cluster_id=cluster.id,
                pipeline="technical",
                alias_text="   ",
            )
    finally:
        session.close()


def test_theme_cluster_response_contract_rejects_null_or_invalid_identity_fields():
    with pytest.raises(ValidationError):
        ThemeClusterResponse.model_validate(
            {
                "id": 1,
                "name": "AI Infrastructure",
                "canonical_key": "AI Infrastructure",
                "display_name": "AI Infrastructure",
                "aliases": ["AI Infra"],
                "description": None,
                "pipeline": "macro",
                "category": None,
                "is_emerging": True,
                "is_validated": False,
                "discovery_source": "llm_extraction",
                "first_seen_at": None,
                "last_seen_at": None,
            }
        )
