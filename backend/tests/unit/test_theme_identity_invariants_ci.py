"""Fast identity invariants for CI quality gates."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base
from app.api.v1.themes import _safe_theme_cluster_response
from app.infra.db.repositories.theme_alias_repo import SqlThemeAliasRepository
from app.models.theme import ThemeCluster
from app.schemas.theme import ThemeClusterResponse


def test_db_enforces_pipeline_scoped_uniqueness_for_cluster_keys():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionFactory = sessionmaker(bind=engine)

    session = SessionFactory()
    try:
        session.add(ThemeCluster(
            name="AI Infrastructure",
            canonical_key="ai_infrastructure",
            display_name="AI Infrastructure",
            pipeline="technical",
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
            is_active=True,
        ))
        session.commit()

        with pytest.raises(IntegrityError):
            session.add(ThemeCluster(
                name="AI Infra Duplicate",
                canonical_key="ai_infrastructure",
                display_name="AI Infra Duplicate",
                pipeline="technical",
                first_seen_at=datetime.utcnow(),
                last_seen_at=datetime.utcnow(),
                is_active=True,
            ))
            session.commit()
    finally:
        session.close()


def test_db_enforces_pipeline_scoped_uniqueness_for_alias_keys():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionFactory = sessionmaker(bind=engine)

    session = SessionFactory()
    try:
        cluster = ThemeCluster(
            name="AI Infrastructure",
            canonical_key="ai_infrastructure",
            display_name="AI Infrastructure",
            pipeline="technical",
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
            is_active=True,
        )
        session.add(cluster)
        session.flush()

        session.execute(
            text(
                """
                INSERT INTO theme_aliases(theme_cluster_id, pipeline, alias_text, alias_key, source, confidence, evidence_count, is_active)
                VALUES (:cluster_id, 'technical', 'AI Infrastructure', 'ai_infrastructure', 'test', 1.0, 1, 1)
                """
            ),
            {"cluster_id": int(cluster.id)},
        )
        session.commit()

        with pytest.raises(IntegrityError):
            session.execute(
                text(
                    """
                    INSERT INTO theme_aliases(theme_cluster_id, pipeline, alias_text, alias_key, source, confidence, evidence_count, is_active)
                    VALUES (:cluster_id, 'technical', 'A.I. Infrastructure', 'ai_infrastructure', 'test', 1.0, 1, 1)
                    """
                ),
                {"cluster_id": int(cluster.id)},
            )
            session.commit()
    finally:
        session.close()


def test_repository_rejects_invalid_pipeline_or_empty_alias_transitions():
    engine = create_engine("sqlite:///:memory:")
    SessionFactory = sessionmaker(bind=engine)
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


def test_safe_theme_cluster_response_normalizes_legacy_invalid_identity_values():
    cluster = SimpleNamespace(
        id=42,
        name="AI Infrastructure",
        canonical_key="AI Infrastructure!!",
        display_name="",
        aliases="AI Infra",
        description=None,
        pipeline="macro",
        category=None,
        is_emerging=True,
        is_validated=False,
        discovery_source="legacy_import",
        first_seen_at=None,
        last_seen_at=None,
    )

    response = _safe_theme_cluster_response(cluster)

    assert response.id == 42
    assert response.pipeline == "technical"
    assert response.canonical_key == "ai_infrastructure"
    assert response.display_name == "AI Infrastructure"
    assert response.aliases == ["AI Infra"]
