"""Tests for embedding freshness behavior in ThemeMergingService."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.theme import ThemeAlias, ThemeCluster, ThemeEmbedding
from app.services.theme_embedding_service import ThemeEmbeddingRepository
from app.services.theme_merging_service import ThemeMergingService


class _StubEmbeddingEngine:
    def __init__(self):
        self.calls = 0

    def get_encoder(self):
        return object()

    def encode(self, text: str):
        self.calls += 1
        return np.array([float(len(text) % 11), 1.0])


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _make_service(db_session, engine: _StubEmbeddingEngine) -> ThemeMergingService:
    service = ThemeMergingService.__new__(ThemeMergingService)
    service.db = db_session
    service.embedding_repo = ThemeEmbeddingRepository(db_session)
    service.embedding_engine = engine
    service.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    service.EMBEDDING_MODEL_VERSION = "embedding-v1"
    return service


def _make_theme(db_session) -> ThemeCluster:
    theme = ThemeCluster(
        canonical_key="ai_infrastructure",
        display_name="AI Infrastructure",
        name="AI Infrastructure",
        aliases=["AI Infra", "Datacenter Buildout"],
        description="Semiconductor and power infra cycle",
        category="technology",
        pipeline="technical",
        is_active=True,
        first_seen_at=datetime.utcnow(),
        last_seen_at=datetime.utcnow(),
    )
    db_session.add(theme)
    db_session.commit()
    return theme


def _make_embedding(db_session, theme_id: int, *, stale: bool = False) -> ThemeEmbedding:
    embedding = ThemeEmbedding(
        theme_cluster_id=theme_id,
        embedding="[0.5, 0.5]",
        embedding_model="all-MiniLM-L6-v2",
        model_version="embedding-v1",
        content_hash="hash-v1",
        is_stale=stale,
    )
    db_session.add(embedding)
    db_session.commit()
    return embedding


def test_update_theme_embedding_skips_recompute_when_content_and_model_unchanged(db_session):
    engine = _StubEmbeddingEngine()
    service = _make_service(db_session, engine)
    theme = _make_theme(db_session)

    first_record, first_refreshed = service.update_theme_embedding(theme)
    second_record, second_refreshed = service.update_theme_embedding(theme)

    assert first_record is not None
    assert first_refreshed is True
    assert second_record is not None
    assert second_refreshed is False
    assert engine.calls == 1


def test_update_theme_embedding_recomputes_when_content_hash_changes(db_session):
    engine = _StubEmbeddingEngine()
    service = _make_service(db_session, engine)
    theme = _make_theme(db_session)

    initial_record, initial_refreshed = service.update_theme_embedding(theme)
    assert initial_record is not None
    assert initial_refreshed is True
    initial_hash = initial_record.content_hash

    theme.description = "Grid modernization and datacenter interconnect buildout"
    db_session.commit()

    refreshed_record, refreshed = service.update_theme_embedding(theme)
    assert refreshed_record is not None
    assert refreshed is True
    assert refreshed_record.content_hash != initial_hash
    assert engine.calls == 2


def test_update_theme_embedding_recomputes_when_marked_stale(db_session):
    engine = _StubEmbeddingEngine()
    service = _make_service(db_session, engine)
    theme = _make_theme(db_session)

    record, refreshed = service.update_theme_embedding(theme)
    assert record is not None
    assert refreshed is True

    record.is_stale = True
    db_session.commit()

    stale_refresh_record, stale_refreshed = service.update_theme_embedding(theme)
    assert stale_refresh_record is not None
    assert stale_refreshed is True
    assert stale_refresh_record.is_stale is False
    assert engine.calls == 2


def test_find_similar_themes_refreshes_stale_records_before_similarity(db_session):
    engine = _StubEmbeddingEngine()
    service = _make_service(db_session, engine)

    source = ThemeCluster(
        canonical_key="ai_infrastructure",
        display_name="AI Infrastructure",
        name="AI Infrastructure",
        aliases=["AI Infra"],
        description="Compute + power infra",
        category="technology",
        pipeline="technical",
        is_active=True,
        first_seen_at=datetime.utcnow(),
        last_seen_at=datetime.utcnow(),
    )
    peer = ThemeCluster(
        canonical_key="ai_datacenter_power",
        display_name="AI Datacenter Power",
        name="AI Datacenter Power",
        aliases=["AI Power"],
        description="Datacenter power demand",
        category="technology",
        pipeline="technical",
        is_active=True,
        first_seen_at=datetime.utcnow(),
        last_seen_at=datetime.utcnow(),
    )
    db_session.add_all([source, peer])
    db_session.flush()
    db_session.add_all(
        [
            ThemeEmbedding(
                theme_cluster_id=source.id,
                embedding="[0.0, 1.0]",
                embedding_model="all-MiniLM-L6-v2",
                model_version="embedding-v1",
                is_stale=True,
            ),
            ThemeEmbedding(
                theme_cluster_id=peer.id,
                embedding="[1.0, 0.0]",
                embedding_model="all-MiniLM-L6-v2",
                model_version="embedding-v1",
                is_stale=True,
            ),
        ]
    )
    db_session.commit()

    similar = service.find_similar_themes(source.id, threshold=0.0)

    assert engine.calls >= 2
    assert any(item["theme_id"] == peer.id for item in similar)

    refreshed_source = service.embedding_repo.get_for_cluster(source.id)
    refreshed_peer = service.embedding_repo.get_for_cluster(peer.id)
    assert refreshed_source is not None and refreshed_source.is_stale is False
    assert refreshed_peer is not None and refreshed_peer.is_stale is False


def test_cluster_identity_update_marks_embedding_stale_in_same_commit(db_session):
    theme = _make_theme(db_session)
    _make_embedding(db_session, theme.id, stale=False)

    theme.display_name = "AI Infrastructure and Grid"
    theme.name = "AI Infrastructure and Grid"
    db_session.commit()

    refreshed = db_session.query(ThemeEmbedding).filter(
        ThemeEmbedding.theme_cluster_id == theme.id
    ).one()
    assert refreshed.is_stale is True


def test_alias_insert_marks_embedding_stale(db_session):
    theme = _make_theme(db_session)
    _make_embedding(db_session, theme.id, stale=False)

    db_session.add(
        ThemeAlias(
            theme_cluster_id=theme.id,
            pipeline="technical",
            alias_text="AI Infra Trade",
            alias_key="ai_infra_trade",
            source="llm_extraction",
            confidence=0.82,
            evidence_count=1,
            is_active=True,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
    )
    db_session.commit()

    refreshed = db_session.query(ThemeEmbedding).filter(
        ThemeEmbedding.theme_cluster_id == theme.id
    ).one()
    assert refreshed.is_stale is True


def test_alias_telemetry_update_does_not_mark_embedding_stale(db_session):
    theme = _make_theme(db_session)
    _make_embedding(db_session, theme.id, stale=False)

    alias = ThemeAlias(
        theme_cluster_id=theme.id,
        pipeline="technical",
        alias_text="AI Infra Trade",
        alias_key="ai_infra_trade",
        source="llm_extraction",
        confidence=0.70,
        evidence_count=1,
        is_active=True,
        first_seen_at=datetime.utcnow(),
        last_seen_at=datetime.utcnow(),
    )
    db_session.add(alias)
    db_session.commit()

    # Reset to isolate telemetry-only update behavior.
    embedding = db_session.query(ThemeEmbedding).filter(
        ThemeEmbedding.theme_cluster_id == theme.id
    ).one()
    embedding.is_stale = False
    db_session.commit()

    alias.confidence = 0.75
    alias.evidence_count = 2
    alias.last_seen_at = datetime.utcnow()
    db_session.commit()

    refreshed = db_session.query(ThemeEmbedding).filter(
        ThemeEmbedding.theme_cluster_id == theme.id
    ).one()
    assert refreshed.is_stale is False
