"""Tests for embedding freshness behavior in ThemeMergingService."""

from __future__ import annotations

from datetime import datetime
from types import MethodType

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


def test_recompute_stale_embeddings_processes_bounded_batches_with_progress(db_session):
    engine = _StubEmbeddingEngine()
    service = _make_service(db_session, engine)
    theme_a = _make_theme(db_session)
    theme_b = ThemeCluster(
        canonical_key="grid_modernization",
        display_name="Grid Modernization",
        name="Grid Modernization",
        aliases=["Grid Upgrade"],
        description="Utility capex cycle",
        category="utilities",
        pipeline="technical",
        is_active=True,
        first_seen_at=datetime.utcnow(),
        last_seen_at=datetime.utcnow(),
    )
    db_session.add(theme_b)
    db_session.commit()
    _make_embedding(db_session, theme_a.id, stale=True)
    _make_embedding(db_session, theme_b.id, stale=True)

    progress = []
    result = service.recompute_stale_embeddings(
        batch_size=1,
        max_batches=2,
        on_batch=lambda payload: progress.append(dict(payload)),
    )

    assert result["stale_total_before"] == 2
    assert result["processed"] == 2
    assert result["refreshed"] == 2
    assert result["failed"] == 0
    assert result["stale_remaining_after"] == 0
    assert result["has_more"] is False
    assert len(progress) == 2


def test_recompute_stale_embeddings_prevents_failed_row_starvation_in_run(db_session):
    engine = _StubEmbeddingEngine()
    service = _make_service(db_session, engine)
    theme_a = _make_theme(db_session)
    theme_b = ThemeCluster(
        canonical_key="ai_datacenter_power",
        display_name="AI Datacenter Power",
        name="AI Datacenter Power",
        aliases=["AI Power"],
        description="Power demand for datacenters",
        category="technology",
        pipeline="technical",
        is_active=True,
        first_seen_at=datetime.utcnow(),
        last_seen_at=datetime.utcnow(),
    )
    db_session.add(theme_b)
    db_session.commit()
    _make_embedding(db_session, theme_a.id, stale=True)
    _make_embedding(db_session, theme_b.id, stale=True)

    original_update = service.update_theme_embedding

    def _patched_update(self, theme):
        if theme.id == theme_a.id:
            raise RuntimeError("simulated-encode-failure")
        return original_update(theme)

    service.update_theme_embedding = MethodType(_patched_update, service)

    result = service.recompute_stale_embeddings(batch_size=1, max_batches=2)

    assert result["processed"] == 2
    assert result["failed"] == 1
    assert result["refreshed"] == 1
    assert result["stale_remaining_after"] == 1
    assert result["has_more"] is True
    assert any(item["theme_cluster_id"] == theme_a.id for item in result["failed_clusters"])


def test_recompute_stale_embeddings_prevents_cross_run_starvation(db_session):
    engine = _StubEmbeddingEngine()
    service = _make_service(db_session, engine)
    theme_a = _make_theme(db_session)
    theme_b = ThemeCluster(
        canonical_key="ai_power_grid",
        display_name="AI Power Grid",
        name="AI Power Grid",
        aliases=["Power Grid AI"],
        description="Power grid modernization for AI workloads",
        category="utilities",
        pipeline="technical",
        is_active=True,
        first_seen_at=datetime.utcnow(),
        last_seen_at=datetime.utcnow(),
    )
    db_session.add(theme_b)
    db_session.commit()
    emb_a = _make_embedding(db_session, theme_a.id, stale=True)
    emb_b = _make_embedding(db_session, theme_b.id, stale=True)
    emb_a.updated_at = datetime(2020, 1, 1, 0, 0, 0)
    emb_b.updated_at = datetime(2021, 1, 1, 0, 0, 0)
    db_session.commit()

    original_update = service.update_theme_embedding

    def _fail_oldest(self, theme):
        if theme.id == theme_a.id:
            raise RuntimeError("simulated-stale-failure")
        return original_update(theme)

    service.update_theme_embedding = MethodType(_fail_oldest, service)
    first = service.recompute_stale_embeddings(batch_size=1, max_batches=1)
    assert first["processed"] == 1
    assert first["failed"] == 1

    # Next run should process theme_b first because failed theme_a was deferred.
    service.update_theme_embedding = original_update
    second = service.recompute_stale_embeddings(batch_size=1, max_batches=1)
    assert second["processed"] == 1
    assert second["refreshed"] == 1

    refreshed_a = db_session.query(ThemeEmbedding).filter(ThemeEmbedding.theme_cluster_id == theme_a.id).one()
    refreshed_b = db_session.query(ThemeEmbedding).filter(ThemeEmbedding.theme_cluster_id == theme_b.id).one()
    assert refreshed_a.is_stale is True
    assert refreshed_b.is_stale is False
