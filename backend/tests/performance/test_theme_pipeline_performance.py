"""Performance regression gates for theme extraction and consolidation.

Covers:
1) Extraction-time cluster matching latency/throughput
2) Consolidation candidate generation latency/throughput
3) Theme extraction/consolidation API response-time budgets
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta

import httpx
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base, get_db
from app.main import app
from app.models.theme import ContentItem, ContentSource, ThemeCluster, ThemeEmbedding
from app.services.theme_discovery_service import ThemeDiscoveryService
from app.services.theme_embedding_service import ThemeEmbeddingEngine
from app.services.theme_extraction_service import ThemeExtractionService
from app.services.theme_merging_service import ThemeMergingService


# ── Budget constants (generous but regression-sensitive) ─────────────────────
MATCH_P95_BUDGET_MS = 30.0
MATCH_MIN_THROUGHPUT_OPS = 120.0

PAIR_GEN_P95_BUDGET_MS = 900.0
PAIR_GEN_MIN_THROUGHPUT_PAIRS = 800.0

API_EXTRACT_P95_BUDGET_MS = 260.0
API_EXTRACT_MIN_THROUGHPUT_ITEMS = 15.0

API_CONSOLIDATE_P95_BUDGET_MS = 900.0
API_CONSOLIDATE_MIN_THROUGHPUT_PAIRS = 120.0


def _p95(values: list[float]) -> float:
    """Return deterministic nearest-rank p95."""
    assert values, "p95 requires at least one value"
    ordered = sorted(values)
    idx = max(0, int(np.ceil(0.95 * len(ordered))) - 1)
    return float(ordered[idx])


def _make_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def _seed_theme_clusters(session: Session, *, count: int, pipeline: str = "technical") -> None:
    for i in range(count):
        cluster = ThemeCluster(
            name=f"Theme {i}",
            canonical_key=f"theme_{i}",
            display_name=f"Theme {i}",
            aliases=[f"Theme {i}", f"T{i}"],
            pipeline=pipeline,
            category="technology" if i % 2 == 0 else "macro",
            is_active=True,
            first_seen_at=datetime.utcnow() - timedelta(days=10),
            last_seen_at=datetime.utcnow(),
            discovery_source="llm_extraction",
        )
        session.add(cluster)
    session.commit()


def _seed_cluster_embeddings(session: Session, *, groups: int = 12, per_group: int = 10) -> int:
    rng = np.random.default_rng(42)
    created = 0
    for g in range(groups):
        base = rng.normal(0.0, 1.0, 384)
        base_norm = np.linalg.norm(base) or 1.0
        base = base / base_norm

        for i in range(per_group):
            idx = (g * per_group) + i
            cluster = ThemeCluster(
                name=f"Cluster {idx}",
                canonical_key=f"cluster_{idx}",
                display_name=f"Cluster {idx}",
                aliases=[f"Cluster {idx}", f"G{g}"],
                pipeline="technical",
                category="technology" if g % 2 == 0 else "macro",
                is_active=True,
                first_seen_at=datetime.utcnow() - timedelta(days=30),
                last_seen_at=datetime.utcnow(),
                discovery_source="llm_extraction",
            )
            session.add(cluster)
            session.flush()

            noise = rng.normal(0.0, 0.02, 384)
            vec = base + noise
            vec_norm = np.linalg.norm(vec) or 1.0
            vec = vec / vec_norm
            session.add(
                ThemeEmbedding(
                    theme_cluster_id=cluster.id,
                    embedding=json.dumps(vec.tolist()),
                    embedding_model=ThemeMergingService.EMBEDDING_MODEL,
                    model_version=ThemeMergingService.EMBEDDING_MODEL_VERSION,
                    embedding_text=f"Cluster {idx}",
                    content_hash=f"hash-{idx}",
                    is_stale=False,
                )
            )
            created += 1
    session.commit()
    return created


@pytest.mark.performance
def test_extraction_cluster_match_p95_and_throughput(monkeypatch):
    session = _make_session()
    try:
        _seed_theme_clusters(session, count=280)

        # Ensure one canonical exact hit is present and stable.
        session.add(
            ThemeCluster(
                name="AI Infrastructure",
                canonical_key="ai_infrastructure",
                display_name="AI Infrastructure",
                aliases=["AI Infrastructure", "AI Infra"],
                pipeline="technical",
                category="technology",
                is_active=True,
                first_seen_at=datetime.utcnow() - timedelta(days=5),
                last_seen_at=datetime.utcnow(),
                discovery_source="llm_extraction",
            )
        )
        session.commit()

        monkeypatch.setattr(ThemeExtractionService, "_load_configured_model", lambda self: None)
        monkeypatch.setattr(ThemeExtractionService, "_init_client", lambda self: None)
        service = ThemeExtractionService(session, pipeline="technical")

        mention = {
            "theme": "AI Infrastructure",
            "tickers": ["NVDA", "AVGO"],
            "sentiment": "bullish",
            "confidence": 0.92,
            "excerpt": "AI infrastructure capex remains strong.",
        }

        # Warmup to stabilize SQLAlchemy/sqlite caches.
        for _ in range(15):
            service._resolve_cluster_match(mention, source_type="news")

        latencies_ms: list[float] = []
        started = time.perf_counter()
        for _ in range(120):
            t0 = time.perf_counter()
            cluster, decision = service._resolve_cluster_match(mention, source_type="news")
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)
            assert cluster.id is not None
            assert decision.method in {"exact_canonical_key", "exact_display_name", "exact_alias_key"}
        elapsed_s = time.perf_counter() - started

        p95 = _p95(latencies_ms)
        throughput = float(len(latencies_ms)) / max(elapsed_s, 1e-9)
        assert p95 <= MATCH_P95_BUDGET_MS, (
            f"_resolve_cluster_match p95 {p95:.2f}ms exceeded {MATCH_P95_BUDGET_MS:.2f}ms"
        )
        assert throughput >= MATCH_MIN_THROUGHPUT_OPS, (
            f"_resolve_cluster_match throughput {throughput:.2f} ops/s below {MATCH_MIN_THROUGHPUT_OPS:.2f}"
        )
    finally:
        session.close()


@pytest.mark.performance
def test_consolidation_pair_generation_p95_and_throughput(monkeypatch):
    session = _make_session()
    try:
        embeddings = _seed_cluster_embeddings(session, groups=12, per_group=10)
        assert embeddings >= 100

        monkeypatch.setattr(ThemeMergingService, "_init_llm_client", lambda self: None)
        service = ThemeMergingService(session)

        # Warmup
        service.find_all_similar_pairs(
            threshold=service.EMBEDDING_THRESHOLD,
            pipeline="technical",
            top_k=12,
        )

        latencies_ms: list[float] = []
        compared_pairs_total = 0
        started = time.perf_counter()
        for _ in range(6):
            t0 = time.perf_counter()
            pairs = service.find_all_similar_pairs(
                threshold=service.EMBEDDING_THRESHOLD,
                pipeline="technical",
                top_k=12,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)
            compared_pairs_total += int(service._last_pair_generation_metrics.get("compared_pairs", 0))
            assert isinstance(pairs, list)
        elapsed_s = time.perf_counter() - started

        p95 = _p95(latencies_ms)
        pair_throughput = float(compared_pairs_total) / max(elapsed_s, 1e-9)
        assert p95 <= PAIR_GEN_P95_BUDGET_MS, (
            f"find_all_similar_pairs p95 {p95:.2f}ms exceeded {PAIR_GEN_P95_BUDGET_MS:.2f}ms"
        )
        assert pair_throughput >= PAIR_GEN_MIN_THROUGHPUT_PAIRS, (
            f"find_all_similar_pairs throughput {pair_throughput:.2f} compared-pairs/s "
            f"below {PAIR_GEN_MIN_THROUGHPUT_PAIRS:.2f}"
        )
    finally:
        session.close()


@pytest.mark.performance
@pytest.mark.asyncio
async def test_extraction_api_response_time_and_throughput(monkeypatch):
    session = _make_session()

    source = ContentSource(
        name="Perf Source",
        source_type="news",
        url="https://example.com/perf",
        is_active=True,
        pipelines=["technical"],
    )
    session.add(source)
    session.flush()

    for i in range(40):
        session.add(
            ContentItem(
                source_id=source.id,
                source_type="news",
                source_name=source.name,
                external_id=f"perf-{i}",
                title=f"Title {i}",
                content=f"Body {i}",
                published_at=datetime.utcnow() - timedelta(minutes=i),
                is_processed=False,
            )
        )
    session.commit()

    monkeypatch.setattr(ThemeExtractionService, "_load_configured_model", lambda self: None)
    monkeypatch.setattr(ThemeExtractionService, "_init_client", lambda self: None)
    monkeypatch.setattr(
        ThemeExtractionService,
        "extract_from_content",
        lambda self, content_item: [
            {
                "theme": "AI Infrastructure" if content_item.id % 2 == 0 else "Defense & Drones",
                "tickers": ["NVDA", "AVGO"] if content_item.id % 2 == 0 else ["LMT", "NOC"],
                "sentiment": "bullish",
                "confidence": 0.9,
                "excerpt": content_item.title or "",
            }
        ],
    )
    monkeypatch.setattr(
        ThemeDiscoveryService,
        "update_all_theme_metrics",
        lambda self: {"themes_updated": 0},
    )

    def _override_get_db():
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    try:
        transport = httpx.ASGITransport(app=app)
        latencies_ms: list[float] = []
        total_processed = 0
        started = time.perf_counter()
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            for _ in range(8):
                t0 = time.perf_counter()
                response = await client.post("/api/v1/themes/extract?pipeline=technical&limit=5")
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(elapsed_ms)
                assert response.status_code == 200
                payload = response.json()
                total_processed += int(payload.get("processed", 0))
        elapsed_s = time.perf_counter() - started

        p95 = _p95(latencies_ms)
        throughput = float(total_processed) / max(elapsed_s, 1e-9)
        assert total_processed >= 35, f"Unexpected low extraction volume in benchmark: {total_processed}"
        assert p95 <= API_EXTRACT_P95_BUDGET_MS, (
            f"/themes/extract p95 {p95:.2f}ms exceeded {API_EXTRACT_P95_BUDGET_MS:.2f}ms"
        )
        assert throughput >= API_EXTRACT_MIN_THROUGHPUT_ITEMS, (
            f"/themes/extract throughput {throughput:.2f} items/s below {API_EXTRACT_MIN_THROUGHPUT_ITEMS:.2f}"
        )
    finally:
        app.dependency_overrides.pop(get_db, None)
        session.close()


@pytest.mark.performance
@pytest.mark.asyncio
async def test_consolidation_api_response_time_and_throughput(monkeypatch):
    session = _make_session()
    _seed_theme_clusters(session, count=60, pipeline="technical")
    cluster_ids = [row[0] for row in session.query(ThemeCluster.id).order_by(ThemeCluster.id.asc()).all()]
    synthetic_pairs = []
    for idx in range(0, min(len(cluster_ids) - 1, 24)):
        synthetic_pairs.append(
            {
                "theme1_id": cluster_ids[idx],
                "theme2_id": cluster_ids[idx + 1],
                "similarity": 0.91,
                "pipeline": "technical",
            }
        )
    assert synthetic_pairs

    monkeypatch.setattr(ThemeMergingService, "_init_llm_client", lambda self: None)
    monkeypatch.setattr(ThemeEmbeddingEngine, "get_encoder", lambda self: object())
    monkeypatch.setattr(
        ThemeMergingService,
        "update_all_embeddings",
        lambda self: {"updated": 0, "skipped": 0, "errors": 0},
    )
    monkeypatch.setattr(
        ThemeMergingService,
        "verify_merge_with_llm",
        lambda self, theme1, theme2, similarity: {
            "should_merge": False,
            "confidence": 0.8,
            "relationship": "related",
            "reasoning": "deterministic perf gate",
            "canonical_name": None,
        },
    )
    monkeypatch.setattr(
        ThemeMergingService,
        "find_all_similar_pairs",
        lambda self, threshold=None, pipeline=None, top_k=None, recall_sample_size=0: synthetic_pairs,
    )

    def _override_get_db():
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    try:
        transport = httpx.ASGITransport(app=app)
        latencies_ms: list[float] = []
        pairs_found_total = 0
        started = time.perf_counter()
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            for _ in range(4):
                t0 = time.perf_counter()
                response = await client.post("/api/v1/themes/consolidate?dry_run=true")
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(elapsed_ms)
                assert response.status_code == 200
                payload = response.json()
                pairs_found_total += int(payload.get("pairs_found", 0))
        elapsed_s = time.perf_counter() - started

        p95 = _p95(latencies_ms)
        throughput = float(pairs_found_total) / max(elapsed_s, 1e-9)
        assert pairs_found_total > 0, "Expected non-zero consolidation pair throughput sample"
        assert p95 <= API_CONSOLIDATE_P95_BUDGET_MS, (
            f"/themes/consolidate p95 {p95:.2f}ms exceeded {API_CONSOLIDATE_P95_BUDGET_MS:.2f}ms"
        )
        assert throughput >= API_CONSOLIDATE_MIN_THROUGHPUT_PAIRS, (
            f"/themes/consolidate throughput {throughput:.2f} pairs/s "
            f"below {API_CONSOLIDATE_MIN_THROUGHPUT_PAIRS:.2f}"
        )
    finally:
        app.dependency_overrides.pop(get_db, None)
        session.close()
