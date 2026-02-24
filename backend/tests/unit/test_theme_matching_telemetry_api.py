"""Tests for theme matching telemetry aggregation endpoint."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.v1.themes import get_matching_telemetry
from app.database import Base
from app.models.theme import ThemeMention


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def _mention(
    *,
    source_type: str,
    pipeline: str,
    method: str,
    threshold_version: str,
    confidence: float | None,
    score: float | None,
    fallback_reason: str | None = None,
    days_ago: int = 1,
) -> ThemeMention:
    return ThemeMention(
        source_type=source_type,
        source_name=source_type,
        raw_theme=f"{source_type}-{method}",
        canonical_theme=f"{source_type}-{method}",
        pipeline=pipeline,
        tickers=[],
        ticker_count=0,
        sentiment="neutral",
        confidence=confidence,
        excerpt="",
        mentioned_at=datetime.utcnow() - timedelta(days=days_ago),
        match_method=method,
        match_score=score,
        match_threshold=0.85 if threshold_version == "embedding-v1" else 1.0,
        threshold_version=threshold_version,
        match_fallback_reason=fallback_reason,
    )


@pytest.mark.asyncio
async def test_matching_telemetry_aggregates_method_source_and_threshold_breakdowns(db_session):
    db_session.add_all(
        [
            _mention(
                source_type="news",
                pipeline="technical",
                method="create_new_cluster",
                threshold_version="match-v1",
                confidence=0.32,
                score=0.0,
                fallback_reason="no_existing_cluster_match",
            ),
            _mention(
                source_type="news",
                pipeline="technical",
                method="fuzzy_lexical",
                threshold_version="match-v1",
                confidence=0.81,
                score=0.92,
            ),
            _mention(
                source_type="substack",
                pipeline="technical",
                method="embedding_similarity",
                threshold_version="embedding-v1",
                confidence=0.88,
                score=0.91,
            ),
            _mention(
                source_type="substack",
                pipeline="technical",
                method="create_new_cluster",
                threshold_version="embedding-v1",
                confidence=0.75,
                score=0.0,
                fallback_reason="embedding_ambiguous_review",
            ),
            # Outside window; should be excluded.
            _mention(
                source_type="news",
                pipeline="technical",
                method="exact_alias_key",
                threshold_version="match-v1",
                confidence=0.9,
                score=1.0,
                days_ago=90,
            ),
        ]
    )
    db_session.commit()

    payload = await get_matching_telemetry(
        days=30,
        pipeline="technical",
        source_type=None,
        threshold_version=None,
        db=db_session,
    )

    assert payload.total_mentions == 4
    assert payload.new_cluster_count == 2
    assert payload.attach_count == 2
    assert payload.new_cluster_rate == 0.5
    assert payload.attach_rate == 0.5
    assert {item.key for item in payload.by_threshold_version} == {"match-v1", "embedding-v1"}
    assert {item.key for item in payload.by_source_type} == {"news", "substack"}
    assert any(item.method == "create_new_cluster" and item.count == 2 for item in payload.method_distribution)
    assert any(item.reason == "none" and item.count == 2 for item in payload.decision_reason_distribution)


@pytest.mark.asyncio
async def test_matching_telemetry_supports_source_and_threshold_filters(db_session):
    db_session.add_all(
        [
            _mention(
                source_type="news",
                pipeline="technical",
                method="fuzzy_lexical",
                threshold_version="match-v1",
                confidence=0.78,
                score=0.90,
            ),
            _mention(
                source_type="substack",
                pipeline="technical",
                method="embedding_similarity",
                threshold_version="embedding-v1",
                confidence=0.86,
                score=0.93,
            ),
            _mention(
                source_type="substack",
                pipeline="technical",
                method="create_new_cluster",
                threshold_version="embedding-v1",
                confidence=0.65,
                score=0.0,
                fallback_reason="embedding_low_confidence_review",
            ),
        ]
    )
    db_session.commit()

    payload = await get_matching_telemetry(
        days=30,
        pipeline="technical",
        source_type="substack",
        threshold_version="embedding-v1",
        db=db_session,
    )

    assert payload.total_mentions == 2
    assert payload.new_cluster_count == 1
    assert payload.attach_count == 1
    assert payload.by_threshold_version[0].key == "embedding-v1"
    assert payload.by_source_type[0].key == "substack"
    assert all(item.method in {"embedding_similarity", "create_new_cluster"} for item in payload.method_distribution)
