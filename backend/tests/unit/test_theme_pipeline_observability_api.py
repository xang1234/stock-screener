"""Tests for pipeline observability dashboard endpoint payload."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.v1.themes import get_pipeline_observability
from app.database import Base
from app.models.theme import (
    ContentItem,
    ContentItemPipelineState,
    ContentSource,
    ThemeCluster,
    ThemeMention,
    ThemeMergeSuggestion,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def _seed_source_and_item(db_session, *, idx: int) -> ContentItem:
    source = ContentSource(
        name=f"Source-{idx}",
        source_type="news",
        url=f"https://example.com/{idx}",
        is_active=True,
        pipelines=["technical", "fundamental"],
    )
    db_session.add(source)
    db_session.flush()
    item = ContentItem(
        source_id=source.id,
        source_type="news",
        source_name=source.name,
        external_id=f"ext-{idx}",
        title=f"Title {idx}",
        content=f"Body {idx}",
        published_at=datetime.utcnow() - timedelta(days=1),
        is_processed=True,
    )
    db_session.add(item)
    db_session.flush()
    return item


def _seed_cluster_pair(db_session, *, base: int) -> tuple[ThemeCluster, ThemeCluster]:
    left = ThemeCluster(
        name=f"Theme-{base}",
        canonical_key=f"theme_{base}",
        display_name=f"Theme-{base}",
        pipeline="technical",
        is_active=True,
        lifecycle_state="active",
    )
    right = ThemeCluster(
        name=f"Theme-{base + 1}",
        canonical_key=f"theme_{base + 1}",
        display_name=f"Theme-{base + 1}",
        pipeline="technical",
        is_active=True,
        lifecycle_state="active",
    )
    db_session.add_all([left, right])
    db_session.flush()
    return left, right


@pytest.mark.asyncio
async def test_pipeline_observability_returns_metrics_without_alerts_for_healthy_state(db_session):
    item = _seed_source_and_item(db_session, idx=1)
    db_session.add(
        ContentItemPipelineState(
            content_item_id=item.id,
            pipeline="technical",
            status="processed",
            attempt_count=1,
        )
    )
    db_session.add(
        ThemeMention(
            content_item_id=item.id,
            source_type="news",
            source_name="Source-1",
            raw_theme="AI Infrastructure",
            canonical_theme="AI Infrastructure",
            theme_cluster_id=None,
            match_method="exact_canonical_key",
            match_score=1.0,
            match_threshold=1.0,
            threshold_version="match-v1",
            pipeline="technical",
            tickers=[],
            ticker_count=0,
            sentiment="neutral",
            confidence=0.8,
            excerpt="",
            mentioned_at=datetime.utcnow() - timedelta(days=1),
        )
    )
    db_session.commit()

    payload = await get_pipeline_observability(
        pipeline="technical",
        window_days=30,
        db=db_session,
    )

    assert payload["pipeline"] == "technical"
    assert payload["window_days"] == 30
    assert payload["metrics"]["total_mentions"] == 1
    assert payload["metrics"]["new_cluster_rate"] == 0.0
    assert payload["alerts"] == []


@pytest.mark.asyncio
async def test_pipeline_observability_emits_actionable_alerts_with_runbook_links(db_session):
    parse_failures = 8
    terminals = 2
    processed_without_mentions = 5
    for idx in range(parse_failures + terminals + processed_without_mentions):
        item = _seed_source_and_item(db_session, idx=idx + 10)
        if idx < parse_failures:
            db_session.add(
                ContentItemPipelineState(
                    content_item_id=item.id,
                    pipeline="technical",
                    status="failed_retryable",
                    attempt_count=1,
                    error_code="json_decode_error",
                )
            )
        elif idx < parse_failures + terminals:
            db_session.add(
                ContentItemPipelineState(
                    content_item_id=item.id,
                    pipeline="technical",
                    status="failed_terminal",
                    attempt_count=1,
                    error_code="provider_auth_error",
                )
            )
        else:
            db_session.add(
                ContentItemPipelineState(
                    content_item_id=item.id,
                    pipeline="technical",
                    status="processed",
                    attempt_count=1,
                )
            )

    # Mention mix with high new-cluster rate.
    for i in range(30):
        db_session.add(
            ThemeMention(
                content_item_id=None,
                source_type="news",
                source_name="news",
                raw_theme=f"Theme-{i}",
                canonical_theme=f"Theme-{i}",
                theme_cluster_id=None,
                match_method="create_new_cluster" if i < 20 else "embedding_similarity",
                match_score=0.0 if i < 20 else 0.9,
                match_threshold=1.0,
                threshold_version="match-v1",
                pipeline="technical",
                tickers=[],
                ticker_count=0,
                sentiment="neutral",
                confidence=0.7,
                excerpt="",
                mentioned_at=datetime.utcnow() - timedelta(days=1),
            )
        )

    # Build pending merge backlog.
    for base in range(0, 104, 2):
        left, right = _seed_cluster_pair(db_session, base=base)
        db_session.add(
            ThemeMergeSuggestion(
                source_cluster_id=left.id,
                target_cluster_id=right.id,
                pair_min_cluster_id=min(left.id, right.id),
                pair_max_cluster_id=max(left.id, right.id),
                status="pending",
                embedding_similarity=0.9,
            )
        )

    db_session.commit()

    payload = await get_pipeline_observability(
        pipeline="technical",
        window_days=30,
        db=db_session,
    )

    keys = {row["key"] for row in payload["alerts"]}
    assert "parse_failure_rate_high" in keys
    assert "processed_without_mentions_ratio_high" in keys
    assert "new_cluster_rate_high" in keys
    assert "merge_pending_backlog_high" in keys

    for row in payload["alerts"]:
        assert row["runbook_url"].startswith(
            "https://github.com/xang1234/stock-screener/blob/main/"
            "docs/theme_identity/e8_t4_pipeline_observability_runbook.md#"
        )
