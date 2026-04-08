"""Regression tests for pipeline-scoped content browser query logic."""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.v1.themes import _fetch_content_items_with_themes
from app.models.theme import (
    ContentItem,
    ContentItemPipelineState,
    ContentSource,
    ThemeCluster,
    ThemeMention,
)


def _build_session():
    engine = create_engine("sqlite:///:memory:")
    ContentSource.__table__.create(engine)
    ContentItem.__table__.create(engine)
    ContentItemPipelineState.__table__.create(engine)
    ThemeCluster.__table__.create(engine)
    ThemeMention.__table__.create(engine)
    return sessionmaker(bind=engine)()


def _seed_fundamental_twitter_data(db):
    now = datetime(2026, 3, 1, 10, 0, 0)
    source = ContentSource(
        name="@fund_source",
        source_type="twitter",
        url="https://x.com/fund_source",
        is_active=True,
        pipelines=["fundamental"],
    )
    db.add(source)
    db.flush()

    newest_unprocessed = ContentItem(
        source_id=source.id,
        source_type="twitter",
        source_name=source.name,
        external_id="new-unprocessed",
        title="Fresh tweet without extracted mentions yet",
        content="No extracted themes yet",
        url="https://x.com/fund_source/status/1",
        author="fund_source",
        published_at=now,
        is_processed=False,
    )
    older_processed = ContentItem(
        source_id=source.id,
        source_type="twitter",
        source_name=source.name,
        external_id="older-processed",
        title="Older tweet mentioning memory pricing",
        content="Memory cycle and pricing",
        url="https://x.com/fund_source/status/2",
        author="fund_source",
        published_at=now - timedelta(minutes=5),
        is_processed=True,
    )
    db.add_all([newest_unprocessed, older_processed])
    db.flush()

    cluster = ThemeCluster(
        name="Memory Pricing",
        canonical_key="memory-pricing",
        display_name="Memory Pricing",
        pipeline="fundamental",
        lifecycle_state="candidate",
        is_active=True,
        is_l1=False,
    )
    db.add(cluster)
    db.flush()

    mention = ThemeMention(
        content_item_id=older_processed.id,
        source_type="twitter",
        source_name=source.name,
        raw_theme="Memory Pricing",
        canonical_theme="Memory Pricing",
        theme_cluster_id=cluster.id,
        pipeline="fundamental",
        tickers=["WDC", "MU"],
        ticker_count=2,
        sentiment="neutral",
        confidence=0.92,
        excerpt="WDC and MU are tied to memory cycle pricing",
        mentioned_at=older_processed.published_at,
    )
    db.add(mention)
    db.add(
        ContentItemPipelineState(
            content_item_id=older_processed.id,
            pipeline="fundamental",
            status="processed",
        )
    )
    db.commit()
    return newest_unprocessed.id, older_processed.id


def test_sentiment_filter_applies_before_pagination():
    db = _build_session()
    try:
        _newest_id, older_id = _seed_fundamental_twitter_data(db)
        items, total = _fetch_content_items_with_themes(
            db,
            source_type="twitter",
            pipeline="fundamental",
            sentiment="neutral",
            limit=1,
            offset=0,
        )
        assert total == 1
        assert len(items) == 1
        assert items[0].id == older_id
        assert items[0].processing_status == "processed"
    finally:
        db.close()


def test_ticker_search_matches_outside_first_page_candidates():
    db = _build_session()
    try:
        _newest_id, older_id = _seed_fundamental_twitter_data(db)
        items, total = _fetch_content_items_with_themes(
            db,
            source_type="twitter",
            pipeline="fundamental",
            search="WDC",
            limit=1,
            offset=0,
        )
        assert total == 1
        assert len(items) == 1
        assert items[0].id == older_id
        assert "WDC" in items[0].tickers
    finally:
        db.close()


def test_pipeline_mode_defaults_missing_state_to_pending():
    db = _build_session()
    try:
        newest_id, older_id = _seed_fundamental_twitter_data(db)
        items, total = _fetch_content_items_with_themes(
            db,
            source_type="twitter",
            pipeline="fundamental",
            limit=10,
            offset=0,
        )
        assert total == 2
        assert [item.id for item in items] == [newest_id, older_id]
        assert items[0].processing_status == "pending"
        assert items[1].processing_status == "processed"
    finally:
        db.close()


def test_primary_sentiment_uses_actual_mention_frequency():
    db = _build_session()
    try:
        _newest_id, older_id = _seed_fundamental_twitter_data(db)

        # Add extra mention rows to make bearish the dominant sentiment.
        target_content = db.query(ContentItem).filter(ContentItem.id == older_id).first()
        assert target_content is not None
        cluster = db.query(ThemeCluster).first()
        assert cluster is not None

        db.add_all(
            [
                ThemeMention(
                    content_item_id=older_id,
                    source_type="twitter",
                    source_name="@fund_source",
                    raw_theme="Memory Pricing",
                    canonical_theme="Memory Pricing",
                    theme_cluster_id=cluster.id,
                    pipeline="fundamental",
                    tickers=["MU"],
                    ticker_count=1,
                    sentiment="bearish",
                    confidence=0.8,
                    excerpt="Bearish update 1",
                    mentioned_at=target_content.published_at,
                ),
                ThemeMention(
                    content_item_id=older_id,
                    source_type="twitter",
                    source_name="@fund_source",
                    raw_theme="Memory Pricing",
                    canonical_theme="Memory Pricing",
                    theme_cluster_id=cluster.id,
                    pipeline="fundamental",
                    tickers=["MU"],
                    ticker_count=1,
                    sentiment="bearish",
                    confidence=0.79,
                    excerpt="Bearish update 2",
                    mentioned_at=target_content.published_at,
                ),
                ThemeMention(
                    content_item_id=older_id,
                    source_type="twitter",
                    source_name="@fund_source",
                    raw_theme="Memory Pricing",
                    canonical_theme="Memory Pricing",
                    theme_cluster_id=cluster.id,
                    pipeline="fundamental",
                    tickers=["MU"],
                    ticker_count=1,
                    sentiment="bullish",
                    confidence=0.75,
                    excerpt="Bullish update",
                    mentioned_at=target_content.published_at,
                ),
            ]
        )
        db.commit()

        items, total = _fetch_content_items_with_themes(
            db,
            source_type="twitter",
            pipeline="fundamental",
            limit=10,
            offset=0,
        )
        assert total == 2
        older_item = next(item for item in items if item.id == older_id)
        assert older_item.primary_sentiment == "bearish"
    finally:
        db.close()
