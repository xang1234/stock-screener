"""Integration-style unit tests for ContentIngestionService twitter flow."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.theme import ContentItem, ContentItemPipelineState, ContentSource
from app.services.content_ingestion_service import ContentIngestionService


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_fetch_source_inserts_items_and_deduplicates_external_ids(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = ContentSource(
        name="@alice",
        source_type="twitter",
        url="https://x.com/alice",
        is_active=True,
        fetch_interval_minutes=60,
    )
    db_session.add(source)
    db_session.commit()

    class FakeTwitterFetcher:
        def fetch(self, _source, _since=None):
            now = datetime.utcnow()
            return [
                {
                    "external_id": "dup-id",
                    "title": "",
                    "content": "hello",
                    "url": "https://x.com/alice/status/1",
                    "author": "@alice",
                    "published_at": now,
                },
                {
                    "external_id": "dup-id",
                    "title": "",
                    "content": "hello duplicate",
                    "url": "https://x.com/alice/status/1",
                    "author": "@alice",
                    "published_at": now,
                },
            ]

    monkeypatch.setattr(
        "app.services.content_ingestion_service.TwitterFetcher",
        lambda: FakeTwitterFetcher(),
    )
    service = ContentIngestionService(db_session)

    inserted = service.fetch_source(source)
    assert inserted == 1

    rows = db_session.query(ContentItem).filter(ContentItem.source_type == "twitter").all()
    assert len(rows) == 1
    assert rows[0].external_id == "dup-id"
    db_session.refresh(source)
    assert source.last_fetched_at is not None
    assert source.total_items_fetched == 1

    states = db_session.query(ContentItemPipelineState).filter(
        ContentItemPipelineState.content_item_id == rows[0].id,
    ).order_by(ContentItemPipelineState.pipeline.asc()).all()
    assert [state.pipeline for state in states] == ["fundamental", "technical"]
    assert all(state.status == "pending" for state in states)


def test_fetch_source_seeds_only_assigned_pipeline_state_rows(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = ContentSource(
        name="Stratechery",
        source_type="substack",
        url="https://stratechery.com/feed",
        is_active=True,
        pipelines=["fundamental"],
    )
    db_session.add(source)
    db_session.commit()

    class FakeRssFetcher:
        def fetch(self, _source, _since=None):
            return [
                {
                    "external_id": "fundamental-only",
                    "title": "Deep dive",
                    "content": "Long form analysis",
                    "url": "https://stratechery.com/p/deep-dive",
                    "author": "Ben Thompson",
                    "published_at": datetime.utcnow(),
                }
            ]

    monkeypatch.setattr(
        "app.services.content_ingestion_service.RSSFetcher",
        lambda: FakeRssFetcher(),
    )
    service = ContentIngestionService(db_session)

    inserted = service.fetch_source(source)
    assert inserted == 1

    item = db_session.query(ContentItem).filter(ContentItem.external_id == "fundamental-only").one()
    states = db_session.query(ContentItemPipelineState).filter(
        ContentItemPipelineState.content_item_id == item.id,
    ).all()
    assert len(states) == 1
    assert states[0].pipeline == "fundamental"
    assert states[0].status == "pending"


def test_fetch_source_backfills_missing_pipeline_state_for_existing_item(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = ContentSource(
        name="@alice",
        source_type="twitter",
        url="https://x.com/alice",
        is_active=True,
        pipelines=["technical", "fundamental"],
    )
    db_session.add(source)
    db_session.flush()

    item = ContentItem(
        source_id=source.id,
        source_type=source.source_type,
        source_name=source.name,
        external_id="existing-item",
        title="Existing post",
        content="Already ingested body",
        url="https://x.com/alice/status/42",
        author="@alice",
        published_at=datetime.utcnow(),
        is_processed=False,
    )
    db_session.add(item)
    db_session.flush()
    db_session.add(
        ContentItemPipelineState(
            content_item_id=item.id,
            pipeline="technical",
            status="pending",
            attempt_count=0,
        )
    )
    db_session.commit()

    class FakeTwitterFetcher:
        def fetch(self, _source, _since=None):
            return [
                {
                    "external_id": "existing-item",
                    "title": "Existing post",
                    "content": "Already ingested body",
                    "url": "https://x.com/alice/status/42",
                    "author": "@alice",
                    "published_at": datetime.utcnow(),
                }
            ]

    monkeypatch.setattr(
        "app.services.content_ingestion_service.TwitterFetcher",
        lambda: FakeTwitterFetcher(),
    )
    service = ContentIngestionService(db_session)

    inserted = service.fetch_source(source)
    assert inserted == 0

    states = db_session.query(ContentItemPipelineState).filter(
        ContentItemPipelineState.content_item_id == item.id,
    ).order_by(ContentItemPipelineState.pipeline.asc()).all()
    assert [state.pipeline for state in states] == ["fundamental", "technical"]
    assert all(state.status == "pending" for state in states)


def test_fetch_source_propagates_adapter_error_and_does_not_advance_last_fetched_at(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_fetch = datetime.utcnow() - timedelta(hours=2)
    source = ContentSource(
        name="@alice",
        source_type="twitter",
        url="https://x.com/alice",
        is_active=True,
        fetch_interval_minutes=60,
        last_fetched_at=previous_fetch,
    )
    db_session.add(source)
    db_session.commit()

    class FailingTwitterFetcher:
        def fetch(self, _source, _since=None):
            raise RuntimeError("XUI auth not ready")

    monkeypatch.setattr(
        "app.services.content_ingestion_service.TwitterFetcher",
        lambda: FailingTwitterFetcher(),
    )
    service = ContentIngestionService(db_session)

    with pytest.raises(RuntimeError, match="XUI auth not ready"):
        service.fetch_source(source)

    db_session.refresh(source)
    assert source.last_fetched_at == previous_fetch
    assert db_session.query(ContentItem).count() == 0
