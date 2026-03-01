"""Task-level tests for ingestion error surfacing."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.theme import ContentSource
from app.tasks.theme_discovery_tasks import ingest_content, poll_due_sources


@pytest.fixture
def db_session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session


def test_ingest_content_reports_error_count_from_service(
    monkeypatch: pytest.MonkeyPatch,
    db_session_factory,
) -> None:
    monkeypatch.setattr("app.tasks.theme_discovery_tasks.SessionLocal", db_session_factory)

    def fake_fetch_all_active_sources(self, lookback_days=None):
        return {
            "total_sources": 1,
            "total_new_items": 0,
            "sources_fetched": [],
            "errors": [{"name": "@alice", "error": "XUI auth not ready"}],
        }

    monkeypatch.setattr(
        "app.services.content_ingestion_service.ContentIngestionService.fetch_all_active_sources",
        fake_fetch_all_active_sources,
    )

    result = ingest_content()
    assert result["errors"] == 1
    assert result["new_items"] == 0


def test_poll_due_sources_surfaces_auth_failures(
    monkeypatch: pytest.MonkeyPatch,
    db_session_factory,
) -> None:
    session = db_session_factory()
    source = ContentSource(
        name="@alice",
        source_type="twitter",
        url="https://x.com/alice",
        is_active=True,
        fetch_interval_minutes=60,
        last_fetched_at=datetime.utcnow() - timedelta(hours=2),
    )
    session.add(source)
    session.commit()
    session.close()

    monkeypatch.setattr("app.tasks.theme_discovery_tasks.SessionLocal", db_session_factory)

    def fake_fetch_source(self, source, lookback_days=None):
        raise RuntimeError("XUI auth not ready")

    monkeypatch.setattr(
        "app.services.content_ingestion_service.ContentIngestionService.fetch_source",
        fake_fetch_source,
    )

    result = poll_due_sources()
    assert result["errors"] == 1
    assert result["sources_polled"] == 1
    assert result["results"][0]["status"] == "error"
    assert "XUI auth not ready" in result["results"][0]["error"]
