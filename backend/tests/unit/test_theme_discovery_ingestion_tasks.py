"""Task-level tests for ingestion error surfacing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.theme import ContentSource
from app.tasks.theme_discovery_tasks import (
    apply_lifecycle_policies,
    calculate_theme_metrics,
    extract_themes,
    ingest_content,
    poll_due_sources,
    promote_candidate_themes,
    reprocess_failed_themes,
)


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
    monkeypatch.setattr(
        "app.tasks.theme_discovery_tasks._theme_automation_gate_result",
        lambda _db: None,
    )

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


def test_poll_due_sources_handles_aware_last_fetched_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeQuery:
        def filter(self, *args, **kwargs):
            return self

        def order_by(self, *args, **kwargs):
            return self

        def all(self):
            return [
                ContentSource(
                    name="@aware",
                    source_type="twitter",
                    url="https://x.com/aware",
                    is_active=True,
                    fetch_interval_minutes=60,
                    last_fetched_at=datetime.now(timezone.utc) - timedelta(hours=2),
                )
            ]

    class FakeSession:
        def query(self, *args, **kwargs):
            return FakeQuery()

        def rollback(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr("app.tasks.theme_discovery_tasks.SessionLocal", lambda: FakeSession())
    monkeypatch.setattr(
        "app.tasks.theme_discovery_tasks._theme_automation_gate_result",
        lambda _db: None,
    )

    def fake_fetch_source(self, source, lookback_days=None):
        return 2

    monkeypatch.setattr(
        "app.services.content_ingestion_service.ContentIngestionService.fetch_source",
        fake_fetch_source,
    )

    result = poll_due_sources()
    assert result["errors"] == 0
    assert result["sources_polled"] == 1
    assert result["results"][0]["status"] == "success"


def test_extract_themes_stops_after_provider_quota_abort(
    monkeypatch: pytest.MonkeyPatch,
    db_session_factory,
) -> None:
    monkeypatch.setattr("app.tasks.theme_discovery_tasks.SessionLocal", db_session_factory)
    monkeypatch.setattr(
        "app.tasks.theme_discovery_tasks._theme_automation_gate_result",
        lambda _db: None,
    )
    calls: list[str] = []

    class FakeThemeExtractionService:
        def __init__(self, db, pipeline: str) -> None:
            _ = db
            self.pipeline = pipeline

        def process_batch(self, limit: int = 50) -> dict:
            _ = limit
            calls.append(self.pipeline)
            if self.pipeline == "technical":
                return {
                    "processed": 0,
                    "total_mentions": 0,
                    "errors": 1,
                    "new_themes": [],
                    "aborted": True,
                    "abort_reason": "tokens per day limit reached",
                }
            return {
                "processed": 1,
                "total_mentions": 1,
                "errors": 0,
                "new_themes": [f"{self.pipeline}-theme"],
            }

    monkeypatch.setattr(
        "app.services.theme_extraction_service.ThemeExtractionService",
        FakeThemeExtractionService,
    )

    result = extract_themes(limit=10)

    assert calls == ["fundamental", "technical"]
    assert result["processed"] == 1
    assert result["errors"] == 1
    assert result["aborted"] is True
    assert result["abort_reason"] == "tokens per day limit reached"
    assert result["pipeline_results"]["fundamental"]["processed"] == 1
    assert result["pipeline_results"]["technical"]["aborted"] is True


@pytest.mark.parametrize(
    ("task", "kwargs"),
    [
        (poll_due_sources, {}),
        (reprocess_failed_themes, {}),
        (extract_themes, {"limit": 10}),
        (calculate_theme_metrics, {}),
        (promote_candidate_themes, {}),
        (apply_lifecycle_policies, {}),
    ],
)
def test_theme_beat_tasks_use_shared_automation_gate(
    monkeypatch: pytest.MonkeyPatch,
    db_session_factory,
    task,
    kwargs,
) -> None:
    monkeypatch.setattr("app.tasks.theme_discovery_tasks.SessionLocal", db_session_factory)
    monkeypatch.setattr(
        "app.tasks.theme_discovery_tasks._theme_automation_gate_result",
        lambda _db: {
            "status": "skipped",
            "reason": "bootstrap_running",
            "message": "Theme automation is blocked until bootstrap finishes.",
            "timestamp": "2026-04-18T12:00:00+00:00",
        },
    )

    result = task(**kwargs)

    assert result["status"] == "skipped"
    assert result["reason"] == "bootstrap_running"


def test_poll_due_sources_skips_when_bootstrap_is_running(
    monkeypatch: pytest.MonkeyPatch,
    db_session_factory,
) -> None:
    session = db_session_factory()
    session.add(
        ContentSource(
            name="@alice",
            source_type="twitter",
            url="https://x.com/alice",
            is_active=True,
            fetch_interval_minutes=60,
        )
    )
    session.commit()
    session.close()

    monkeypatch.setattr("app.tasks.theme_discovery_tasks.SessionLocal", db_session_factory)
    monkeypatch.setattr(
        "app.tasks.theme_discovery_tasks.get_runtime_bootstrap_status",
        lambda _db: SimpleNamespace(
            bootstrap_required=True,
            empty_system=True,
            primary_market="US",
            enabled_markets=["US"],
            bootstrap_state="running",
            supported_markets=["US", "HK", "JP", "TW"],
        ),
    )

    result = poll_due_sources()

    assert result["status"] == "skipped"
    assert result["reason"] == "bootstrap_running"


def test_extract_themes_skips_when_no_active_sources_exist(
    monkeypatch: pytest.MonkeyPatch,
    db_session_factory,
) -> None:
    monkeypatch.setattr("app.tasks.theme_discovery_tasks.SessionLocal", db_session_factory)
    monkeypatch.setattr(
        "app.tasks.theme_discovery_tasks.get_runtime_bootstrap_status",
        lambda _db: SimpleNamespace(
            bootstrap_required=False,
            empty_system=False,
            primary_market="US",
            enabled_markets=["US"],
            bootstrap_state="ready",
            supported_markets=["US", "HK", "JP", "TW"],
        ),
    )

    result = extract_themes(limit=10)

    assert result["status"] == "skipped"
    assert result["reason"] == "no_active_content_sources"


def test_extract_themes_runs_when_bootstrap_is_ready_and_source_exists(
    monkeypatch: pytest.MonkeyPatch,
    db_session_factory,
) -> None:
    session = db_session_factory()
    session.add(
        ContentSource(
            name="@alice",
            source_type="twitter",
            url="https://x.com/alice",
            is_active=True,
            fetch_interval_minutes=60,
        )
    )
    session.commit()
    session.close()

    monkeypatch.setattr("app.tasks.theme_discovery_tasks.SessionLocal", db_session_factory)
    monkeypatch.setattr(
        "app.tasks.theme_discovery_tasks.get_runtime_bootstrap_status",
        lambda _db: SimpleNamespace(
            bootstrap_required=False,
            empty_system=False,
            primary_market="US",
            enabled_markets=["US"],
            bootstrap_state="ready",
            supported_markets=["US", "HK", "JP", "TW"],
        ),
    )
    monkeypatch.setattr(
        "app.services.ui_snapshot_service.safe_publish_themes_bootstrap_variants",
        lambda *_args, **_kwargs: None,
    )

    class FakeThemeExtractionService:
        def __init__(self, db, pipeline: str) -> None:
            _ = db
            self.pipeline = pipeline

        def process_batch(self, limit: int = 50) -> dict:
            _ = limit
            return {
                "processed": 1,
                "total_mentions": 1,
                "errors": 0,
                "new_themes": [f"{self.pipeline}-theme"],
                "aborted": False,
            }

    monkeypatch.setattr(
        "app.services.theme_extraction_service.ThemeExtractionService",
        FakeThemeExtractionService,
    )

    result = extract_themes(limit=10)

    assert result["processed"] == 2
    assert result["errors"] == 0
    assert result["new_themes"] == ["fundamental-theme", "technical-theme"]
