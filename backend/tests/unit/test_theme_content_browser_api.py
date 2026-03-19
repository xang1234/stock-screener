"""Endpoint tests for pipeline-scoped theme content browser APIs."""

from __future__ import annotations

from datetime import datetime

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DatabaseError

import app.api.v1.themes as themes_api
from app.main import app
from app.models.theme import ContentSource
from app.schemas.theme import ContentItemWithThemesResponse


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_list_content_items_accepts_pipeline_and_returns_processing_status(
    monkeypatch: pytest.MonkeyPatch,
    client,
):
    captured: dict[str, object] = {}

    def _mock_fetch(*_args, **kwargs):
        captured.update(kwargs)
        return (
            [
                ContentItemWithThemesResponse(
                    id=101,
                    title="Sample tweet",
                    content="Content",
                    url="https://x.com/example/status/101",
                    source_type="twitter",
                    source_name="@fund_source",
                    author="fund_source",
                    published_at=datetime(2026, 3, 1, 10, 0, 0),
                    themes=[],
                    sentiments=[],
                    primary_sentiment=None,
                    tickers=[],
                    processing_status="pending",
                )
            ],
            1,
        )

    monkeypatch.setattr("app.api.v1.themes._fetch_content_items_with_themes", _mock_fetch)

    response = await client.get(
        "/api/v1/themes/content",
        params={"pipeline": "fundamental", "source_type": "twitter", "limit": 10, "offset": 0},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["processing_status"] == "pending"
    assert captured["pipeline"] == "fundamental"


@pytest.mark.asyncio
async def test_export_content_items_includes_processing_status_column(
    monkeypatch: pytest.MonkeyPatch,
    client,
):
    def _mock_fetch(*_args, **_kwargs):
        return (
            [
                ContentItemWithThemesResponse(
                    id=102,
                    title="Another tweet",
                    content="Body",
                    url="https://x.com/example/status/102",
                    source_type="twitter",
                    source_name="@fund_source",
                    author="fund_source",
                    published_at=datetime(2026, 3, 1, 10, 5, 0),
                    themes=[],
                    sentiments=[],
                    primary_sentiment=None,
                    tickers=[],
                    processing_status="in_progress",
                )
            ],
            1,
        )

    monkeypatch.setattr("app.api.v1.themes._fetch_content_items_with_themes", _mock_fetch)

    response = await client.get(
        "/api/v1/themes/content/export",
        params={"pipeline": "fundamental", "source_type": "twitter"},
    )

    assert response.status_code == 200
    csv_text = response.content.decode("utf-8-sig")
    assert "Processing Status" in csv_text.splitlines()[0]
    assert "in_progress" in csv_text


@pytest.mark.asyncio
async def test_list_content_items_recovers_from_corrupt_theme_content_storage(
    monkeypatch: pytest.MonkeyPatch,
    client,
):
    calls = {"count": 0, "reindex": 0, "reset": 0}

    class _DummySession:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, tb):
            return False

    def _mock_fetch(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] <= 2:
            raise DatabaseError(
                "SELECT * FROM content_items",
                {},
                Exception("database disk image is malformed"),
            )
        return (
            [
                ContentItemWithThemesResponse(
                    id=103,
                    title="Recovered article",
                    content="Recovered content",
                    url="https://example.com/recovered",
                    source_type="news",
                    source_name="Recovered Source",
                    author="Reporter",
                    published_at=datetime(2026, 3, 2, 9, 0, 0),
                    themes=[],
                    sentiments=[],
                    primary_sentiment=None,
                    tickers=[],
                    processing_status="processed",
                )
            ],
            1,
        )

    monkeypatch.setattr("app.api.v1.themes._fetch_content_items_with_themes", _mock_fetch)
    monkeypatch.setattr(
        "app.api.v1.themes._attempt_reindex_theme_content_storage",
        lambda exc: calls.__setitem__("reindex", calls["reindex"] + 1) or False,
    )
    monkeypatch.setattr("app.api.v1.themes._corruption_targets_theme_content_storage", lambda: True)
    monkeypatch.setattr(
        "app.api.v1.themes._reset_corrupt_theme_content_storage",
        lambda exc: calls.__setitem__("reset", calls["reset"] + 1),
    )
    monkeypatch.setattr("app.api.v1.themes.SessionLocal", lambda: _DummySession())

    response = await client.get(
        "/api/v1/themes/content",
        params={"pipeline": "fundamental", "limit": 10, "offset": 0},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["title"] == "Recovered article"
    assert calls["count"] == 3
    assert calls["reindex"] == 1
    assert calls["reset"] == 1


def test_theme_content_recovery_retries_after_reindex_without_reset(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = {"count": 0, "reindex": 0, "reset": 0}

    class _DummySession:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, tb):
            return False

    def _mock_fetch(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise DatabaseError(
                "SELECT * FROM content_items",
                {},
                Exception("database disk image is malformed"),
            )
        return ([], 0)

    monkeypatch.setattr("app.api.v1.themes._fetch_content_items_with_themes", _mock_fetch)
    monkeypatch.setattr(
        "app.api.v1.themes._attempt_reindex_theme_content_storage",
        lambda exc: calls.__setitem__("reindex", calls["reindex"] + 1) or True,
    )
    monkeypatch.setattr("app.api.v1.themes._corruption_targets_theme_content_storage", lambda: False)
    monkeypatch.setattr(
        "app.api.v1.themes._reset_corrupt_theme_content_storage",
        lambda exc: calls.__setitem__("reset", calls["reset"] + 1),
    )
    monkeypatch.setattr("app.api.v1.themes.SessionLocal", lambda: _DummySession())

    items, total = themes_api._fetch_content_items_with_themes_with_recovery(object(), pipeline="fundamental")

    assert items == []
    assert total == 0
    assert calls["count"] == 2
    assert calls["reindex"] == 1
    assert calls["reset"] == 0


def test_theme_content_recovery_does_not_reset_for_non_resettable_corruption(
    monkeypatch: pytest.MonkeyPatch,
):
    reset_calls = {"count": 0, "reindex": 0}

    class _DummySession:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, tb):
            return False

    def _raise_non_resettable_corruption(*_args, **_kwargs):
        raise DatabaseError(
            "SELECT * FROM content_sources",
            {},
            Exception("database disk image is malformed"),
        )

    monkeypatch.setattr("app.api.v1.themes._fetch_content_items_with_themes", _raise_non_resettable_corruption)
    monkeypatch.setattr(
        "app.api.v1.themes._attempt_reindex_theme_content_storage",
        lambda exc: reset_calls.__setitem__("reindex", reset_calls["reindex"] + 1) or False,
    )
    monkeypatch.setattr("app.api.v1.themes._corruption_targets_theme_content_storage", lambda: False)
    monkeypatch.setattr(
        "app.api.v1.themes._reset_corrupt_theme_content_storage",
        lambda exc: reset_calls.__setitem__("count", reset_calls["count"] + 1),
    )
    monkeypatch.setattr("app.api.v1.themes.SessionLocal", lambda: _DummySession())

    with pytest.raises(DatabaseError):
        themes_api._fetch_content_items_with_themes_with_recovery(object(), pipeline="fundamental")

    assert reset_calls["reindex"] == 1
    assert reset_calls["count"] == 0


@pytest.mark.asyncio
async def test_export_content_items_recovers_from_reindex_without_reset(
    monkeypatch: pytest.MonkeyPatch,
    client,
):
    calls = {"count": 0, "reindex": 0, "reset": 0}

    class _DummySession:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, tb):
            return False

    def _mock_fetch(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise DatabaseError(
                "SELECT content_items.id FROM content_items JOIN content_sources",
                {},
                Exception("database disk image is malformed"),
            )
        return (
            [
                ContentItemWithThemesResponse(
                    id=104,
                    title="Recovered export article",
                    content="Recovered export content",
                    url="https://example.com/export-recovered",
                    source_type="news",
                    source_name="Recovered Source",
                    author="Reporter",
                    published_at=datetime(2026, 3, 2, 9, 30, 0),
                    themes=[],
                    sentiments=[],
                    primary_sentiment=None,
                    tickers=[],
                    processing_status="processed",
                )
            ],
            1,
        )

    monkeypatch.setattr("app.api.v1.themes._fetch_content_items_with_themes", _mock_fetch)
    monkeypatch.setattr(
        "app.api.v1.themes._attempt_reindex_theme_content_storage",
        lambda exc: calls.__setitem__("reindex", calls["reindex"] + 1) or True,
    )
    monkeypatch.setattr("app.api.v1.themes._corruption_targets_theme_content_storage", lambda: False)
    monkeypatch.setattr(
        "app.api.v1.themes._reset_corrupt_theme_content_storage",
        lambda exc: calls.__setitem__("reset", calls["reset"] + 1),
    )
    monkeypatch.setattr("app.api.v1.themes.SessionLocal", lambda: _DummySession())

    response = await client.get(
        "/api/v1/themes/content/export",
        params={"pipeline": "fundamental", "source_type": "twitter"},
    )

    assert response.status_code == 200
    csv_text = response.content.decode("utf-8-sig")
    assert "Recovered export article" in csv_text
    assert calls["count"] == 2
    assert calls["reindex"] == 1
    assert calls["reset"] == 0


def test_theme_content_storage_probe_queries_cover_browser_query_paths():
    queries = themes_api._theme_content_storage_probe_queries()

    assert len(queries) == 4
    assert any("ORDER BY published_at DESC" in query for query in queries)
    assert any("JOIN content_sources" in query and "content_sources.is_active = 1" in query for query in queries)
    assert any("FROM theme_mentions" in query and "content_item_id" in query for query in queries)
    assert any(
        "FROM content_item_pipeline_state" in query and "pipeline = 'technical'" in query
        for query in queries
    )


def test_rewind_theme_content_source_cursors_rewinds_fetch_window():
    engine = create_engine("sqlite:///:memory:")
    ContentSource.__table__.create(engine)
    now = datetime.utcnow()

    with engine.begin() as conn:
        conn.execute(
            ContentSource.__table__.insert(),
            [
                {
                    "name": "Source A",
                    "source_type": "news",
                    "url": "https://example.com/a",
                    "is_active": True,
                    "priority": 50,
                    "fetch_interval_minutes": 60,
                    "last_fetched_at": now,
                    "total_items_fetched": 123,
                    "pipelines": '["fundamental"]',
                }
            ],
        )
        themes_api._rewind_theme_content_source_cursors(conn)
        row = conn.execute(
            text("SELECT last_fetched_at, total_items_fetched FROM content_sources WHERE name = 'Source A'")
        ).fetchone()

    rewind_at = datetime.fromisoformat(str(row[0]))
    assert row[1] == 0
    assert rewind_at <= now
    assert (now - rewind_at).days >= themes_api._THEME_CONTENT_RESET_LOOKBACK_DAYS - 1
