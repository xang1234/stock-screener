"""Endpoint tests for pipeline-scoped theme content browser APIs."""

from __future__ import annotations

from datetime import datetime

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DatabaseError

import app.api.v1.themes as themes_api
import app.services.theme_content_recovery_service as recovery_service
from app.main import app
from app.models.theme import ContentSource
from app.schemas.theme import ContentItemWithThemesResponse
from app.services import server_auth


@pytest_asyncio.fixture
async def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
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
    captured_kwargs: list[dict[str, object]] = []

    def _mock_fetch(*_args, **kwargs):
        captured_kwargs.append(kwargs)
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
    assert captured_kwargs[0]["limit"] == 500
    assert captured_kwargs[0]["offset"] == 0


@pytest.mark.asyncio
async def test_export_content_items_streams_multiple_pages(
    monkeypatch: pytest.MonkeyPatch,
    client,
):
    calls: list[tuple[int, int]] = []

    def _mock_fetch(*_args, **kwargs):
        offset = int(kwargs["offset"])
        limit = int(kwargs["limit"])
        calls.append((offset, limit))
        if offset == 0:
            return (
                [
                    ContentItemWithThemesResponse(
                        id=201,
                        title="Page one",
                        content="Body one",
                        url="https://example.com/one",
                        source_type="news",
                        source_name="Example Source",
                        author="Author one",
                        published_at=datetime(2026, 3, 1, 11, 0, 0),
                        themes=[],
                        sentiments=[],
                        primary_sentiment=None,
                        tickers=[],
                        processing_status="processed",
                    )
                ],
                2,
            )
        if offset == 1:
            return (
                [
                    ContentItemWithThemesResponse(
                        id=202,
                        title="Page two",
                        content="Body two",
                        url="https://example.com/two",
                        source_type="news",
                        source_name="Example Source",
                        author="Author two",
                        published_at=datetime(2026, 3, 1, 11, 5, 0),
                        themes=[],
                        sentiments=[],
                        primary_sentiment=None,
                        tickers=[],
                        processing_status="processed",
                    )
                ],
                2,
            )
        return ([], 2)

    monkeypatch.setattr("app.api.v1.themes._fetch_content_items_with_themes", _mock_fetch)

    response = await client.get(
        "/api/v1/themes/content/export",
        params={"pipeline": "fundamental", "source_type": "news"},
    )

    assert response.status_code == 200
    csv_text = response.content.decode("utf-8-sig")
    assert "Page one" in csv_text
    assert "Page two" in csv_text
    assert calls == [(0, 500), (1, 500)]


@pytest.mark.asyncio
async def test_list_content_items_recovers_from_corrupt_theme_content_storage(
    monkeypatch: pytest.MonkeyPatch,
    client,
):
    calls = {"count": 0, "reindex": 0, "reset": 0}
    captured_classifier_kwargs: dict[str, object] = {}

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
    monkeypatch.setattr(
        "app.api.v1.themes._corruption_targets_theme_content_storage",
        lambda **kwargs: captured_classifier_kwargs.update(kwargs) or True,
    )
    monkeypatch.setattr(
        "app.api.v1.themes._reset_corrupt_theme_content_storage",
        lambda exc: calls.__setitem__("reset", calls["reset"] + 1),
    )
    monkeypatch.setattr("app.api.v1.themes.get_session_factory", lambda: (lambda: _DummySession()))

    response = await client.get(
        "/api/v1/themes/content",
        params={"pipeline": "technical", "source_type": "twitter", "limit": 10, "offset": 0},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["title"] == "Recovered article"
    assert calls["count"] == 3
    assert calls["reindex"] == 1
    assert calls["reset"] == 1
    assert captured_classifier_kwargs["pipeline"] == "technical"
    assert captured_classifier_kwargs["source_type"] == "twitter"


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
    monkeypatch.setattr("app.api.v1.themes._corruption_targets_theme_content_storage", lambda **kwargs: False)
    monkeypatch.setattr(
        "app.api.v1.themes._reset_corrupt_theme_content_storage",
        lambda exc: calls.__setitem__("reset", calls["reset"] + 1),
    )
    monkeypatch.setattr("app.api.v1.themes.get_session_factory", lambda: (lambda: _DummySession()))

    items, total = themes_api._fetch_content_items_with_themes_with_recovery(object(), pipeline="fundamental")

    assert items == []
    assert total == 0
    assert calls["count"] == 2
    assert calls["reindex"] == 1
    assert calls["reset"] == 0


def test_theme_content_recovery_does_not_reset_for_non_resettable_corruption(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    reset_calls = {"count": 0, "reindex": 0}
    caplog.set_level("ERROR")

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
    monkeypatch.setattr("app.api.v1.themes._corruption_targets_theme_content_storage", lambda **kwargs: False)
    monkeypatch.setattr(
        "app.api.v1.themes._reset_corrupt_theme_content_storage",
        lambda exc: reset_calls.__setitem__("count", reset_calls["count"] + 1),
    )
    monkeypatch.setattr("app.api.v1.themes.get_session_factory", lambda: (lambda: _DummySession()))

    with pytest.raises(DatabaseError):
        themes_api._fetch_content_items_with_themes_with_recovery(object(), pipeline="fundamental")

    assert reset_calls["reindex"] == 1
    assert reset_calls["count"] == 0
    assert "check_db_integrity.py --repair" in caplog.text


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
    monkeypatch.setattr("app.api.v1.themes._corruption_targets_theme_content_storage", lambda **kwargs: False)
    monkeypatch.setattr(
        "app.api.v1.themes._reset_corrupt_theme_content_storage",
        lambda exc: calls.__setitem__("reset", calls["reset"] + 1),
    )
    monkeypatch.setattr("app.api.v1.themes.get_session_factory", lambda: (lambda: _DummySession()))

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


@pytest.mark.asyncio
async def test_export_content_items_recovers_from_corrupt_theme_content_storage(
    monkeypatch: pytest.MonkeyPatch,
    client,
):
    calls = {"count": 0, "reindex": 0, "reset": 0}
    captured_classifier_kwargs: dict[str, object] = {}

    class _DummySession:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, tb):
            return False

    def _mock_fetch(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] <= 2:
            raise DatabaseError(
                "SELECT count(*) FROM content_items JOIN content_sources",
                {},
                Exception("database disk image is malformed"),
            )
        return (
            [
                ContentItemWithThemesResponse(
                    id=105,
                    title="Recovered export after reset",
                    content="Recovered export content",
                    url="https://example.com/export-reset",
                    source_type="twitter",
                    source_name="@tech_source",
                    author="tech_source",
                    published_at=datetime(2026, 3, 2, 9, 45, 0),
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
    monkeypatch.setattr(
        "app.api.v1.themes._corruption_targets_theme_content_storage",
        lambda **kwargs: captured_classifier_kwargs.update(kwargs) or True,
    )
    monkeypatch.setattr(
        "app.api.v1.themes._reset_corrupt_theme_content_storage",
        lambda exc: calls.__setitem__("reset", calls["reset"] + 1),
    )
    monkeypatch.setattr("app.api.v1.themes.get_session_factory", lambda: (lambda: _DummySession()))

    response = await client.get(
        "/api/v1/themes/content/export",
        params={"pipeline": "technical", "source_type": "twitter"},
    )

    assert response.status_code == 200
    csv_text = response.content.decode("utf-8-sig")
    assert "Recovered export after reset" in csv_text
    assert calls["count"] == 3
    assert calls["reindex"] == 1
    assert calls["reset"] == 1
    assert captured_classifier_kwargs["pipeline"] == "technical"
    assert captured_classifier_kwargs["source_type"] == "twitter"


def test_theme_content_storage_classifier_uses_filtered_browser_query(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class _SourceProbe:
        def filter(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def all(self):
            return [(1,)]

    class _ProbeQuery:
        def order_by(self, *_args, **_kwargs):
            return self

        def count(self):
            raise DatabaseError(
                "SELECT count(*) FROM content_items JOIN content_sources",
                {},
                Exception("database disk image is malformed"),
            )

    class _DummySession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def query(self, *_args, **_kwargs):
            return _SourceProbe()

    monkeypatch.setattr("app.api.v1.themes.get_session_factory", lambda: (lambda: _DummySession()))
    monkeypatch.setattr(
        "app.api.v1.themes._resolve_source_ids_for_pipeline",
        lambda _db, pipeline: [11] if pipeline == "technical" else [],
    )
    monkeypatch.setattr(
        "app.api.v1.themes._build_content_items_browser_base_query",
        lambda _db, **kwargs: captured.update(kwargs) or _ProbeQuery(),
    )

    assert themes_api._corruption_targets_theme_content_storage(
        source_type="twitter",
        pipeline="technical",
        date_from="2026-03-01",
        date_to="2026-03-21",
    )
    assert captured["source_type"] == "twitter"
    assert captured["pipeline"] == "technical"
    assert captured["date_from"] == "2026-03-01"
    assert captured["date_to"] == "2026-03-21"
    assert captured["pipeline_source_ids"] == [11]


def test_theme_content_storage_classifier_skips_reset_for_content_source_corruption(
    monkeypatch: pytest.MonkeyPatch,
):
    class _SourceProbe:
        def filter(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def all(self):
            raise DatabaseError(
                "SELECT id FROM content_sources WHERE is_active = 1",
                {},
                Exception("database disk image is malformed"),
            )

    class _DummySession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def query(self, *_args, **_kwargs):
            return _SourceProbe()

    monkeypatch.setattr("app.api.v1.themes.get_session_factory", lambda: (lambda: _DummySession()))
    monkeypatch.setattr(
        "app.api.v1.themes._build_content_items_browser_base_query",
        lambda *_args, **_kwargs: pytest.fail("browser query probe should not run when content_sources is corrupt"),
    )

    assert not themes_api._corruption_targets_theme_content_storage(
        source_type="twitter",
        pipeline="technical",
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
    assert (now - rewind_at).days >= recovery_service._THEME_CONTENT_RESET_LOOKBACK_DAYS - 1
