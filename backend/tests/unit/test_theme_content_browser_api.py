"""Endpoint tests for pipeline-scoped theme content browser APIs."""

from __future__ import annotations

from datetime import datetime

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.exc import DatabaseError

from app.main import app
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
    calls = {"count": 0, "reset": 0}

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
    assert calls["count"] == 2
    assert calls["reset"] == 1
