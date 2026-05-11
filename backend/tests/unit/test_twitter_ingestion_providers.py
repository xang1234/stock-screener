"""Unit tests for official X API and private xui twitter ingestion providers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from types import SimpleNamespace

import pytest

from app.config import settings
from app.models.theme import ContentSource
from app.services import twitter_ingestion_providers as provider_mod
from app.services.twitter_ingestion_providers import (
    OfficialXTwitterFetcher,
    PrivateXUIFetcher,
    TwitterIngestionProviderError,
    build_twitter_fetcher,
)


def test_provider_router_defaults_to_official_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "x_ingest_provider", "official")

    fetcher = build_twitter_fetcher()

    assert isinstance(fetcher, OfficialXTwitterFetcher)


def test_provider_router_uses_private_xui_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "x_ingest_provider", "xui")

    fetcher = build_twitter_fetcher()

    assert isinstance(fetcher, PrivateXUIFetcher)


def test_provider_router_rejects_invalid_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "x_ingest_provider", "selenium")

    with pytest.raises(TwitterIngestionProviderError, match="X_INGEST_PROVIDER"):
        build_twitter_fetcher()


def test_official_fetcher_maps_user_timeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "twitter_bearer_token", "token")
    monkeypatch.setattr(settings, "xui_limit_per_source", 50)
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeResponse:
        status_code = 200
        headers = {
            "x-rate-limit-limit": "300",
            "x-rate-limit-remaining": "299",
            "x-rate-limit-reset": "1770000000",
        }

        def __init__(self, payload):
            self._payload = payload
            self.text = str(payload)

        def json(self):
            return self._payload

    def fake_get(url, *, headers, params, timeout):
        assert headers["Authorization"] == "Bearer token"
        calls.append((url, dict(params)))
        if url.endswith("/2/users/by/username/alice"):
            return FakeResponse({"data": {"id": "42", "username": "alice"}})
        if url.endswith("/2/users/42/tweets"):
            return FakeResponse(
                {
                    "data": [
                        {
                            "id": "100",
                            "text": "hello world",
                            "created_at": "2026-03-01T00:00:00.000Z",
                            "author_id": "42",
                        }
                    ],
                    "includes": {"users": [{"id": "42", "username": "alice"}]},
                }
            )
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(provider_mod.requests, "get", fake_get)

    source = ContentSource(name="@alice", source_type="twitter", url="https://twitter.com/alice")
    rows = OfficialXTwitterFetcher().fetch(source, since=datetime(2026, 2, 28, tzinfo=timezone.utc))

    assert rows == [
        {
            "external_id": hashlib.md5("twitter:100".encode("utf-8")).hexdigest(),
            "title": "",
            "content": "hello world",
            "url": "https://x.com/alice/status/100",
            "author": "@alice",
            "published_at": datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc),
        }
    ]
    timeline_url, timeline_params = calls[1]
    assert timeline_url.endswith("/2/users/42/tweets")
    assert timeline_params["start_time"] == "2026-02-28T00:00:00Z"
    assert timeline_params["tweet.fields"] == "created_at,author_id"
    assert timeline_params["expansions"] == "author_id"


def test_official_fetcher_maps_list_timeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "twitter_bearer_token", "token")
    monkeypatch.setattr(settings, "xui_limit_per_source", 20)

    class FakeResponse:
        status_code = 200
        headers = {}
        text = "ok"

        def json(self):
            return {
                "data": [
                    {
                        "id": "200",
                        "text": "list item",
                        "created_at": "2026-03-01T01:00:00Z",
                        "author_id": "7",
                    }
                ],
                "includes": {"users": [{"id": "7", "username": "bob"}]},
            }

    def fake_get(url, *, headers, params, timeout):
        assert url.endswith("/2/lists/84839422/tweets")
        assert params["max_results"] == 20
        return FakeResponse()

    monkeypatch.setattr(provider_mod.requests, "get", fake_get)

    source = ContentSource(name="Tech List", source_type="twitter", url="https://x.com/i/lists/84839422")
    rows = OfficialXTwitterFetcher().fetch(source, since=None)

    assert rows[0]["url"] == "https://x.com/bob/status/200"
    assert rows[0]["author"] == "@bob"


def test_official_fetcher_requires_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "twitter_bearer_token", "")
    source = ContentSource(name="@alice", source_type="twitter", url="@alice")

    with pytest.raises(TwitterIngestionProviderError, match="TWITTER_BEARER_TOKEN"):
        OfficialXTwitterFetcher().fetch(source, since=None)


def test_official_fetcher_raises_on_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "twitter_bearer_token", "token")

    class FakeResponse:
        status_code = 429
        headers = {"x-rate-limit-reset": "1770000000"}
        text = "rate limited"

        def json(self):
            return {"title": "Too Many Requests"}

    monkeypatch.setattr(provider_mod.requests, "get", lambda *args, **kwargs: FakeResponse())

    source = ContentSource(name="@alice", source_type="twitter", url="@alice")
    with pytest.raises(TwitterIngestionProviderError, match="rate limit"):
        OfficialXTwitterFetcher().fetch(source, since=None)


def test_private_xui_missing_package_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_bindings():
        raise TwitterIngestionProviderError("Private xui package is not available. Install with `pip install git+ssh://git@github.com/xang1234/xui.git`.")

    monkeypatch.setattr(provider_mod, "_load_private_xui_bindings", missing_bindings)

    source = ContentSource(name="@alice", source_type="twitter", url="@alice")
    with pytest.raises(TwitterIngestionProviderError, match="github.com/xang1234/xui"):
        PrivateXUIFetcher().fetch(source, since=None)


def test_private_xui_normalizes_mocked_results(monkeypatch: pytest.MonkeyPatch) -> None:
    created_at = datetime(2026, 3, 1, 2, 0, tzinfo=timezone.utc)
    bindings = SimpleNamespace(
        read_source=lambda **_kwargs: [
            SimpleNamespace(
                id="300",
                text="private item",
                author_handle="@carol",
                created_at=created_at,
                url=None,
            )
        ]
    )
    monkeypatch.setattr(provider_mod, "_load_private_xui_bindings", lambda: bindings)

    source = ContentSource(name="@carol", source_type="twitter", url="@carol")
    rows = PrivateXUIFetcher().fetch(source, since=None)

    assert rows == [
        {
            "external_id": hashlib.md5("twitter:300".encode("utf-8")).hexdigest(),
            "title": "",
            "content": "private item",
            "url": "https://x.com/carol/status/300",
            "author": "@carol",
            "published_at": created_at,
        }
    ]
