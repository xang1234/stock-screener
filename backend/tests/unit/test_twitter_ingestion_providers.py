"""Unit tests for official X API and private xui twitter ingestion providers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.database import Base
from app.models.app_settings import AppSetting
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
                    "meta": {"newest_id": "100"},
                    "includes": {"users": [{"id": "42", "username": "alice"}]},
                }
            )
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(provider_mod.requests, "get", fake_get)

    source = ContentSource(name="@alice", source_type="twitter", url="https://twitter.com/alice")
    rows = OfficialXTwitterFetcher().fetch(source, since=datetime(2026, 2, 28, tzinfo=timezone.utc))

    assert len(rows) == 1
    assert rows[0]["external_id"] == hashlib.md5("twitter:100".encode("utf-8")).hexdigest()
    assert rows[0]["title"] == ""
    assert rows[0]["content"] == "hello world"
    assert rows[0]["url"] == "https://x.com/alice/status/100"
    assert rows[0]["author"] == "@alice"
    assert rows[0]["published_at"] == datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
    assert rows[0]["_twitter_since_id"] == "100"
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
                "meta": {"newest_id": "200"},
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


def test_official_fetcher_follows_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "twitter_bearer_token", "token")
    monkeypatch.setattr(settings, "xui_limit_per_source", 50)
    monkeypatch.setattr(settings, "x_api_max_pages_per_source", 5)
    timeline_params: list[dict[str, object]] = []

    class FakeResponse:
        status_code = 200
        headers = {}
        text = "ok"

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_get(url, *, headers, params, timeout):
        if url.endswith("/2/users/by/username/alice"):
            return FakeResponse({"data": {"id": "42", "username": "alice"}})
        if url.endswith("/2/users/42/tweets"):
            timeline_params.append(dict(params))
            if "pagination_token" not in params:
                return FakeResponse(
                    {
                        "data": [
                            {"id": "101", "text": "new", "created_at": "2026-03-01T01:00:00Z", "author_id": "42"}
                        ],
                        "includes": {"users": [{"id": "42", "username": "alice"}]},
                        "meta": {"newest_id": "101", "next_token": "NEXT"},
                    }
                )
            return FakeResponse(
                {
                    "data": [
                        {"id": "100", "text": "old", "created_at": "2026-03-01T00:00:00Z", "author_id": "42"}
                    ],
                    "includes": {"users": [{"id": "42", "username": "alice"}]},
                    "meta": {},
                }
            )
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(provider_mod.requests, "get", fake_get)

    rows = OfficialXTwitterFetcher().fetch(
        ContentSource(name="@alice", source_type="twitter", url="@alice"),
        since=None,
    )

    assert [row["content"] for row in rows] == ["new", "old"]
    assert timeline_params[1]["pagination_token"] == "NEXT"
    assert all(row["_twitter_since_id"] == "101" for row in rows)


def test_official_fetcher_raises_when_page_cap_would_skip_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "twitter_bearer_token", "token")
    monkeypatch.setattr(settings, "x_api_max_pages_per_source", 1)

    class FakeResponse:
        status_code = 200
        headers = {}
        text = "ok"

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_get(url, *, headers, params, timeout):
        if url.endswith("/2/users/by/username/alice"):
            return FakeResponse({"data": {"id": "42", "username": "alice"}})
        return FakeResponse(
            {
                "data": [{"id": "101", "text": "new", "created_at": "2026-03-01T01:00:00Z"}],
                "meta": {"newest_id": "101", "next_token": "NEXT"},
            }
        )

    monkeypatch.setattr(provider_mod.requests, "get", fake_get)

    with pytest.raises(TwitterIngestionProviderError, match="pagination cap"):
        OfficialXTwitterFetcher().fetch(ContentSource(name="@alice", source_type="twitter", url="@alice"))


def test_official_fetcher_uses_persisted_since_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "twitter_bearer_token", "token")
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        source = ContentSource(name="@alice", source_type="twitter", url="@alice", is_active=True)
        db.add(source)
        db.flush()
        db.add(
            AppSetting(
                key=f"twitter.official_x_api.source.{source.id}.since_id",
                value="99",
                category="theme",
            )
        )
        db.commit()
        captured_params: list[dict[str, object]] = []

        class FakeResponse:
            status_code = 200
            headers = {}
            text = "ok"

            def __init__(self, payload):
                self._payload = payload

            def json(self):
                return self._payload

        def fake_get(url, *, headers, params, timeout):
            if url.endswith("/2/users/by/username/alice"):
                return FakeResponse({"data": {"id": "42", "username": "alice"}})
            captured_params.append(dict(params))
            return FakeResponse({"data": [], "meta": {}})

        monkeypatch.setattr(provider_mod.requests, "get", fake_get)

        OfficialXTwitterFetcher().fetch(source, since=None)

        assert captured_params[0]["since_id"] == "99"
        assert "start_time" not in captured_params[0]
    finally:
        db.close()


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


def test_private_xui_requires_stable_read_source_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider_mod.importlib, "import_module", lambda _name: SimpleNamespace(fetch=lambda **_kwargs: []))

    with pytest.raises(TwitterIngestionProviderError, match="read_source"):
        provider_mod._load_private_xui_bindings()
