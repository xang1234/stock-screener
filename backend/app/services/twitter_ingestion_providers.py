"""Twitter/X ingestion providers for theme content sources."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import logging
from typing import Any, Callable
from urllib.parse import urlparse, urlunparse

import requests

from ..config import settings
from ..models.theme import ContentSource

logger = logging.getLogger(__name__)

_X_API_BASE_URL = "https://api.x.com"
_REQUEST_TIMEOUT_SECONDS = 30


class TwitterIngestionProviderError(RuntimeError):
    """Raised when a Twitter/X ingestion provider cannot fetch a source."""


def build_twitter_fetcher():
    provider = _normalize_provider(settings.x_ingest_provider)
    if provider == "official":
        return OfficialXTwitterFetcher()
    if provider == "xui":
        return PrivateXUIFetcher()
    raise TwitterIngestionProviderError(
        "Unsupported X_INGEST_PROVIDER value "
        f"'{settings.x_ingest_provider}'. Expected 'official' or 'xui'."
    )


class OfficialXTwitterFetcher:
    """Fetch Twitter/X source content through the official X API v2."""

    provider_name = "official_x_api"

    def fetch(self, source: ContentSource, since: datetime | None = None) -> list[dict[str, Any]]:
        token = (settings.twitter_bearer_token or "").strip()
        if not token:
            raise TwitterIngestionProviderError(
                "TWITTER_BEARER_TOKEN is required for official X API ingestion."
            )

        locator = _normalized_locator(source)
        source_ref = _parse_source_locator(locator, source)
        if source_ref.kind == "user":
            user = self._lookup_user(source_ref.value, token)
            payload = self._get(
                f"/2/users/{user['id']}/tweets",
                token,
                params=_timeline_params(since),
            )
            fallback_author = str(user.get("username") or source_ref.value)
        else:
            payload = self._get(
                f"/2/lists/{source_ref.value}/tweets",
                token,
                params=_timeline_params(since),
            )
            fallback_author = None

        rows = _records_from_api_payload(payload, source, fallback_author=fallback_author)
        logger.info(
            "twitter_fetch provider=%s source_kind=%s source=%s items_fetched=%d",
            self.provider_name,
            source_ref.kind,
            source.name,
            len(rows),
        )
        return rows

    def _lookup_user(self, username: str, token: str) -> dict[str, Any]:
        payload = self._get(
            f"/2/users/by/username/{username.lstrip('@')}",
            token,
            params={"user.fields": "username"},
        )
        user = payload.get("data")
        if not isinstance(user, dict) or not user.get("id"):
            raise TwitterIngestionProviderError(f"Official X API did not return user id for @{username}.")
        return user

    def _get(self, path: str, token: str, *, params: dict[str, object]) -> dict[str, Any]:
        response = requests.get(
            f"{_X_API_BASE_URL}{path}",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        _log_rate_limit_headers(path, response.headers)
        if response.status_code == 429:
            reset = response.headers.get("x-rate-limit-reset")
            raise TwitterIngestionProviderError(
                "Official X API rate limit reached"
                + (f"; reset={reset}" if reset else "")
                + "."
            )
        if response.status_code >= 400:
            detail = _response_error_detail(response)
            raise TwitterIngestionProviderError(
                f"Official X API request failed ({response.status_code}) for {path}: {detail}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise TwitterIngestionProviderError(
                f"Official X API returned non-JSON response for {path}."
            ) from exc
        if not isinstance(payload, dict):
            raise TwitterIngestionProviderError(f"Official X API returned unexpected payload for {path}.")
        return payload


class PrivateXUIFetcher:
    """Fetch Twitter/X source content through the private xui package."""

    provider_name = "xui"

    def fetch(self, source: ContentSource, since: datetime | None = None) -> list[dict[str, Any]]:
        bindings = _load_private_xui_bindings()
        locator = _normalized_locator(source)
        source_ref = _parse_source_locator(locator, source)
        raw_items = bindings.read_source(
            source=source_ref,
            locator=locator,
            source_name=source.name,
            since=since,
            limit=settings.xui_limit_per_source,
        )
        rows = [
            _record_from_private_item(item, source, fallback_author=source_ref.value if source_ref.kind == "user" else None)
            for item in raw_items or []
        ]
        logger.info(
            "twitter_fetch provider=%s source_kind=%s source=%s items_fetched=%d",
            self.provider_name,
            source_ref.kind,
            source.name,
            len(rows),
        )
        return rows


@dataclass(frozen=True)
class _SourceRef:
    kind: str
    value: str
    label: str


@dataclass(frozen=True)
class _PrivateXUIBindings:
    read_source: Callable[..., Any]


def _normalize_provider(raw: str | None) -> str:
    provider = str(raw or "").strip().lower()
    return provider or "official"


def _normalized_locator(source: ContentSource) -> str:
    raw = (source.url or source.name or "").strip()
    if not raw:
        raise TwitterIngestionProviderError(f"Twitter source '{source.name}' is missing url/identifier.")
    parsed = urlparse(raw)
    host = parsed.netloc.lower()
    if host in {"twitter.com", "www.twitter.com", "mobile.twitter.com"}:
        parsed = parsed._replace(netloc="x.com")
        raw = urlunparse(parsed)
    return raw


def _parse_source_locator(locator: str, source: ContentSource) -> _SourceRef:
    parsed = urlparse(locator)
    if parsed.netloc.lower() in {"x.com", "www.x.com"}:
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 3 and parts[0] == "i" and parts[1] == "lists" and parts[2].isdigit():
            return _SourceRef(kind="list", value=parts[2], label=source.name)
        if len(parts) == 1 and _looks_like_handle(parts[0]):
            return _SourceRef(kind="user", value=parts[0].lstrip("@"), label=source.name)
    raw = locator.strip()
    if raw.startswith("@") and _looks_like_handle(raw[1:]):
        return _SourceRef(kind="user", value=raw[1:], label=source.name)
    if _looks_like_handle(raw):
        return _SourceRef(kind="user", value=raw, label=source.name)
    raise TwitterIngestionProviderError(
        f"Unable to parse twitter source locator '{locator}' for source '{source.name}'. "
        "Expected @handle, x.com/<handle>, or x.com/i/lists/<id>."
    )


def _looks_like_handle(value: str) -> bool:
    return bool(value) and len(value) <= 15 and all(ch.isalnum() or ch == "_" for ch in value)


def _timeline_params(since: datetime | None) -> dict[str, object]:
    params: dict[str, object] = {
        "max_results": max(5, min(int(settings.xui_limit_per_source), 100)),
        "tweet.fields": "created_at,author_id",
        "expansions": "author_id",
        "user.fields": "username",
    }
    since_bound = _normalize_datetime(since)
    if since_bound is not None:
        params["start_time"] = since_bound.strftime("%Y-%m-%dT%H:%M:%SZ")
    return params


def _records_from_api_payload(
    payload: dict[str, Any],
    source: ContentSource,
    *,
    fallback_author: str | None,
) -> list[dict[str, Any]]:
    users_by_id = {
        str(user.get("id")): str(user.get("username"))
        for user in payload.get("includes", {}).get("users", [])
        if isinstance(user, dict) and user.get("id") and user.get("username")
    }
    data = payload.get("data") or []
    if not isinstance(data, list):
        raise TwitterIngestionProviderError("Official X API returned unexpected timeline data.")
    return [_record_from_api_tweet(item, source, users_by_id, fallback_author) for item in data if isinstance(item, dict)]


def _record_from_api_tweet(
    item: dict[str, Any],
    source: ContentSource,
    users_by_id: dict[str, str],
    fallback_author: str | None,
) -> dict[str, Any]:
    tweet_id = str(item.get("id") or "").strip()
    if not tweet_id:
        raise TwitterIngestionProviderError("Official X API returned tweet without id.")
    author = users_by_id.get(str(item.get("author_id"))) or fallback_author or source.name
    return {
        "external_id": hashlib.md5(f"twitter:{tweet_id}".encode("utf-8")).hexdigest(),
        "title": "",
        "content": str(item.get("text") or ""),
        "url": _tweet_url(tweet_id, author),
        "author": _format_author(author),
        "published_at": _parse_x_datetime(item.get("created_at")),
    }


def _record_from_private_item(
    item: Any,
    source: ContentSource,
    *,
    fallback_author: str | None,
) -> dict[str, Any]:
    tweet_id = str(
        _first_attr(item, "tweet_id", "id", "post_id", "rest_id")
        or ""
    ).strip()
    if not tweet_id:
        raise TwitterIngestionProviderError("Private xui returned tweet without id.")
    author = _first_attr(item, "author_handle", "author", "username", "handle") or fallback_author or source.name
    url = _first_attr(item, "url", "link")
    return {
        "external_id": hashlib.md5(f"twitter:{tweet_id}".encode("utf-8")).hexdigest(),
        "title": "",
        "content": str(_first_attr(item, "text", "content", "body") or ""),
        "url": str(url or _tweet_url(tweet_id, str(author))),
        "author": _format_author(str(author)),
        "published_at": _normalize_datetime(_first_attr(item, "created_at", "published_at", "timestamp")),
    }


def _tweet_url(tweet_id: str, author: str | None) -> str:
    normalized = str(author or "").strip().lstrip("@")
    if normalized and _looks_like_handle(normalized):
        return f"https://x.com/{normalized}/status/{tweet_id}"
    return f"https://x.com/i/web/status/{tweet_id}"


def _format_author(author: str | None) -> str:
    normalized = str(author or "").strip()
    if normalized and _looks_like_handle(normalized.lstrip("@")):
        return "@" + normalized.lstrip("@")
    return normalized


def _parse_x_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _normalize_datetime(value)
    raw = str(value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return _normalize_datetime(datetime.fromisoformat(raw))
    except ValueError:
        return None


def _normalize_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _parse_x_datetime(value)
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _first_attr(item: Any, *names: str) -> Any:
    for name in names:
        if isinstance(item, dict) and name in item:
            return item.get(name)
        if hasattr(item, name):
            return getattr(item, name)
    return None


def _log_rate_limit_headers(path: str, headers: Any) -> None:
    limit = headers.get("x-rate-limit-limit")
    remaining = headers.get("x-rate-limit-remaining")
    reset = headers.get("x-rate-limit-reset")
    if limit or remaining or reset:
        logger.info(
            "official_x_api_rate_limit path=%s limit=%s remaining=%s reset=%s",
            path,
            limit,
            remaining,
            reset,
        )


def _response_error_detail(response: Any) -> str:
    try:
        payload = response.json()
    except Exception:
        return str(getattr(response, "text", "") or "no response body")
    if isinstance(payload, dict):
        for key in ("detail", "title", "message"):
            if payload.get(key):
                return str(payload[key])
        errors = payload.get("errors")
        if errors:
            return str(errors)
    return str(payload)


def _load_private_xui_bindings() -> _PrivateXUIBindings:
    try:
        xui_mod = importlib.import_module("xui")
    except ModuleNotFoundError as exc:
        raise TwitterIngestionProviderError(
            "Private xui package is not available. Install with "
            "`pip install git+ssh://git@github.com/xang1234/xui.git`."
        ) from exc

    for attr in ("read_source", "fetch_source", "fetch", "read"):
        candidate = getattr(xui_mod, attr, None)
        if callable(candidate):
            return _PrivateXUIBindings(read_source=candidate)
    raise TwitterIngestionProviderError(
        "Private xui package is installed but no supported read function was found "
        "(expected one of: read_source, fetch_source, fetch, read)."
    )
