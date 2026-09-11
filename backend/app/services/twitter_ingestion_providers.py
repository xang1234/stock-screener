"""Twitter/X ingestion providers for theme content sources."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse, urlunparse

import requests
from sqlalchemy.orm import object_session

from ..config import settings
from ..models.app_settings import AppSetting
from ..models.theme import ContentSource

logger = logging.getLogger(__name__)

_X_API_BASE_URL = "https://api.x.com"
_REQUEST_TIMEOUT_SECONDS = 30


class TwitterIngestionProviderError(RuntimeError):
    """Raised when a Twitter/X ingestion provider cannot fetch a source."""


def build_twitter_fetcher():
    provider = resolve_x_ingest_provider(settings.x_ingest_provider)
    if provider == "official":
        return OfficialXTwitterFetcher()
    if provider == "xui":
        return PrivateXUIFetcher()
    raise AssertionError(f"Unhandled X ingestion provider: {provider}")


def resolve_x_ingest_provider(raw: str | None) -> str:
    provider = _normalize_provider(raw)
    if provider in {"official", "xui"}:
        return provider
    raise TwitterIngestionProviderError(
        f"Unsupported X_INGEST_PROVIDER value '{raw}'. Expected 'official' or 'xui'."
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
        since_id = _get_official_since_id(source)
        if source_ref.kind == "user":
            user = self._lookup_user(source_ref.value, token)
            payload = self._fetch_timeline(
                f"/2/users/{user['id']}/tweets",
                token,
                since=since,
                since_id=since_id,
            )
            fallback_author = str(user.get("username") or source_ref.value)
        else:
            payload = self._fetch_timeline(
                f"/2/lists/{source_ref.value}/tweets",
                token,
                since=since,
                since_id=since_id,
            )
            fallback_author = None

        rows = _records_from_api_payload(payload, source, fallback_author=fallback_author)
        if payload.get("cap_reached"):
            meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
            logger.warning(
                "official_x_api_pagination_cap_reached source=%s newest_id=%s items_fetched=%d",
                source.name,
                meta.get("newest_id"),
                len(rows),
            )
        logger.info(
            "twitter_fetch provider=%s source_kind=%s source=%s items_fetched=%d",
            self.provider_name,
            source_ref.kind,
            source.name,
            len(rows),
        )
        return rows

    def _fetch_timeline(
        self,
        path: str,
        token: str,
        *,
        since: datetime | None,
        since_id: str | None,
    ) -> dict[str, Any]:
        max_pages = max(1, int(settings.x_api_max_pages_per_source))
        combined_data: list[dict[str, Any]] = []
        users_by_id: dict[str, dict[str, Any]] = {}
        media_by_key: dict[str, dict[str, Any]] = {}
        fallback_newest_id: str | None = None
        next_token: str | None = None

        for _page in range(max_pages):
            params = _timeline_params(since, since_id=since_id)
            if next_token:
                params["pagination_token"] = next_token
            payload = self._get(path, token, params=params)

            data = payload.get("data") or []
            if not isinstance(data, list):
                raise TwitterIngestionProviderError("Official X API returned unexpected timeline data.")
            combined_data.extend(item for item in data if isinstance(item, dict))

            for user in payload.get("includes", {}).get("users", []):
                if isinstance(user, dict) and user.get("id"):
                    users_by_id[str(user["id"])] = user
            for media in payload.get("includes", {}).get("media", []):
                if isinstance(media, dict) and isinstance(media.get("media_key"), str):
                    media_by_key[media["media_key"]] = media

            meta = payload.get("meta") or {}
            if isinstance(meta, dict):
                fallback_newest_id = fallback_newest_id or _normalize_tweet_id(meta.get("newest_id"))
                next_token = str(meta.get("next_token") or "").strip() or None
            else:
                next_token = None
            if not next_token:
                return _timeline_payload(combined_data, users_by_id, media_by_key, fallback_newest_id)

        return _timeline_payload(
            combined_data,
            users_by_id,
            media_by_key,
            fallback_newest_id,
            cap_reached=True,
        )

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
    """Compatibility boundary for the retired private theme-source integration."""

    provider_name = "xui"

    def fetch(self, source: ContentSource, since: datetime | None = None) -> list[dict[str, Any]]:
        raise TwitterIngestionProviderError(
            "The legacy private X theme-source integration is unsupported. "
            "Configure the social-signal xui CLI provider for list reads."
        )


@dataclass(frozen=True)
class _SourceRef:
    kind: str
    value: str
    label: str


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


def _timeline_params(since: datetime | None, *, since_id: str | None = None) -> dict[str, object]:
    params: dict[str, object] = {
        "max_results": max(5, min(int(settings.x_api_max_results_per_page), 100)),
        "tweet.fields": "created_at,author_id,entities",
        "expansions": "author_id,attachments.media_keys",
        "user.fields": "username",
        "media.fields": "media_key,type,url",
    }
    if since_id:
        params["since_id"] = since_id
        return params
    since_bound = _normalize_datetime(since)
    if since_bound is not None:
        params["start_time"] = since_bound.strftime("%Y-%m-%dT%H:%M:%SZ")
    return params


def _timeline_payload(
    data: list[dict[str, Any]],
    users_by_id: dict[str, dict[str, Any]],
    media_by_key: dict[str, dict[str, Any]],
    fallback_newest_id: str | None,
    *,
    cap_reached: bool = False,
) -> dict[str, Any]:
    payload = {
        "data": data,
        "includes": {"users": list(users_by_id.values()), "media": list(media_by_key.values())},
        "meta": {"newest_id": _max_tweet_id(data) or fallback_newest_id},
    }
    if cap_reached:
        payload["cap_reached"] = True
    return payload


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
    media_by_key = {
        str(media.get("media_key")): media
        for media in payload.get("includes", {}).get("media", [])
        if isinstance(media, dict) and media.get("media_key")
    }
    data = payload.get("data") or []
    if not isinstance(data, list):
        raise TwitterIngestionProviderError("Official X API returned unexpected timeline data.")
    newest_id = None
    meta = payload.get("meta") or {}
    if isinstance(meta, dict):
        newest_id = _normalize_tweet_id(meta.get("newest_id")) or _max_tweet_id(data)
    return [
        _record_from_api_tweet(item, source, users_by_id, media_by_key, fallback_author, since_id=newest_id)
        for item in data
        if isinstance(item, dict)
    ]


def _record_from_api_tweet(
    item: dict[str, Any],
    source: ContentSource,
    users_by_id: dict[str, str],
    media_by_key: dict[str, dict[str, Any]],
    fallback_author: str | None,
    *,
    since_id: str | None,
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
        "attachments": _attachment_refs(item, media_by_key),
        "_twitter_since_id": since_id or tweet_id,
    }


def _attachment_refs(item: dict[str, Any], media_by_key: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    attachments = item.get("attachments", {})
    if isinstance(attachments, dict):
        keys = attachments.get("media_keys", [])
        if isinstance(keys, list):
            for key in keys:
                media = media_by_key.get(str(key))
                if media is None or media.get("type") != "photo":
                    continue
                url = media.get("url")
                if _is_http_url(url):
                    refs.append({"kind": "image", "url": url})
    entities = item.get("entities", {})
    if isinstance(entities, dict):
        urls = entities.get("urls", [])
        if isinstance(urls, list):
            for entry in urls:
                if not isinstance(entry, dict):
                    continue
                url = entry.get("unwound_url") or entry.get("expanded_url") or entry.get("url")
                if _is_article_url(url):
                    refs.append({"kind": "article", "url": url})
    return [
        {"kind": kind, "url": url}
        for kind, url in dict.fromkeys((ref["kind"], ref["url"]) for ref in refs)
    ]


def _is_http_url(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)


def _is_article_url(value: Any) -> bool:
    if not _is_http_url(value):
        return False
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    x_hosts = {"x.com", "www.x.com", "twitter.com", "www.twitter.com", "mobile.twitter.com"}
    return (host not in x_hosts | {"t.co"} and not host.endswith(".twimg.com")) or (
        host in x_hosts and "/article/" in parsed.path
    )


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


def official_since_id_key(source_id: int) -> str:
    return f"twitter.official_x_api.source.{source_id}.since_id"


def _get_official_since_id(source: ContentSource) -> str | None:
    if source.id is None:
        return None
    session = object_session(source)
    if session is None:
        return None
    row = session.query(AppSetting).filter(AppSetting.key == official_since_id_key(source.id)).first()
    if row is None:
        return None
    return _normalize_tweet_id(row.value)


def _normalize_tweet_id(value: Any) -> str | None:
    normalized = str(value or "").strip()
    return normalized if normalized.isdigit() else None


def _max_tweet_id(items: list[dict[str, Any]]) -> str | None:
    ids = [_normalize_tweet_id(item.get("id")) for item in items if isinstance(item, dict)]
    numeric_ids = [int(tweet_id) for tweet_id in ids if tweet_id is not None]
    if not numeric_ids:
        return None
    return str(max(numeric_ids))
