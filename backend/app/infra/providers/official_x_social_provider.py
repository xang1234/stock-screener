"""Fail-closed Official X list reader for the social-signal pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Callable
from zoneinfo import ZoneInfo

import httpx

from app.domain.social_signals.records import (
    SocialPostRecord, SocialReadRequest, SocialSourceBatch, SocialSourceOutcome,
)

Reservation = Callable[[object, int, int], int]
X_API_MIN_RESULTS = 5


class OfficialXSocialProvider:
    def __init__(self, *, bearer_token: str, reservation: Reservation,
                 client: httpx.Client, daily_post_limit: int,
                 budget_timezone: str, max_results_per_page: int = 100,
                 retry_delay_seconds: float = 1.0,
                 sleep: Callable[[float], None] = time.sleep):
        if reservation is None:
            raise ValueError("official_reservation_required")
        if not X_API_MIN_RESULTS <= max_results_per_page <= 100:
            raise ValueError("invalid_max_results_per_page")
        self._token = bearer_token
        self._reserve = reservation
        self._client = client
        self._daily_limit = daily_post_limit
        self._timezone = ZoneInfo(budget_timezone)
        self._page_size = max_results_per_page
        self._retry_delay = retry_delay_seconds
        self._sleep = sleep

    def read_source(self, request: SocialReadRequest) -> SocialSourceBatch:
        if not isinstance(self._token, str) or not self._token.strip():
            return self._failed(request, "reauthentication_required")
        posts, cursor = [], request.application_progress if request.intent == "initial" else None
        effective_limit = min(request.limit, 5) if request.intent == "test" else request.limit
        reached_boundary = False
        while len(posts) < effective_limit:
            capacity = min(self._page_size, effective_limit - len(posts))
            response = self._request_page(request, cursor, capacity)
            if isinstance(response, SocialSourceBatch):
                return response
            try:
                page_posts, next_cursor = self._normalize(response, request)
            except (TypeError, ValueError, KeyError):
                return self._failed(request, "invalid_provider_schema")
            for post in page_posts:
                if post.created_at < request.target_published_after:
                    reached_boundary = True
                    break
                if len(posts) < effective_limit:
                    posts.append(post)
            cursor = next_cursor
            if reached_boundary or not cursor or not page_posts:
                break
        ordered = tuple(posts)
        timestamps = [post.created_at for post in ordered]
        observed_initial_window = (
            request.intent == "initial" and (reached_boundary or not cursor)
        )
        outcome = SocialSourceOutcome(
            read_status="success", processing_status="pending",
            history_status=("observed_window" if observed_initial_window
                            else "warming_up" if request.intent == "initial" else "limited"),
            coverage_reason_codes=(() if observed_initial_window
                                   else ("bounded_provider_read",)), known_gap_intervals=(),
            observed_oldest_at=min(timestamps) if timestamps else None,
            observed_newest_at=max(timestamps) if timestamps else None,
            received_count=len(ordered), committed_progress=None, error_code=None,
            proposed_progress=cursor if request.intent != "test" else None)
        return SocialSourceBatch(request, ordered, outcome)

    def _request_page(self, request, cursor, capacity):
        day = request.observed_at.astimezone(self._timezone).date()
        provider_capacity = max(X_API_MIN_RESULTS, capacity)
        params = {
            "tweet.fields": "created_at,public_metrics,author_id,referenced_tweets,entities",
            "expansions": "author_id", "user.fields": "username",
            "max_results": str(provider_capacity),
        }
        if cursor:
            params["pagination_token"] = cursor
        for attempt in range(2):
            granted = self._reserve(day, provider_capacity, self._daily_limit)
            if (not isinstance(granted, int) or isinstance(granted, bool)
                    or not 0 <= granted <= provider_capacity):
                return self._failed(request, "invalid_reservation_grant")
            if granted < X_API_MIN_RESULTS:
                return self._failed(request, "daily_limit_exhausted")
            params["max_results"] = str(granted)
            try:
                response = self._client.get(
                    f"https://api.x.com/2/lists/{request.list_id}/tweets", params=params,
                    headers={"Authorization": f"Bearer {self._token.strip()}"})
            except httpx.TransportError:
                if attempt == 0:
                    self._sleep(self._retry_delay)
                    continue
                return self._failed(request, "provider_network_error")
            if response.status_code == 429:
                reset = self._reset_at(response.headers.get("x-rate-limit-reset"))
                return self._failed(request, "rate_limited", reset)
            if response.status_code in {401, 403}:
                return self._failed(request, "reauthentication_required")
            if not 200 <= response.status_code < 300:
                return self._failed(request, "provider_error")
            try:
                return response.json()
            except ValueError:
                return self._failed(request, "invalid_provider_json")
        raise AssertionError("unreachable")

    @staticmethod
    def _reset_at(value):
        try:
            return datetime.fromtimestamp(int(value), timezone.utc) if value is not None else None
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _normalize(payload, request):
        if not isinstance(payload, dict):
            raise ValueError("payload")
        if "errors" in payload:
            raise ValueError("provider_errors")
        data, meta = payload.get("data", []), payload.get("meta", {})
        includes = payload.get("includes", {})
        if not isinstance(data, list) or not isinstance(meta, dict) or not isinstance(includes, dict):
            raise ValueError("envelope")
        users = includes.get("users", [])
        if not isinstance(users, list):
            raise ValueError("users")
        authors = {}
        for user in users:
            if not isinstance(user, dict) or not isinstance(user.get("id"), str) or not isinstance(user.get("username"), str):
                raise ValueError("user")
            authors[user["id"]] = user["username"]
        posts = []
        for raw in data:
            if not isinstance(raw, dict):
                raise ValueError("post")
            author_id = raw.get("author_id")
            if author_id not in authors:
                raise ValueError("author")
            references = OfficialXSocialProvider._aliased(raw, "referenced_tweets", "referenced_posts", default=[])
            if not isinstance(references, list):
                raise ValueError("references")
            if any(not isinstance(ref, dict) or not isinstance(ref.get("type"), str)
                   or not isinstance(ref.get("id"), str) for ref in references):
                raise ValueError("reference")
            metrics = raw.get("public_metrics", {})
            if not isinstance(metrics, dict):
                raise ValueError("metrics")
            reposts = OfficialXSocialProvider._aliased(metrics, "retweet_count", "repost_count")
            normalized_metrics = {
                "likes": metrics.get("like_count"), "reposts": reposts,
                "replies": metrics.get("reply_count"), "quotes": metrics.get("quote_count"),
                "bookmarks": metrics.get("bookmark_count"), "views": metrics.get("impression_count"),
            }
            for value in normalized_metrics.values():
                if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                    raise ValueError("metric")
            post_id, username = raw.get("id"), authors[author_id]
            if not isinstance(post_id, str) or not isinstance(raw.get("text"), str) or not isinstance(raw.get("created_at"), str):
                raise ValueError("required")
            url = f"https://x.com/{username}/status/{post_id}"
            posts.append(SocialPostRecord.from_untrusted({
                "id": post_id, "text": raw["text"], "created_at": raw["created_at"],
                "url": url, "canonical_url": url, "username": username,
                "is_repost": any(ref.get("type") in {"retweeted", "reposted"} for ref in references if isinstance(ref, dict)),
                **normalized_metrics,
            }, provider="official", source_id=request.source_id, observed_at=request.observed_at))
        next_token = meta.get("next_token")
        if next_token is not None and (not isinstance(next_token, str) or not next_token):
            raise ValueError("next_token")
        return posts, next_token

    @staticmethod
    def _aliased(mapping, old, new, default=None):
        if old in mapping and new in mapping:
            raise ValueError("conflicting_aliases")
        return mapping.get(old, mapping.get(new, default))

    @staticmethod
    def _failed(request, code, reset_at=None):
        return SocialSourceBatch(request, (), SocialSourceOutcome(
            read_status="failed", processing_status="failed", history_status="limited",
            coverage_reason_codes=(code,), known_gap_intervals=(), observed_oldest_at=None,
            observed_newest_at=None, received_count=0, committed_progress=None,
            error_code=code, proposed_progress=None, rate_limit_reset_at=reset_at))
