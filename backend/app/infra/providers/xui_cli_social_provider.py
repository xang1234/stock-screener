"""Fail-closed subprocess boundary for the private xui reader CLI."""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import timedelta
from typing import Callable
from urllib.parse import urlparse

from app.domain.social_signals.records import (
    SocialPostRecord,
    SocialReadRequest,
    SocialSourceBatch,
    SocialSourceOutcome,
)

Runner = Callable[..., subprocess.CompletedProcess[str]]
_READ_KEYS = frozenset({
    "succeeded_sources", "failed_sources", "page_loads", "scroll_rounds",
    "seen_items", "outcomes", "items",
})
_OUTCOME_KEYS = frozenset({
    "source_id", "source_kind", "ok", "item_count", "page_loads",
    "scroll_rounds", "observed_ids", "error", "html_artifact_path",
    "selector_report_path",
})
_AUTH_REASONS = (
    "challenge", "login wall", "login_wall", "reauth", "reauthentication",
    "auth", "authentication", "unauthorized", "session",
)
_NETWORK_REASONS = (
    "network", "connection reset", "connection refused", "temporary failure",
    "timed out", "timeout", "dns",
)
_RETRY_DELAY_SECONDS = 30


class XuiCliSocialProvider:
    def __init__(
        self,
        *,
        config_path: str,
        profile: str,
        runner: Runner = subprocess.run,
        timeout_seconds: int = 60,
        cooldown_seconds: int = 3600,
    ) -> None:
        if not config_path or not profile:
            raise ValueError("xui_configuration_required")
        if timeout_seconds <= 0 or cooldown_seconds <= 0:
            raise ValueError("invalid_xui_timeout")
        self._config_path = config_path
        self._profile = profile
        self._runner = runner
        self._timeout = timeout_seconds
        self._cooldown = timedelta(seconds=cooldown_seconds)
        self._cooldown_until = None
        self._cooldown_code = None

    def read_source(self, request: SocialReadRequest) -> SocialSourceBatch:
        if self._cooldown_until is not None and request.observed_at < self._cooldown_until:
            return self._failed(
                request, self._cooldown_code or "provider_error",
                reset_at=self._cooldown_until,
            )

        auth_result = self._execute([
            "xui", "auth", "status", "--path", self._config_path,
            "--profile", self._profile, "--json",
        ])
        if isinstance(auth_result, str):
            return self._failed(request, auth_result)
        try:
            auth_payload = json.loads(auth_result.stdout)
        except (TypeError, json.JSONDecodeError):
            return self._failed(request, "invalid_provider_json")
        if not self._authenticated(auth_payload, auth_result.returncode):
            return self._failed(request, "reauthentication_required", cooldown=True)

        intent_cap = {"initial": 1000, "incremental": 200, "test": 5}[request.intent]
        effective_limit = min(request.limit, intent_cap)
        read_command = [
            "xui", "read", "--path", self._config_path, "--profile", self._profile,
            "--limit", str(effective_limit), "--json", "--sources", f"list:{request.list_id}",
        ]
        for attempt in range(2):
            read_result = self._execute(read_command)
            if isinstance(read_result, str):
                if read_result == "provider_timeout" and attempt == 0:
                    time.sleep(_RETRY_DELAY_SECONDS)
                    continue
                return self._failed(request, read_result)
            try:
                payload = json.loads(read_result.stdout)
            except (TypeError, json.JSONDecodeError):
                return self._failed(request, "invalid_provider_json")
            try:
                posts = self._normalize(payload, request, effective_limit)
            except (TypeError, ValueError, KeyError):
                return self._failed(request, "invalid_provider_schema")

            if read_result.returncode != 0 or payload["failed_sources"] != 0:
                code = self._failure_code(payload)
                if code == "provider_network_error" and attempt == 0:
                    time.sleep(_RETRY_DELAY_SECONDS)
                    continue
                return self._failed(
                    request, code,
                    cooldown=code not in {"provider_network_error", "provider_timeout"},
                )
            break

        timestamps = [post.created_at for post in posts]
        return SocialSourceBatch(request, tuple(posts), SocialSourceOutcome(
            read_status="success", processing_status="pending",
            history_status="warming_up" if request.intent == "initial" else "limited",
            coverage_reason_codes=("bounded_provider_read",), known_gap_intervals=(),
            observed_oldest_at=min(timestamps) if timestamps else None,
            observed_newest_at=max(timestamps) if timestamps else None,
            received_count=len(posts), committed_progress=None, error_code=None,
            proposed_progress=None,
        ))

    def _execute(self, command):
        try:
            return self._runner(
                command, capture_output=True, text=True, timeout=self._timeout, shell=False,
            )
        except subprocess.TimeoutExpired:
            return "provider_timeout"
        except (FileNotFoundError, OSError):
            return "provider_unavailable"

    @staticmethod
    def _authenticated(payload, returncode):
        return (
            returncode == 0 and isinstance(payload, dict)
            and payload.get("authenticated") is True
            and isinstance(payload.get("status_code"), str)
        )

    @staticmethod
    def _normalize(payload, request, limit):
        if not isinstance(payload, dict) or set(payload) != _READ_KEYS:
            raise ValueError("envelope")
        succeeded, failed = payload["succeeded_sources"], payload["failed_sources"]
        outcomes, items = payload["outcomes"], payload["items"]
        if (
            not isinstance(succeeded, int) or isinstance(succeeded, bool)
            or not isinstance(failed, int) or isinstance(failed, bool)
            or not isinstance(outcomes, list) or not isinstance(items, list)
            or len(outcomes) != 1
        ):
            raise ValueError("summary")
        for field in ("page_loads", "scroll_rounds", "seen_items"):
            value = payload[field]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("envelope_count")
        outcome = outcomes[0]
        expected_source = f"list:{request.list_id}"
        if (
            not isinstance(outcome, dict) or set(outcome) != _OUTCOME_KEYS
            or outcome.get("source_id") != expected_source
            or outcome.get("source_kind") != "list"
            or not isinstance(outcome.get("ok"), bool)
        ):
            raise ValueError("outcome")
        for field in ("item_count", "page_loads", "scroll_rounds", "observed_ids"):
            value = outcome.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("outcome_count")
        for field in ("error", "html_artifact_path", "selector_report_path"):
            if outcome.get(field) is not None and not isinstance(outcome.get(field), str):
                raise ValueError("outcome_detail")
        documented_totals = {
            "page_loads": "page_loads",
            "scroll_rounds": "scroll_rounds",
            "seen_items": "observed_ids",
        }
        if any(payload[total] != outcome[component] for total, component in documented_totals.items()):
            raise ValueError("envelope_total")
        is_success = outcome["ok"] is True
        if (succeeded, failed) != ((1, 0) if is_success else (0, 1)):
            raise ValueError("summary_coherence")
        if not is_success:
            return []
        if outcome.get("item_count") != len(items):
            raise ValueError("item_count")
        posts = []
        for raw in items[:limit]:
            if not isinstance(raw, dict) or raw.get("source_id", expected_source) != expected_source:
                raise ValueError("item")
            if (
                "is_repost" in raw
                and raw["is_repost"] is not None
                and not isinstance(raw["is_repost"], bool)
            ):
                raise ValueError("is_repost")
            url = raw.get("tweet_url")
            normalized = {
                "tweet_id": raw.get("tweet_id"), "text": raw.get("text"),
                "created_at": raw.get("created_at"), "url": url,
                "canonical_url": url, "author_handle": raw.get("author_handle"),
                "likes": raw.get("likes"), "reposts": raw.get("retweets"),
                "replies": raw.get("replies"), "bookmarks": raw.get("bookmarks"),
                "views": raw.get("views"), "is_repost": raw.get("is_repost", False),
                "attachments": _attachment_refs(raw),
            }
            post = SocialPostRecord.from_untrusted(
                normalized, provider="xui", source_id=request.source_id,
                observed_at=request.observed_at,
            )
            if post.created_at >= request.target_published_after:
                posts.append(post)
        return posts

    @staticmethod
    def _failure_code(payload):
        outcomes = payload.get("outcomes") if isinstance(payload, dict) else None
        error = outcomes[0].get("error") if isinstance(outcomes, list) and outcomes and isinstance(outcomes[0], dict) else ""
        lowered = str(error or "").lower()
        normalized = re.sub(r"[_-]+", " ", lowered)
        if any(
            re.search(rf"\b{re.escape(term.replace('_', ' '))}\b", normalized)
            for term in _AUTH_REASONS
        ):
            return "reauthentication_required"
        if any(term in lowered for term in _NETWORK_REASONS):
            return "provider_network_error"
        return "provider_error"

    def _failed(self, request, code, *, cooldown=False, reset_at=None):
        if cooldown:
            self._cooldown_until = request.observed_at + self._cooldown
            self._cooldown_code = code
            reset_at = self._cooldown_until
        return SocialSourceBatch(request, (), SocialSourceOutcome(
            read_status="failed", processing_status="failed", history_status="limited",
            coverage_reason_codes=(code,), known_gap_intervals=(),
            observed_oldest_at=None, observed_newest_at=None, received_count=0,
            committed_progress=None, error_code=code, proposed_progress=None,
            rate_limit_reset_at=reset_at,
        ))


def _attachment_refs(raw: dict) -> list[dict[str, str]]:
    """Map xui's post-owned evidence fields without treating video previews as photos."""

    refs: list[dict[str, str]] = []
    for value in _string_list(raw.get("image_urls")):
        parsed = urlparse(value)
        host = (parsed.hostname or "").lower()
        if host == "pbs.twimg.com" and parsed.path.startswith("/media/"):
            refs.append({"kind": "image", "url": value})
    for value in _string_list(raw.get("article_urls")):
        parsed = urlparse(value)
        host = (parsed.hostname or "").lower()
        x_hosts = {"x.com", "www.x.com", "twitter.com", "www.twitter.com", "mobile.twitter.com"}
        is_x_article = host in x_hosts and "/article/" in parsed.path
        is_external_article = host not in x_hosts | {"t.co"} and not host.endswith(".twimg.com")
        if parsed.scheme in {"http", "https"} and host and (is_x_article or is_external_article):
            refs.append({"kind": "article", "url": value})
    return [
        {"kind": kind, "url": url}
        for kind, url in dict.fromkeys((ref["kind"], ref["url"]) for ref in refs)
    ]


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("attachment_list")
    return [item.strip() for item in value if item.strip()]
