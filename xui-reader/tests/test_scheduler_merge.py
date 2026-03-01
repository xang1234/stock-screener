"""Deterministic merge behavior for multi-source output."""

from __future__ import annotations

from datetime import datetime

from xui_reader.models import TweetItem
from xui_reader.scheduler.merge import merge_tweet_items


def test_merge_tweet_items_is_deterministic_with_null_timestamp_ties() -> None:
    batch_a = (
        _tweet("100", "list:1", "2026-02-01T00:00:00+00:00"),
        _tweet("099", "list:1", None),
    )
    batch_b = (
        _tweet("101", "user:1", "2026-02-01T00:00:00+00:00"),
        _tweet("110", "list:2", "2026-02-02T00:00:00+00:00"),
        _tweet("120", "user:1", None),
    )

    first = merge_tweet_items((batch_a, batch_b))
    second = merge_tweet_items((batch_b, batch_a))

    assert [item.tweet_id for item in first] == ["110", "101", "100", "120", "099"]
    assert [item.tweet_id for item in second] == ["110", "101", "100", "120", "099"]


def test_merge_tweet_items_uses_source_id_as_final_tie_breaker() -> None:
    a = _tweet("300", "list:aaa", None)
    b = _tweet("300", "list:bbb", None)

    merged = merge_tweet_items(((a, b),))
    assert [item.source_id for item in merged] == ["list:bbb", "list:aaa"]


def _tweet(tweet_id: str, source_id: str, created_at: str | None) -> TweetItem:
    return TweetItem(
        tweet_id=tweet_id,
        created_at=datetime.fromisoformat(created_at) if created_at else None,
        author_handle="@a",
        text=tweet_id,
        source_id=source_id,
        is_reply=False,
        is_repost=False,
        is_pinned=False,
        has_quote=False,
        quote_tweet_id=None,
    )
