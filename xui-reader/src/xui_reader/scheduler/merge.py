"""Deterministic merge helpers for multi-source read output."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime, timezone

from xui_reader.models import TweetItem


def merge_tweet_items(batches: Iterable[Sequence[TweetItem]]) -> tuple[TweetItem, ...]:
    """Merge source batches with deterministic ordering."""
    merged: list[TweetItem] = []
    for batch in batches:
        merged.extend(batch)

    return tuple(
        sorted(
            merged,
            key=lambda item: (
                _normalize_datetime(item.created_at) or datetime.min.replace(tzinfo=timezone.utc),
                _tweet_id_sort_key(item.tweet_id),
                item.source_id,
            ),
            reverse=True,
        )
    )


def _normalize_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _tweet_id_sort_key(tweet_id: str) -> tuple[int, int | str]:
    if tweet_id.isdigit():
        return (1, int(tweet_id))
    return (0, tweet_id)
