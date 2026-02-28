"""Checkpoint transition helpers for new-only filtering semantics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from xui_reader.models import Checkpoint, TweetItem

CheckpointMode = Literal["id", "time"]


@dataclass(frozen=True)
class CheckpointTransition:
    mode: CheckpointMode
    new_items: tuple[TweetItem, ...]
    next_checkpoint: Checkpoint


def apply_checkpoint_mode(
    source_id: str,
    items: tuple[TweetItem, ...],
    *,
    checkpoint: Checkpoint | None = None,
    mode: CheckpointMode | Literal["auto"] = "auto",
    now: datetime | None = None,
) -> CheckpointTransition:
    """Filter new items and compute deterministic checkpoint advancement."""
    selected_mode = _resolve_mode(checkpoint, mode)
    if selected_mode == "id":
        return _apply_id_mode(source_id, items, checkpoint=checkpoint, now=now)
    return _apply_time_mode(source_id, items, checkpoint=checkpoint, now=now)


def _apply_id_mode(
    source_id: str,
    items: tuple[TweetItem, ...],
    *,
    checkpoint: Checkpoint | None,
    now: datetime | None,
) -> CheckpointTransition:
    seen = set()
    baseline_id = checkpoint.last_seen_id if checkpoint is not None else None
    baseline_int = _tweet_id_to_int(baseline_id)
    new_items: list[TweetItem] = []

    for item in items:
        if item.tweet_id in seen:
            continue
        seen.add(item.tweet_id)
        item_int = _tweet_id_to_int(item.tweet_id)

        if baseline_id is None:
            new_items.append(item)
            continue
        if baseline_int is not None and item_int is not None:
            if item_int > baseline_int:
                new_items.append(item)
            continue
        if item.tweet_id != baseline_id:
            new_items.append(item)

    if new_items:
        next_seen_id = _max_tweet_id(new_items)
        next_seen_time = _max_created_at(new_items) or (checkpoint.last_seen_time if checkpoint else None)
    else:
        next_seen_id = baseline_id
        next_seen_time = checkpoint.last_seen_time if checkpoint else None

    return CheckpointTransition(
        mode="id",
        new_items=tuple(new_items),
        next_checkpoint=Checkpoint(
            source_id=source_id,
            last_seen_id=next_seen_id,
            last_seen_time=next_seen_time,
            updated_at=_normalize_now(now),
        ),
    )


def _apply_time_mode(
    source_id: str,
    items: tuple[TweetItem, ...],
    *,
    checkpoint: Checkpoint | None,
    now: datetime | None,
) -> CheckpointTransition:
    seen = set()
    baseline_time = checkpoint.last_seen_time if checkpoint is not None else None
    baseline_time = _normalize_dt(baseline_time)
    new_items: list[TweetItem] = []

    for item in items:
        if item.tweet_id in seen:
            continue
        seen.add(item.tweet_id)
        created = _normalize_dt(item.created_at)
        if baseline_time is None:
            new_items.append(item)
            continue
        if created is not None and created > baseline_time:
            new_items.append(item)

    if new_items:
        next_seen_time = _max_created_at(new_items) or baseline_time
    else:
        next_seen_time = baseline_time

    return CheckpointTransition(
        mode="time",
        new_items=tuple(new_items),
        next_checkpoint=Checkpoint(
            source_id=source_id,
            # Keep time-mode transitions sticky under auto resolution.
            last_seen_id=None,
            last_seen_time=next_seen_time,
            updated_at=_normalize_now(now),
        ),
    )


def _resolve_mode(
    checkpoint: Checkpoint | None,
    mode: CheckpointMode | Literal["auto"],
) -> CheckpointMode:
    if mode != "auto":
        return mode
    if checkpoint is not None and checkpoint.last_seen_id:
        return "id"
    return "time"


def _normalize_now(now: datetime | None) -> datetime:
    return _normalize_dt(now) or datetime.now(timezone.utc)


def _normalize_dt(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _tweet_id_to_int(tweet_id: str | None) -> int | None:
    if tweet_id is None:
        return None
    if not tweet_id.isdigit():
        return None
    return int(tweet_id)


def _max_tweet_id(items: list[TweetItem]) -> str:
    numeric: list[tuple[int, str]] = []
    for item in items:
        parsed = _tweet_id_to_int(item.tweet_id)
        if parsed is None:
            continue
        numeric.append((parsed, item.tweet_id))
    if numeric:
        numeric.sort(key=lambda row: row[0], reverse=True)
        return numeric[0][1]
    return items[0].tweet_id


def _max_created_at(items: list[TweetItem]) -> datetime | None:
    values = [_normalize_dt(item.created_at) for item in items]
    present = [value for value in values if value is not None]
    if not present:
        return None
    return max(present)
