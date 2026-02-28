"""Checkpoint mode transition behavior."""

from __future__ import annotations

from datetime import datetime, timezone

from xui_reader.models import Checkpoint, TweetItem
from xui_reader.store.checkpoints import apply_checkpoint_mode


def test_checkpoint_id_mode_advances_from_highest_new_id() -> None:
    checkpoint = Checkpoint(
        source_id="src:1",
        last_seen_id="100",
        last_seen_time=datetime(2026, 2, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 2, 1, 0, 1, tzinfo=timezone.utc),
    )
    items = (
        _tweet("102", "2026-02-03T00:00:00+00:00"),
        _tweet("101", "2026-02-02T00:00:00+00:00"),
        _tweet("100", "2026-02-01T00:00:00+00:00"),
    )

    transition = apply_checkpoint_mode(
        "src:1",
        items,
        checkpoint=checkpoint,
        mode="id",
        now=datetime(2026, 2, 3, 0, 5, tzinfo=timezone.utc),
    )

    assert transition.mode == "id"
    assert [item.tweet_id for item in transition.new_items] == ["102", "101"]
    assert transition.next_checkpoint.last_seen_id == "102"
    assert transition.next_checkpoint.last_seen_time == datetime(2026, 2, 3, 0, 0, tzinfo=timezone.utc)


def test_checkpoint_time_mode_filters_by_timestamp_and_advances_time() -> None:
    checkpoint = Checkpoint(
        source_id="src:1",
        last_seen_id=None,
        last_seen_time=datetime(2026, 2, 2, tzinfo=timezone.utc),
        updated_at=datetime(2026, 2, 2, 0, 1, tzinfo=timezone.utc),
    )
    items = (
        _tweet("12", "2026-02-03T00:00:00+00:00"),
        _tweet("11", "2026-02-02T12:00:00+00:00"),
        _tweet("10", "2026-02-01T00:00:00+00:00"),
    )

    transition = apply_checkpoint_mode(
        "src:1",
        items,
        checkpoint=checkpoint,
        mode="time",
        now=datetime(2026, 2, 3, 0, 5, tzinfo=timezone.utc),
    )

    assert transition.mode == "time"
    assert [item.tweet_id for item in transition.new_items] == ["12", "11"]
    assert transition.next_checkpoint.last_seen_time == datetime(2026, 2, 3, 0, 0, tzinfo=timezone.utc)


def test_checkpoint_no_new_items_keeps_boundary() -> None:
    checkpoint = Checkpoint(
        source_id="src:1",
        last_seen_id="300",
        last_seen_time=datetime(2026, 2, 4, tzinfo=timezone.utc),
        updated_at=datetime(2026, 2, 4, 0, 1, tzinfo=timezone.utc),
    )
    items = (
        _tweet("300", "2026-02-04T00:00:00+00:00"),
        _tweet("299", "2026-02-03T00:00:00+00:00"),
    )

    transition = apply_checkpoint_mode(
        "src:1",
        items,
        checkpoint=checkpoint,
        mode="id",
        now=datetime(2026, 2, 5, 0, 0, tzinfo=timezone.utc),
    )

    assert transition.new_items == ()
    assert transition.next_checkpoint.last_seen_id == "300"
    assert transition.next_checkpoint.last_seen_time == datetime(2026, 2, 4, tzinfo=timezone.utc)
    assert transition.next_checkpoint.updated_at == datetime(2026, 2, 5, 0, 0, tzinfo=timezone.utc)


def test_checkpoint_mixed_order_id_mode_remains_deterministic() -> None:
    checkpoint = Checkpoint(
        source_id="src:1",
        last_seen_id="100",
        last_seen_time=datetime(2026, 2, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 2, 1, 0, 1, tzinfo=timezone.utc),
    )
    # Mixed order in input should still advance to max id among new items.
    items = (
        _tweet("102", "2026-02-03T00:00:00+00:00"),
        _tweet("100", "2026-02-01T00:00:00+00:00"),
        _tweet("101", "2026-02-02T00:00:00+00:00"),
    )

    transition = apply_checkpoint_mode(
        "src:1",
        items,
        checkpoint=checkpoint,
        mode="id",
        now=datetime(2026, 2, 3, 0, 5, tzinfo=timezone.utc),
    )

    assert [item.tweet_id for item in transition.new_items] == ["102", "101"]
    assert transition.next_checkpoint.last_seen_id == "102"
    assert transition.next_checkpoint.last_seen_time == datetime(2026, 2, 3, 0, 0, tzinfo=timezone.utc)


def test_checkpoint_time_mode_clears_id_and_keeps_auto_in_time_mode() -> None:
    checkpoint = Checkpoint(
        source_id="src:1",
        last_seen_id="100",
        last_seen_time=datetime(2026, 2, 2, tzinfo=timezone.utc),
        updated_at=datetime(2026, 2, 2, 0, 1, tzinfo=timezone.utc),
    )
    items = (
        _tweet("102", "2026-02-03T00:00:00+00:00"),
        _tweet("101", "2026-02-02T12:00:00+00:00"),
        _tweet("100", "2026-02-01T00:00:00+00:00"),
    )

    first = apply_checkpoint_mode(
        "src:1",
        items,
        checkpoint=checkpoint,
        mode="time",
        now=datetime(2026, 2, 3, 0, 5, tzinfo=timezone.utc),
    )

    assert first.mode == "time"
    assert first.next_checkpoint.last_seen_id is None

    followup = apply_checkpoint_mode(
        "src:1",
        (),
        checkpoint=first.next_checkpoint,
        mode="auto",
        now=datetime(2026, 2, 3, 0, 6, tzinfo=timezone.utc),
    )

    assert followup.mode == "time"


def _tweet(tweet_id: str, created_at: str) -> TweetItem:
    return TweetItem(
        tweet_id=tweet_id,
        created_at=datetime.fromisoformat(created_at),
        author_handle="@a",
        text=tweet_id,
        source_id="src:1",
    )
