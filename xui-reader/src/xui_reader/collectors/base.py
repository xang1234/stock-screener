"""Collection interfaces for timeline/list readers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from xui_reader.models import Checkpoint, SourceRef, TweetItem


@dataclass(frozen=True)
class CollectionStats:
    source_id: str
    dom_snapshots: int
    observed_ids: int
    scroll_rounds: int
    stagnation_rounds: int
    stop_reason: str


@dataclass(frozen=True)
class CollectionBatch:
    items: tuple[TweetItem, ...]
    checkpoint: Checkpoint | None = None
    dom_snapshots: tuple[str, ...] = ()
    stats: CollectionStats | None = None


class Collector(Protocol):
    def collect(self, source: SourceRef, limit: int | None = None) -> CollectionBatch:
        """Collect items for a source and return a checkpoint candidate."""
