"""Storage interfaces for items and checkpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from xui_reader.models import Checkpoint, TweetItem


class Store(Protocol):
    def save_items(self, source_id: str, items: tuple[TweetItem, ...]) -> int:
        """Persist items and return count saved."""

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Persist a checkpoint for a source."""

    def load_checkpoint(self, source_id: str) -> Checkpoint | None:
        """Load a source checkpoint if present."""

    def load_new_since(self, since: datetime) -> tuple[TweetItem, ...]:
        """Load items newer than the given timestamp."""
