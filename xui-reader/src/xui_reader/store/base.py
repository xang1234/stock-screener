"""Storage interfaces for items and checkpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from xui_reader.models import Checkpoint, SourceRef, TweetItem


class Store(Protocol):
    def upsert_source(self, source: SourceRef) -> None:
        """Persist or update source metadata."""

    def save_items(self, source_id: str, items: tuple[TweetItem, ...]) -> int:
        """Persist items and return count saved."""

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Persist a checkpoint for a source."""

    def load_checkpoint(self, source_id: str) -> Checkpoint | None:
        """Load a source checkpoint if present."""

    def load_new_since(self, since: datetime) -> tuple[TweetItem, ...]:
        """Load items newer than the given timestamp."""

    def begin_run(self, source_id: str, started_at: datetime | None = None) -> int:
        """Create a run record and return its run id."""

    def finish_run(
        self,
        run_id: int,
        *,
        status: str,
        observed_count: int = 0,
        saved_count: int = 0,
        error: str | None = None,
        finished_at: datetime | None = None,
    ) -> None:
        """Complete a run record with terminal status and counters."""
