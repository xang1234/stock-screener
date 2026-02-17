"""Ports (abstract interfaces) for the feature store domain.

These define WHAT the domain needs from the outside world without
specifying HOW it's provided.  Concrete implementations live in
``app.infra.db.repositories``.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from datetime import date

from app.domain.common.query import FilterSpec, PageSpec, SortSpec
from app.domain.feature_store.quality import DQResult

from .models import FeaturePage, FeatureRowWrite, FeatureRunDomain, RunStats, RunType


# ---------------------------------------------------------------------------
# Feature Run Repository
# ---------------------------------------------------------------------------


class FeatureRunRepository(abc.ABC):
    """Persist and retrieve feature run lifecycle records."""

    @abc.abstractmethod
    def start_run(
        self,
        as_of_date: date,
        run_type: RunType,
        code_version: str | None = None,
        universe_hash: str | None = None,
        input_hash: str | None = None,
        correlation_id: str | None = None,
    ) -> FeatureRunDomain:
        """Create a new run in RUNNING status and return its domain object."""
        ...

    @abc.abstractmethod
    def mark_completed(
        self,
        run_id: int,
        stats: RunStats,
        warnings: Sequence[str] = (),
    ) -> FeatureRunDomain:
        """Transition RUNNING → COMPLETED, storing stats and warnings.

        Raises:
            EntityNotFoundError: If *run_id* does not exist.
            InvalidTransitionError: If current status is not RUNNING.
        """
        ...

    @abc.abstractmethod
    def mark_quarantined(
        self,
        run_id: int,
        dq_results: Sequence[DQResult],
    ) -> FeatureRunDomain:
        """Transition COMPLETED → QUARANTINED, storing DQ results.

        Raises:
            EntityNotFoundError: If *run_id* does not exist.
            InvalidTransitionError: If current status is not COMPLETED.
        """
        ...

    @abc.abstractmethod
    def publish_atomically(self, run_id: int) -> FeatureRunDomain:
        """Transition COMPLETED → PUBLISHED and update the pointer.

        Both the status change and the pointer swap happen in the same
        flush, so the UoW's ``commit()`` makes them visible atomically.

        Raises:
            EntityNotFoundError: If *run_id* does not exist.
            InvalidTransitionError: If current status is not COMPLETED.
        """
        ...

    @abc.abstractmethod
    def get_latest_published(self) -> FeatureRunDomain | None:
        """Return the run pointed to by 'latest_published', or None."""
        ...

    @abc.abstractmethod
    def get_run(self, run_id: int) -> FeatureRunDomain:
        """Return a run by PK.

        Raises:
            EntityNotFoundError: If *run_id* does not exist.
        """
        ...


# ---------------------------------------------------------------------------
# Feature Store Repository
# ---------------------------------------------------------------------------


class FeatureStoreRepository(abc.ABC):
    """Persist and query per-symbol feature rows."""

    @abc.abstractmethod
    def upsert_snapshot_rows(
        self,
        run_id: int,
        rows: Sequence[FeatureRowWrite],
    ) -> int:
        """Insert or update snapshot rows for a run.  Returns row count.

        Uses true UPSERT (INSERT OR REPLACE) so Celery retries are safe.
        Batched internally to avoid SQLite variable limit errors.
        """
        ...

    @abc.abstractmethod
    def save_run_universe_symbols(
        self,
        run_id: int,
        symbols: Sequence[str],
    ) -> None:
        """Bulk-insert universe symbols for a run."""
        ...

    @abc.abstractmethod
    def count_by_run_id(self, run_id: int) -> int:
        """Count feature rows for a run (supports DQ check_row_count)."""
        ...

    @abc.abstractmethod
    def query_latest(
        self,
        filters: FilterSpec | None = None,
        sort: SortSpec | None = None,
        page: PageSpec | None = None,
    ) -> FeaturePage:
        """Query the latest published snapshot via pointer lookup.

        Returns an empty FeaturePage if no pointer exists.
        """
        ...

    @abc.abstractmethod
    def query_run(
        self,
        run_id: int,
        filters: FilterSpec | None = None,
        sort: SortSpec | None = None,
        page: PageSpec | None = None,
    ) -> FeaturePage:
        """Query feature rows for a specific run.

        Raises:
            EntityNotFoundError: If *run_id* does not exist.
        """
        ...


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "FeatureRunRepository",
    "FeatureStoreRepository",
]
