"""Ports (abstract interfaces) for the feature store domain.

These define WHAT the domain needs from the outside world without
specifying HOW it's provided.  Concrete implementations live in
``app.infra.db.repositories``.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from datetime import date

from app.domain.common.query import FilterSpec, PageSpec, QuerySpec, SortSpec
from app.domain.feature_store.quality import DQInputs, DQResult
from app.domain.scanning.models import FilterOptions, ResultPage, ScanResultItemDomain

from .models import (
    FeaturePage,
    FeatureRow,
    FeatureRowWrite,
    FeatureRunDomain,
    RunStats,
    RunStatus,
    RunType,
)


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
        """Transition COMPLETED|QUARANTINED → PUBLISHED and update the pointer.

        Both the status change and the pointer swap happen in the same
        flush, so the UoW's ``commit()`` makes them visible atomically.

        Raises:
            EntityNotFoundError: If *run_id* does not exist.
            InvalidTransitionError: If current status is not COMPLETED
                or QUARANTINED.
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

    @abc.abstractmethod
    def list_runs_with_counts(
        self,
        *,
        status: RunStatus | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        limit: int = 50,
    ) -> Sequence[tuple[FeatureRunDomain, int, bool]]:
        """Return (run, row_count, is_latest_published) tuples.

        Ordered by created_at DESC.  Uses a COUNT subquery and pointer
        JOIN — single SQL query, no N+1.
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

    @abc.abstractmethod
    def get_run_dq_inputs(self, run_id: int) -> DQInputs:
        """Build DQInputs from persisted feature rows and universe symbols.

        Used by the standalone publish path when DQ inputs are not
        pre-computed from in-memory rows.
        """
        ...

    @abc.abstractmethod
    def get_row_by_symbol(self, run_id: int, symbol: str) -> FeatureRow | None:
        """Get a single feature row by run_id and symbol (case-insensitive).

        Does NOT validate run_id exists (caller handles EntityNotFoundError).
        """
        ...

    @abc.abstractmethod
    def get_scores_for_run(
        self, run_id: int
    ) -> dict[str, tuple[float | None, int | None]]:
        """Return {symbol: (composite_score, overall_rating)} for all symbols in a run.

        Raises EntityNotFoundError if run_id doesn't exist.
        """
        ...

    # -- Bridge methods (scanning-domain reads via feature store) -----------

    @abc.abstractmethod
    def query_run_as_scan_results(
        self,
        run_id: int,
        spec: QuerySpec,
        include_sparklines: bool = True,
        include_setup_payload: bool = True,
    ) -> ResultPage:
        """Paginated query of a feature run mapped to scanning-domain models.

        Raises:
            EntityNotFoundError: If *run_id* does not exist.
        """
        ...

    @abc.abstractmethod
    def get_by_symbol_for_run(
        self,
        run_id: int,
        symbol: str,
        include_sparklines: bool = True,
        include_setup_payload: bool = True,
    ) -> ScanResultItemDomain | None:
        """Single symbol lookup from a feature run.

        Returns None if the symbol is not in the run.

        Raises:
            EntityNotFoundError: If *run_id* does not exist.
        """
        ...

    @abc.abstractmethod
    def get_peers_by_industry_for_run(
        self,
        run_id: int,
        ibd_industry_group: str,
    ) -> tuple[ScanResultItemDomain, ...]:
        """Return peers sharing the same IBD industry group.

        Raises:
            EntityNotFoundError: If *run_id* does not exist.
        """
        ...

    @abc.abstractmethod
    def get_peers_by_sector_for_run(
        self,
        run_id: int,
        gics_sector: str,
    ) -> tuple[ScanResultItemDomain, ...]:
        """Return peers sharing the same GICS sector.

        Raises:
            EntityNotFoundError: If *run_id* does not exist.
        """
        ...

    @abc.abstractmethod
    def get_filter_options_for_run(
        self,
        run_id: int,
    ) -> FilterOptions:
        """Return distinct filter option values for a feature run.

        Raises:
            EntityNotFoundError: If *run_id* does not exist.
        """
        ...

    @abc.abstractmethod
    def query_all_as_scan_results(
        self,
        run_id: int,
        filters: FilterSpec | None = None,
        sort: SortSpec | None = None,
        include_sparklines: bool = False,
    ) -> tuple[ScanResultItemDomain, ...]:
        """Return all rows from a feature run (no pagination; for export).

        Raises:
            EntityNotFoundError: If *run_id* does not exist.
        """
        ...

    @abc.abstractmethod
    def query_run_symbols(
        self,
        run_id: int,
        filters: FilterSpec,
        sort: SortSpec,
        page: PageSpec | None = None,
    ) -> tuple[tuple[str, ...], int]:
        """Return filtered, sorted symbols and total count for a feature run."""
        ...

    @abc.abstractmethod
    def get_setup_payload_for_run(
        self,
        run_id: int,
        symbol: str,
    ) -> dict | None:
        """Return setup-engine explain payload for a symbol in a feature run."""
        ...


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "FeatureRunRepository",
    "FeatureStoreRepository",
]
