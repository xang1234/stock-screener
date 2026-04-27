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
        config_json: dict | None = None,
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
    def mark_failed(
        self,
        run_id: int,
        stats: RunStats,
        warnings: Sequence[str] = (),
    ) -> FeatureRunDomain:
        """Transition RUNNING → FAILED, storing best-effort stats and warnings.

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
    def publish_atomically(
        self,
        run_id: int,
        pointer_key: str = "latest_published",
    ) -> FeatureRunDomain:
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
    def get_latest_published(
        self,
        pointer_key: str = "latest_published",
    ) -> FeatureRunDomain | None:
        """Return the run pointed to by *pointer_key*, or None."""
        ...

    @abc.abstractmethod
    def find_latest_published_exact(
        self,
        *,
        input_hash: str,
        universe_hash: str,
        as_of_date: date | None = None,
    ) -> FeatureRunDomain | None:
        """Return the newest published run matching the exact signature."""
        ...

    @abc.abstractmethod
    def find_latest_published_covering(
        self,
        *,
        symbols: Sequence[str],
        market: str | None = None,
    ) -> FeatureRunDomain | None:
        """Return the newest published run whose universe covers all *symbols*.

        Used by the feature-store-first custom scan path: when the criteria
        compile cleanly into queryable fields we don't need an exact
        signature match — we just need a published run that already covers
        every symbol the scan would otherwise process.

        If *market* is provided the implementation tries the
        ``latest_published_market:{market}`` pointer first (matching the
        per-market publish convention) and falls back to the global
        ``latest_published`` pointer. Returns ``None`` when no covering run
        exists.
        """
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

        Uses true UPSERT semantics so Celery retries are safe.
        Batched internally to keep statement size bounded.
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

    @abc.abstractmethod
    def query_run_details(
        self,
        run_id: int,
        filters: FilterSpec | None = None,
        *,
        symbols: Sequence[str] | None = None,
    ) -> list[tuple[str, dict]]:
        """Return ``(symbol, details_json)`` pairs from a feature run.

        Used by the feature-store-first custom scan path: the use case
        compiles user criteria into a ``FilterSpec`` and asks for the raw
        per-symbol details so it can persist them as scan results without
        recomputing scores. ``details_json`` matches the orchestrator's
        output shape, so the result is directly usable by
        ``ScanResultRepository.persist_orchestrator_results``.

        Args:
            run_id: Feature run ID (caller validates existence).
            filters: Optional feature-store filter spec to apply (range,
                categorical, boolean filters on indexed columns or JSON
                paths).
            symbols: Optional symbol allow-list to intersect with the run's
                rows. Useful when the scan's universe is narrower than the
                run's published universe (e.g., index/custom scoped scans
                served from an all-market run).
        """
        ...


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "FeatureRunRepository",
    "FeatureStoreRepository",
]
