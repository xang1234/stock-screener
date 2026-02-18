"""Ports (abstract interfaces) for the scanning domain.

These define WHAT the domain needs from the outside world without
specifying HOW it's provided.  Concrete implementations live in infra/.

Note: No infrastructure types (Session, Engine, Redis) appear here.
Repositories receive their session/connection through the UnitOfWork,
not through method parameters.
"""

from __future__ import annotations

import abc
from typing import Protocol

from .filter_spec import FilterSpec, QuerySpec, SortSpec
from .models import FilterOptions, ProgressEvent, ResultPage, ScanResultItemDomain


# ---------------------------------------------------------------------------
# Repositories
# ---------------------------------------------------------------------------


class ScanRepository(abc.ABC):
    """Persist and retrieve scan metadata."""

    @abc.abstractmethod
    def create(self, *, scan_id: str, **fields) -> object:
        ...

    @abc.abstractmethod
    def get_by_scan_id(self, scan_id: str) -> object | None:
        ...

    @abc.abstractmethod
    def get_by_idempotency_key(self, key: str) -> object | None:
        """Return an existing scan matching the idempotency key, or None."""
        ...

    @abc.abstractmethod
    def update_status(self, scan_id: str, status: str, **fields) -> None:
        """Update scan status and optional fields (total_stocks, passed_stocks, etc.)."""
        ...

    @abc.abstractmethod
    def list_recent(self, limit: int = 20) -> list[object]:
        """Return the most recent scans, ordered by started_at descending."""
        ...

    @abc.abstractmethod
    def delete(self, scan_id: str) -> bool:
        """Delete a scan by scan_id. Returns True if deleted, False if not found."""
        ...


class ScanResultRepository(abc.ABC):
    """Persist and retrieve individual scan results."""

    @abc.abstractmethod
    def bulk_insert(self, rows: list[dict]) -> int:
        ...

    @abc.abstractmethod
    def persist_orchestrator_results(
        self, scan_id: str, results: list[tuple[str, dict]]
    ) -> int:
        """Persist raw orchestrator output, handling field mapping internally.

        Args:
            scan_id: Scan identifier.
            results: ``[(symbol, result_dict), ...]`` pairs from orchestrator.

        Returns:
            Number of results persisted.
        """
        ...

    @abc.abstractmethod
    def count_by_scan_id(self, scan_id: str) -> int:
        """Return the number of results already stored for *scan_id*."""
        ...

    @abc.abstractmethod
    def query(
        self,
        scan_id: str,
        spec: QuerySpec,
        *,
        include_sparklines: bool = True,
    ) -> ResultPage:
        """Return a paginated, filtered, sorted page of scan results.

        Args:
            scan_id: Scan identifier (must exist — caller validates).
            spec: Domain-level query specification (filters, sort, pagination).
            include_sparklines: Whether to populate sparkline arrays in results.

        Returns:
            A :class:`ResultPage` with the matching items and total count.
        """
        ...

    @abc.abstractmethod
    def query_all(
        self,
        scan_id: str,
        filters: FilterSpec,
        sort: SortSpec,
        *,
        include_sparklines: bool = False,
    ) -> tuple[ScanResultItemDomain, ...]:
        """Return ALL filtered, sorted results (no pagination) for export."""
        ...

    @abc.abstractmethod
    def delete_by_scan_id(self, scan_id: str) -> int:
        """Delete all results for a scan. Returns the number of rows deleted."""
        ...

    @abc.abstractmethod
    def get_filter_options(self, scan_id: str) -> FilterOptions:
        """Return distinct categorical values for filtering a scan's results.

        Args:
            scan_id: Scan identifier (must exist — caller validates).

        Returns:
            A :class:`FilterOptions` with sorted, de-duplicated values
            for each categorical column.
        """
        ...

    @abc.abstractmethod
    def get_by_symbol(
        self, scan_id: str, symbol: str
    ) -> ScanResultItemDomain | None:
        """Return a single result by scan_id + symbol, or None.

        Args:
            scan_id: Scan identifier (caller validates existence).
            symbol: Stock symbol (expected to be already normalised to
                uppercase).
        """
        ...

    @abc.abstractmethod
    def get_peers_by_industry(
        self, scan_id: str, ibd_industry_group: str
    ) -> tuple[ScanResultItemDomain, ...]:
        """Return all results in the same IBD industry group, ordered by composite_score DESC."""
        ...

    @abc.abstractmethod
    def get_peers_by_sector(
        self, scan_id: str, gics_sector: str
    ) -> tuple[ScanResultItemDomain, ...]:
        """Return all results in the same GICS sector, ordered by composite_score DESC."""
        ...


class UniverseRepository(abc.ABC):
    """Resolve which symbols belong to a universe."""

    @abc.abstractmethod
    def resolve_symbols(self, universe_def: object) -> list[str]:
        ...


# ---------------------------------------------------------------------------
# Infrastructure services
# ---------------------------------------------------------------------------


class TaskDispatcher(abc.ABC):
    """Dispatch background tasks (e.g. Celery) for scan execution."""

    @abc.abstractmethod
    def dispatch_scan(
        self, scan_id: str, symbols: list[str], criteria: dict
    ) -> str:
        """Queue a bulk-scan task. Returns the task ID."""
        ...


class StockDataProvider(abc.ABC):
    """Fetch market/fundamental data for scoring."""

    @abc.abstractmethod
    def prepare_data(
        self, symbol: str, requirements: object, *, allow_partial: bool = True
    ) -> object:
        ...

    @abc.abstractmethod
    def prepare_data_bulk(
        self, symbols: list[str], requirements: object, *, allow_partial: bool = True
    ) -> dict[str, object]:
        ...


# ---------------------------------------------------------------------------
# Workflow collaborators
# ---------------------------------------------------------------------------


class ProgressSink(abc.ABC):
    """Report scan progress to the outside world (Celery, WebSocket, log, …)."""

    @abc.abstractmethod
    def emit(self, event: ProgressEvent) -> None:
        ...


class CancellationToken(abc.ABC):
    """Check whether the current operation has been requested to stop."""

    @abc.abstractmethod
    def is_cancelled(self) -> bool:
        ...


class NullProgressSink(ProgressSink):
    """No-op sink for testing and non-Celery contexts."""

    def emit(self, event: ProgressEvent) -> None:
        pass


class NeverCancelledToken(CancellationToken):
    """Token that never cancels — for CLI scripts and tests."""

    def is_cancelled(self) -> bool:
        return False


class StockScanner(Protocol):
    """Structural type satisfied by :class:`ScanOrchestrator`.

    Using a Protocol lets the use case depend on the *shape* of the
    orchestrator without importing its concrete class.
    """

    def scan_stock_multi(
        self,
        symbol: str,
        screener_names: list[str],
        criteria: dict | None = ...,
        composite_method: str = ...,
    ) -> dict:
        ...
