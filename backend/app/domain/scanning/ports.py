"""Ports (abstract interfaces) for the scanning domain.

These define WHAT the domain needs from the outside world without
specifying HOW it's provided.  Concrete implementations live in infra/.

Note: No infrastructure types (Session, Engine, Redis) appear here.
Repositories receive their session/connection through the UnitOfWork,
not through method parameters.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import date
from typing import Mapping, Protocol, Sequence

from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
)

from app.domain.common.query import PageSpec, SortSpec

from .filter_expression_model import FilterExpression, QuerySpec
from .models import FilterOptions, ProgressEvent, ResultPage, ScanResultItemDomain


# ---------------------------------------------------------------------------
# Repositories
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanResultRsAudit:
    """Market RS publication identity persisted with one scan-result row."""

    symbol: str
    formula_version: str | None
    run_id: int | None

    @classmethod
    def from_payload(
        cls,
        symbol: str,
        payload: Mapping[str, object] | None,
    ) -> "ScanResultRsAudit":
        details = payload or {}
        formula_version = str(details.get("rs_formula_version") or "").strip()
        raw_run_id = details.get("market_rs_run_id")
        run_id = int(raw_run_id) if raw_run_id is not None else None
        return cls(
            symbol=str(symbol).strip().upper(),
            formula_version=formula_version or None,
            run_id=run_id,
        )


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
    def get_active_scan(self) -> object | None:
        """Return the most recent queued/running scan, or None when idle."""
        ...

    @abc.abstractmethod
    def update_status(self, scan_id: str, status: str, **fields) -> None:
        """Update scan status and optional fields (total_stocks, passed_stocks, etc.)."""
        ...

    @abc.abstractmethod
    def list_recent(self, limit: int = 20, market: str | None = None) -> list[object]:
        """Return the most recent scans, ordered by started_at descending.

        ``market`` restricts results to scans of one universe market.
        """
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
    def list_rs_audits_by_scan_id(
        self,
        scan_id: str,
    ) -> tuple[ScanResultRsAudit, ...]:
        """Return persisted Market RS audit identities for a resumable scan."""
        ...

    @abc.abstractmethod
    def query(
        self,
        scan_id: str,
        spec: QuerySpec,
        *,
        include_sparklines: bool = True,
        include_setup_payload: bool = True,
    ) -> ResultPage:
        """Return a paginated, filtered, sorted page of scan results.

        Args:
            scan_id: Scan identifier (must exist — caller validates).
            spec: Domain-level query specification (expression, sort, pagination).
            include_sparklines: Whether to populate sparkline arrays in results.
            include_setup_payload: Whether to include heavy setup-engine explain
                payload fields (se_explain, se_candidates).

        Returns:
            A :class:`ResultPage` with the matching items and total count.
        """
        ...

    @abc.abstractmethod
    def query_symbols(
        self,
        scan_id: str,
        expression: FilterExpression,
        sort: SortSpec,
        *,
        page: PageSpec | None = None,
    ) -> tuple[tuple[str, ...], int]:
        """Return filtered, sorted symbols and total count.

        Args:
            scan_id: Scan identifier (must exist — caller validates).
            expression: Canonical filter expression.
            sort: Sort specification.
            page: Optional pagination for symbol lists.
        """
        ...

    @abc.abstractmethod
    def query_all(
        self,
        scan_id: str,
        expression: FilterExpression,
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
        self,
        scan_id: str,
        symbol: str,
        *,
        include_setup_payload: bool = True,
    ) -> ScanResultItemDomain | None:
        """Return a single result by scan_id + symbol, or None.

        Args:
            scan_id: Scan identifier (caller validates existence).
            symbol: Stock symbol (expected to be already normalised to
                uppercase).
            include_setup_payload: Whether to include heavy setup-engine explain
                payload fields (se_explain, se_candidates).
        """
        ...

    @abc.abstractmethod
    def get_setup_payload(
        self, scan_id: str, symbol: str
    ) -> dict | None:
        """Return setup-engine explain payload for a symbol, or None if missing."""
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

    @abc.abstractmethod
    def get_details_by_symbol(
        self, scan_id: str, symbol: str
    ) -> dict | None:
        """Return the raw details JSON blob for a single result, or None.

        Used by the explain-stock fallback path to reconstruct screener
        outputs from the legacy ``scan_results`` table.
        """
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
        self,
        scan_id: str,
        symbols: list[str],
        criteria: dict,
        *,
        market: str | None = None,
    ) -> str:
        """Queue a bulk-scan task. Returns the task ID."""
        ...


@dataclass(frozen=True)
class LegacyStockRsSource:
    formula_version: str = LEGACY_RS_FORMULA_VERSION

    def __post_init__(self) -> None:
        if self.formula_version != LEGACY_RS_FORMULA_VERSION:
            raise ValueError("legacy stock RS source requires the legacy formula")

    def audit_fields(self) -> dict[str, object]:
        return {
            "rs_formula_version": self.formula_version,
            "market_rs_run_id": None,
            "rs_universe_size": None,
        }


@dataclass(frozen=True)
class CanonicalStockRsSource:
    formula_version: str
    run_id: int
    universe_size: int
    ratings: Mapping[str, int] | None

    def __post_init__(self) -> None:
        if self.formula_version != BALANCED_RS_FORMULA_VERSION:
            raise ValueError("canonical stock RS source requires the balanced formula")
        if self.run_id <= 0:
            raise ValueError("canonical stock RS run_id must be positive")
        if self.universe_size <= 0:
            raise ValueError("canonical stock RS universe_size must be positive")
        if self.ratings is not None:
            object.__setattr__(self, "ratings", dict(self.ratings))

    def audit_fields(self) -> dict[str, object]:
        return {
            "rs_formula_version": self.formula_version,
            "market_rs_run_id": self.run_id,
            "rs_universe_size": self.universe_size,
        }


StockRsSource = LegacyStockRsSource | CanonicalStockRsSource


@dataclass(frozen=True)
class LegacyMarketRsSource:
    formula_version: str = LEGACY_RS_FORMULA_VERSION

    def __post_init__(self) -> None:
        if self.formula_version != LEGACY_RS_FORMULA_VERSION:
            raise ValueError("legacy Market RS source requires the legacy formula")


@dataclass(frozen=True)
class CanonicalMarketRsSource:
    formula_version: str
    run_id: int
    universe_size: int
    ratings_by_symbol: Mapping[str, Mapping[str, int]]

    def __post_init__(self) -> None:
        if self.formula_version != BALANCED_RS_FORMULA_VERSION:
            raise ValueError("canonical Market RS source requires the balanced formula")
        if self.run_id <= 0:
            raise ValueError("canonical Market RS run_id must be positive")
        if self.universe_size <= 0:
            raise ValueError("canonical Market RS universe_size must be positive")
        object.__setattr__(
            self,
            "ratings_by_symbol",
            {
                str(symbol).strip().upper(): dict(ratings)
                for symbol, ratings in self.ratings_by_symbol.items()
            },
        )


MarketRsSource = LegacyMarketRsSource | CanonicalMarketRsSource


@dataclass(frozen=True)
class MarketRsResolution:
    market: str
    as_of_date: date | None
    source: MarketRsSource

    def __post_init__(self) -> None:
        normalized_market = self.market.strip().upper()
        if not normalized_market:
            raise ValueError("Market RS market is required")
        object.__setattr__(self, "market", normalized_market)

    @classmethod
    def legacy(
        cls,
        *,
        market: str,
        as_of_date: date | None,
        formula_version: str = LEGACY_RS_FORMULA_VERSION,
    ) -> "MarketRsResolution":
        return cls(
            market=market,
            as_of_date=as_of_date,
            source=LegacyMarketRsSource(formula_version=formula_version),
        )

    @classmethod
    def canonical(
        cls,
        *,
        market: str,
        as_of_date: date,
        formula_version: str,
        run_id: int,
        universe_size: int,
        ratings_by_symbol: Mapping[str, Mapping[str, int]],
    ) -> "MarketRsResolution":
        return cls(
            market=market,
            as_of_date=as_of_date,
            source=CanonicalMarketRsSource(
                formula_version=formula_version,
                run_id=run_id,
                universe_size=universe_size,
                ratings_by_symbol=ratings_by_symbol,
            ),
        )

    @property
    def formula_version(self) -> str:
        return self.source.formula_version

    @property
    def run_id(self) -> int | None:
        return self.source.run_id if isinstance(self.source, CanonicalMarketRsSource) else None

    @property
    def universe_size(self) -> int | None:
        return (
            self.source.universe_size
            if isinstance(self.source, CanonicalMarketRsSource)
            else None
        )

    @property
    def ratings_by_symbol(self) -> Mapping[str, Mapping[str, int]]:
        return (
            self.source.ratings_by_symbol
            if isinstance(self.source, CanonicalMarketRsSource)
            else {}
        )

    def stock_source(self, symbol: str) -> StockRsSource:
        if isinstance(self.source, LegacyMarketRsSource):
            return LegacyStockRsSource(formula_version=self.source.formula_version)
        return CanonicalStockRsSource(
            formula_version=self.source.formula_version,
            run_id=self.source.run_id,
            universe_size=self.source.universe_size,
            ratings=self.source.ratings_by_symbol.get(symbol.strip().upper()),
        )


class StockDataProvider(abc.ABC):
    """Fetch market/fundamental data for scoring."""

    @abc.abstractmethod
    def prepare_data(
        self, symbol: str, requirements: object, *, allow_partial: bool = True
    ) -> object:
        ...

    @abc.abstractmethod
    def prepare_data_bulk(
        self,
        symbols: list[str],
        requirements: object,
        *,
        allow_partial: bool = True,
        batch_only_prices: bool = False,
        batch_only_fundamentals: bool = False,
    ) -> dict[str, object]:
        ...

    @abc.abstractmethod
    def apply_market_rs_resolution(
        self,
        results: dict[str, object],
        resolution: MarketRsResolution,
    ) -> None:
        raise NotImplementedError


class MarketRsReader(Protocol):
    def get(
        self,
        *,
        market: str,
        symbols: Sequence[str],
        as_of_date: date | None,
        formula_version: str | None = None,
        run_id: int | None = None,
    ) -> MarketRsResolution:
        raise NotImplementedError


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
        pre_merged_requirements: object | None = ...,
        pre_fetched_data: object | None = ...,
        market_rs_resolution: MarketRsResolution | None = ...,
    ) -> dict:
        ...
