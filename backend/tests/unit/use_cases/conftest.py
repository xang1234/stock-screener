"""Shared test fakes and fixtures for scanning use case tests.

Consolidates all in-memory fake implementations of domain ports.
Each fake stores real data and returns it — verifying actual behavior,
not just "was method X called?".

Other test files outside this directory can import these fakes directly::

    from tests.unit.use_cases.conftest import FakeScanRepository, FakeUnitOfWork
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import pytest

from app.domain.common.errors import EntityNotFoundError
from app.domain.common.query import FilterSpec, PageSpec, SortSpec
from app.domain.common.uow import UnitOfWork
from app.domain.feature_store.models import (
    FeaturePage,
    FeatureRow,
    FeatureRowWrite,
    FeatureRunDomain,
    RunStats,
    RunStatus,
    RunType,
    validate_transition,
)
from app.domain.feature_store.ports import FeatureRunRepository, FeatureStoreRepository
from app.domain.feature_store.quality import DQResult
from app.domain.scanning.models import (
    FilterOptions,
    ProgressEvent,
    ResultPage,
    ScanResultItemDomain,
)
from app.domain.scanning.ports import (
    CancellationToken,
    ProgressSink,
    ScanRepository,
    ScanResultRepository,
    StockDataProvider,
    TaskDispatcher,
    UniverseRepository,
)
from app.scanners.base_screener import StockData


# ---------------------------------------------------------------------------
# Mutable ORM-like test records
# ---------------------------------------------------------------------------


@dataclass
class FakeScan:
    """Mutable in-memory scan record (mimics SQLAlchemy ORM model).

    Intentionally *not* frozen — use cases mutate the returned object
    directly (e.g. ``scan.status = "failed"``), matching ORM behaviour.
    """

    scan_id: str
    status: str = "queued"
    screener_types: list[str] | None = None
    composite_method: str | None = None
    total_stocks: int | None = None
    passed_stocks: int | None = None
    completed_at: Any = None
    idempotency_key: str | None = None
    task_id: str | None = None
    universe: str | None = None
    universe_key: str | None = None
    universe_type: str | None = None
    universe_exchange: str | None = None
    universe_index: str | None = None
    universe_symbols: list[str] | None = None
    criteria: dict | None = None


# Alias kept for backward compatibility with scanning_fakes.py consumers.
_ScanRecord = FakeScan


# ---------------------------------------------------------------------------
# Fake repositories
# ---------------------------------------------------------------------------


class FakeScanRepository(ScanRepository):
    """Full-featured in-memory scan repository.

    Supports idempotency key lookup and status transition history tracking,
    covering the needs of all three scanning use cases.
    """

    def __init__(self) -> None:
        self.scans: dict[str, FakeScan] = {}
        self.rows: list[FakeScan] = []  # insertion-order list (for test_create_scan)
        self.status_history: list[tuple[str, str]] = []

    def create(self, *, scan_id: str, **fields) -> FakeScan:
        scan = FakeScan(scan_id=scan_id, **fields)
        self.scans[scan_id] = scan
        self.rows.append(scan)
        return scan

    def get_by_scan_id(self, scan_id: str) -> FakeScan | None:
        return self.scans.get(scan_id)

    def get_by_idempotency_key(self, key: str) -> FakeScan | None:
        for s in self.scans.values():
            if getattr(s, "idempotency_key", None) == key:
                return s
        return None

    def update_status(self, scan_id: str, status: str, **fields) -> None:
        scan = self.scans.get(scan_id)
        if scan is None:
            raise ValueError(f"Cannot update non-existent scan: {scan_id}")
        scan.status = status
        for k, v in fields.items():
            if hasattr(scan, k):
                setattr(scan, k, v)
        self.status_history.append((scan_id, status))


class FakeScanResultRepository(ScanResultRepository):
    """Configurable in-memory scan result repository.

    By default returns empty results.  Pass ``items`` for query tests
    or use ``persist_orchestrator_results`` to accumulate data during
    RunBulkScan tests.
    """

    def __init__(
        self,
        *,
        items: list[ScanResultItemDomain] | None = None,
    ) -> None:
        self._items = items or []
        self._persisted_results: list[tuple[str, str, dict]] = []
        self.last_query_args: dict | None = None

    def bulk_insert(self, rows: list[dict]) -> int:
        return len(rows)

    def persist_orchestrator_results(
        self, scan_id: str, results: list[tuple[str, dict]]
    ) -> int:
        for symbol, result in results:
            self._persisted_results.append((scan_id, symbol, result))
        return len(results)

    def count_by_scan_id(self, scan_id: str) -> int:
        return sum(1 for sid, _, _ in self._persisted_results if sid == scan_id)

    def query(self, scan_id, spec, *, include_sparklines=True):
        self.last_query_args = {
            "scan_id": scan_id,
            "spec": spec,
            "include_sparklines": include_sparklines,
        }
        page_items = self._items[spec.page.offset : spec.page.offset + spec.page.limit]
        return ResultPage(
            items=tuple(page_items),
            total=len(self._items),
            page=spec.page.page,
            per_page=spec.page.per_page,
        )

    def query_all(self, scan_id, filters, sort, *, include_sparklines=False):
        return tuple(self._items)

    def get_filter_options(self, scan_id):
        return FilterOptions(ibd_industries=(), gics_sectors=(), ratings=())

    def get_by_symbol(self, scan_id: str, symbol: str) -> ScanResultItemDomain | None:
        return next((i for i in self._items if i.symbol == symbol), None)

    def get_peers_by_industry(
        self, scan_id: str, ibd_industry_group: str
    ) -> tuple[ScanResultItemDomain, ...]:
        return ()

    def get_peers_by_sector(
        self, scan_id: str, gics_sector: str
    ) -> tuple[ScanResultItemDomain, ...]:
        return ()


class FakeUniverseRepository(UniverseRepository):
    """In-memory universe repository returning a configurable symbol list."""

    def __init__(self, symbols: list[str] | None = None) -> None:
        self._symbols = symbols or []
        self.resolve_calls: list[object] = []

    def resolve_symbols(self, universe_def: object) -> list[str]:
        self.resolve_calls.append(universe_def)
        return self._symbols


# ---------------------------------------------------------------------------
# Fake infrastructure services
# ---------------------------------------------------------------------------


class FakeTaskDispatcher(TaskDispatcher):
    """Records all dispatch calls; optionally raises on dispatch."""

    def __init__(
        self,
        task_id: str = "fake-task-123",
        *,
        should_fail: bool = False,
    ) -> None:
        self._task_id = task_id
        self._should_fail = should_fail
        self.dispatched: list[tuple[str, list[str], dict]] = []

    def dispatch_scan(
        self, scan_id: str, symbols: list[str], criteria: dict
    ) -> str:
        if self._should_fail:
            raise RuntimeError("Celery is down")
        self.dispatched.append((scan_id, symbols, criteria))
        return self._task_id


class FakeStockDataProvider(StockDataProvider):
    """Returns deterministic StockData with synthetic OHLCV data.

    Useful for testing scanner-level code that depends on
    ``StockDataProvider``, though most use case tests only need
    ``FakeScanner`` (which sits one layer above).
    """

    def __init__(
        self,
        *,
        default_price: float = 100.0,
        price_days: int = 252,
    ) -> None:
        self._default_price = default_price
        self._price_days = price_days
        self.prepare_calls: list[str] = []

    def prepare_data(
        self, symbol: str, requirements: object, *, allow_partial: bool = True
    ) -> StockData:
        self.prepare_calls.append(symbol)
        return self._make_stock_data(symbol)

    def prepare_data_bulk(
        self, symbols: list[str], requirements: object, *, allow_partial: bool = True
    ) -> dict[str, StockData]:
        return {s: self.prepare_data(s, requirements) for s in symbols}

    def _make_stock_data(self, symbol: str) -> StockData:
        dates = pd.date_range(
            end=datetime.now(), periods=self._price_days, freq="B"
        )
        price = self._default_price
        df = pd.DataFrame(
            {
                "Open": price,
                "High": price * 1.02,
                "Low": price * 0.98,
                "Close": price,
                "Volume": 1_000_000,
            },
            index=dates,
        )
        return StockData(
            symbol=symbol,
            price_data=df,
            benchmark_data=df.copy(),
            fundamentals={"market_cap": 1_000_000_000},
        )


# ---------------------------------------------------------------------------
# Fake workflow collaborators
# ---------------------------------------------------------------------------


class FakeScanner:
    """Fake StockScanner (satisfies Protocol) with configurable results.

    Default: every symbol passes with score 75.
    Override per-symbol via ``results`` dict.
    """

    def __init__(self, results: dict[str, dict] | None = None) -> None:
        self._results = results or {}
        self.calls: list[str] = []

    def scan_stock_multi(
        self,
        symbol: str,
        screener_names: list[str],
        criteria: dict | None = None,
        composite_method: str = "weighted_average",
    ) -> dict:
        self.calls.append(symbol)
        return self._results.get(
            symbol,
            {
                "composite_score": 75.0,
                "rating": "Buy",
                "passes_template": True,
                "current_price": 100.0,
            },
        )


class FakeProgressSink(ProgressSink):
    """Collects all emitted progress events in a list."""

    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def emit(self, event: ProgressEvent) -> None:
        self.events.append(event)


class FakeCancellationToken(CancellationToken):
    """Configurable cancellation: cancels after *cancel_after* checks.

    ``cancel_after=None`` (default) means never cancel.
    ``cancel_after=1`` cancels on the second call to ``is_cancelled()``.
    """

    def __init__(self, cancel_after: int | None = None) -> None:
        self._cancel_after = cancel_after
        self._checks = 0

    def is_cancelled(self) -> bool:
        self._checks += 1
        if self._cancel_after is not None and self._checks > self._cancel_after:
            return True
        return False


# ---------------------------------------------------------------------------
# Feature store fake repositories
# ---------------------------------------------------------------------------


class FakeFeatureRunRepository(FeatureRunRepository):
    """In-memory feature run repository for use case tests."""

    def __init__(self) -> None:
        self._runs: dict[int, FeatureRunDomain] = {}
        self._next_id = 1
        self._pointer_run_id: int | None = None

    def start_run(self, as_of_date, run_type, code_version=None,
                  universe_hash=None, input_hash=None,
                  correlation_id=None) -> FeatureRunDomain:
        run = FeatureRunDomain(
            id=self._next_id,
            as_of_date=as_of_date,
            run_type=run_type if isinstance(run_type, RunType) else RunType(run_type),
            status=RunStatus.RUNNING,
            created_at=datetime.now(),
            completed_at=None,
            correlation_id=correlation_id,
            code_version=code_version,
            universe_hash=universe_hash,
            input_hash=input_hash,
        )
        self._runs[self._next_id] = run
        self._next_id += 1
        return run

    def mark_completed(self, run_id, stats, warnings=()) -> FeatureRunDomain:
        run = self._get_or_raise(run_id)
        validate_transition(run.status, RunStatus.COMPLETED)
        updated = FeatureRunDomain(
            id=run.id, as_of_date=run.as_of_date, run_type=run.run_type,
            status=RunStatus.COMPLETED, created_at=run.created_at,
            completed_at=datetime.now(), correlation_id=run.correlation_id,
            code_version=run.code_version, universe_hash=run.universe_hash,
            input_hash=run.input_hash, stats=stats, warnings=tuple(warnings),
        )
        self._runs[run_id] = updated
        return updated

    def mark_quarantined(self, run_id, dq_results) -> FeatureRunDomain:
        run = self._get_or_raise(run_id)
        validate_transition(run.status, RunStatus.QUARANTINED)
        updated = FeatureRunDomain(
            id=run.id, as_of_date=run.as_of_date, run_type=run.run_type,
            status=RunStatus.QUARANTINED, created_at=run.created_at,
            completed_at=run.completed_at, correlation_id=run.correlation_id,
            code_version=run.code_version, universe_hash=run.universe_hash,
            input_hash=run.input_hash, stats=run.stats,
            warnings=tuple(r.message for r in dq_results),
        )
        self._runs[run_id] = updated
        return updated

    def publish_atomically(self, run_id) -> FeatureRunDomain:
        run = self._get_or_raise(run_id)
        validate_transition(run.status, RunStatus.PUBLISHED)
        updated = FeatureRunDomain(
            id=run.id, as_of_date=run.as_of_date, run_type=run.run_type,
            status=RunStatus.PUBLISHED, created_at=run.created_at,
            completed_at=run.completed_at, correlation_id=run.correlation_id,
            code_version=run.code_version, universe_hash=run.universe_hash,
            input_hash=run.input_hash, stats=run.stats, warnings=run.warnings,
        )
        self._runs[run_id] = updated
        self._pointer_run_id = run_id
        return updated

    def get_latest_published(self) -> FeatureRunDomain | None:
        if self._pointer_run_id is None:
            return None
        return self._runs.get(self._pointer_run_id)

    def get_run(self, run_id) -> FeatureRunDomain:
        return self._get_or_raise(run_id)

    def _get_or_raise(self, run_id: int) -> FeatureRunDomain:
        run = self._runs.get(run_id)
        if run is None:
            raise EntityNotFoundError("FeatureRun", run_id)
        return run


class FakeFeatureStoreRepository(FeatureStoreRepository):
    """In-memory feature store repository for use case tests."""

    def __init__(self) -> None:
        self._rows: dict[int, list[FeatureRow]] = {}  # run_id -> rows
        self._universe: dict[int, list[str]] = {}
        self._pointer_run_id: int | None = None

    def upsert_snapshot_rows(self, run_id, rows) -> int:
        existing = {r.symbol: r for r in self._rows.get(run_id, [])}
        for row in rows:
            fr = FeatureRow(
                run_id=run_id, symbol=row.symbol, as_of_date=row.as_of_date,
                composite_score=row.composite_score, overall_rating=row.overall_rating,
                passes_count=row.passes_count, details=row.details,
            )
            existing[row.symbol] = fr
        self._rows[run_id] = list(existing.values())
        return len(rows)

    def save_run_universe_symbols(self, run_id, symbols) -> None:
        self._universe[run_id] = list(symbols)

    def count_by_run_id(self, run_id) -> int:
        return len(self._rows.get(run_id, []))

    def query_latest(self, filters=None, sort=None, page=None) -> FeaturePage:
        if self._pointer_run_id is None:
            return FeaturePage(items=(), total=0, page=1, per_page=50)
        return self._build_page(self._pointer_run_id, page)

    def query_run(self, run_id, filters=None, sort=None, page=None) -> FeaturePage:
        if run_id not in self._rows:
            raise EntityNotFoundError("FeatureRun", run_id)
        return self._build_page(run_id, page)

    def _build_page(self, run_id: int, page: PageSpec | None) -> FeaturePage:
        p = page or PageSpec()
        all_rows = self._rows.get(run_id, [])
        items = tuple(all_rows[p.offset : p.offset + p.limit])
        return FeaturePage(items=items, total=len(all_rows), page=p.page, per_page=p.per_page)


# ---------------------------------------------------------------------------
# Fake unit of work
# ---------------------------------------------------------------------------


class FakeUnitOfWork(UnitOfWork):
    """In-memory UoW wiring up all fake repositories.

    Accepts any repository subclass for maximum test flexibility.
    """

    def __init__(
        self,
        *,
        scans: ScanRepository | None = None,
        scan_results: ScanResultRepository | None = None,
        universe: UniverseRepository | None = None,
        feature_runs: FeatureRunRepository | None = None,
        feature_store: FeatureStoreRepository | None = None,
    ) -> None:
        self.scans = scans or FakeScanRepository()
        self.scan_results = scan_results or FakeScanResultRepository()
        self.universe = universe or FakeUniverseRepository()
        self.feature_runs = feature_runs or FakeFeatureRunRepository()
        self.feature_store = feature_store or FakeFeatureStoreRepository()
        self.committed = 0
        self.rolled_back = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.rollback()

    def commit(self):
        self.committed += 1

    def rollback(self):
        self.rolled_back += 1


# ---------------------------------------------------------------------------
# Domain object helpers
# ---------------------------------------------------------------------------


def make_domain_item(
    symbol: str = "AAPL",
    score: float = 85.0,
    **extra_extended_fields: Any,
) -> ScanResultItemDomain:
    """Construct a ``ScanResultItemDomain`` with sensible defaults."""
    ef: dict[str, Any] = {"company_name": f"{symbol} Inc"}
    ef.update(extra_extended_fields)
    return ScanResultItemDomain(
        symbol=symbol,
        composite_score=score,
        rating="Buy",
        current_price=150.0,
        screener_outputs={},
        screeners_run=["minervini"],
        composite_method="weighted_average",
        screeners_passed=1,
        screeners_total=1,
        extended_fields=ef,
    )


def setup_scan(uow: FakeUnitOfWork, scan_id: str = "scan-123") -> None:
    """Pre-populate a scan record so the use case doesn't raise NotFound."""
    uow.scans.create(scan_id=scan_id, status="completed")


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_uow() -> FakeUnitOfWork:
    return FakeUnitOfWork()


@pytest.fixture
def fake_scanner() -> FakeScanner:
    return FakeScanner()


@pytest.fixture
def fake_progress() -> FakeProgressSink:
    return FakeProgressSink()


@pytest.fixture
def fake_cancel() -> FakeCancellationToken:
    return FakeCancellationToken()


@pytest.fixture
def fake_dispatcher() -> FakeTaskDispatcher:
    return FakeTaskDispatcher()


@pytest.fixture
def fake_stock_data_provider() -> FakeStockDataProvider:
    return FakeStockDataProvider()
