"""Shared test fakes for the scanning bounded context.

Consolidates the common fakes (FakeScanRepository, FakeUnitOfWork, etc.)
that were previously copy-pasted across multiple test files.

Each test file keeps its own ``FakeScanResultRepository`` subclass when it
needs custom return values, but imports the shared base and helpers from here.
"""

from __future__ import annotations

from typing import Any

from app.domain.common.uow import UnitOfWork
from app.domain.scanning.models import (
    FilterOptions,
    ResultPage,
    ScanResultItemDomain,
)
from app.domain.scanning.ports import (
    ScanRepository,
    ScanResultRepository,
    UniverseRepository,
)


# ---------------------------------------------------------------------------
# Minimal scan record
# ---------------------------------------------------------------------------


class _ScanRecord:
    """Minimal in-memory scan record used by FakeScanRepository."""

    def __init__(self, **fields: Any) -> None:
        for k, v in fields.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Fake repositories
# ---------------------------------------------------------------------------


class FakeScanRepository(ScanRepository):
    """In-memory scan repository shared across test files."""

    def __init__(self) -> None:
        self.rows: list[_ScanRecord] = []

    def create(self, *, scan_id: str, **fields: Any) -> _ScanRecord:
        rec = _ScanRecord(scan_id=scan_id, **fields)
        self.rows.append(rec)
        return rec

    def get_by_scan_id(self, scan_id: str) -> _ScanRecord | None:
        return next((r for r in self.rows if r.scan_id == scan_id), None)

    def get_by_idempotency_key(self, key: str) -> _ScanRecord | None:
        return None

    def update_status(self, scan_id: str, status: str, **fields: Any) -> None:
        pass


class FakeScanResultRepository(ScanResultRepository):
    """Base in-memory scan result repository.

    Provides default stubs for *all* abstract methods (including
    ``get_by_symbol``).  Test files subclass this to customise only the
    methods under test.
    """

    def bulk_insert(self, rows: list[dict]) -> int:
        return len(rows)

    def persist_orchestrator_results(
        self, scan_id: str, results: list[tuple[str, dict]]
    ) -> int:
        return len(results)

    def count_by_scan_id(self, scan_id: str) -> int:
        return 0

    def query(self, scan_id, spec, *, include_sparklines=True):
        return ResultPage(items=(), total=0, page=1, per_page=50)

    def get_filter_options(self, scan_id):
        return FilterOptions(ibd_industries=(), gics_sectors=(), ratings=())

    def get_by_symbol(self, scan_id: str, symbol: str) -> ScanResultItemDomain | None:
        return None

    def get_peers_by_industry(
        self, scan_id: str, ibd_industry_group: str
    ) -> tuple[ScanResultItemDomain, ...]:
        return ()

    def get_peers_by_sector(
        self, scan_id: str, gics_sector: str
    ) -> tuple[ScanResultItemDomain, ...]:
        return ()


class FakeUniverseRepository(UniverseRepository):
    """In-memory universe repository (returns empty by default)."""

    def resolve_symbols(self, universe_def: object) -> list[str]:
        return []


# ---------------------------------------------------------------------------
# Fake unit of work
# ---------------------------------------------------------------------------


class FakeUnitOfWork(UnitOfWork):
    """In-memory UoW that wires up fake repositories.

    Accepts any ``ScanResultRepository`` subclass so that test files can
    plug in their own specialised fakes.
    """

    def __init__(
        self,
        *,
        scans: ScanRepository | None = None,
        scan_results: ScanResultRepository | None = None,
    ) -> None:
        self.scans = scans or FakeScanRepository()
        self.scan_results = scan_results or FakeScanResultRepository()
        self.universe = FakeUniverseRepository()
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
