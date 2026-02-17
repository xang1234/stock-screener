"""Dependency injection bootstrap — the single place that binds ports to adapters.

Every factory function here can be used as a FastAPI ``Depends()`` target.
Routers never import concrete implementations directly; they depend on
the abstractions returned by these factories.

Example usage in a router::

    from app.wiring.bootstrap import get_uow, get_create_scan_use_case

    @router.post("/scans")
    async def create_scan(
        request: ScanCreateRequest,
        uow: SqlUnitOfWork = Depends(get_uow),
        use_case: CreateScanUseCase = Depends(get_create_scan_use_case),
    ):
        result = use_case.execute(uow, command)
"""

from __future__ import annotations

from typing import Iterator

from app.database import SessionLocal
from app.domain.scanning.ports import StockDataProvider, TaskDispatcher
from app.infra.db.uow import SqlUnitOfWork
from app.infra.providers.stock_data import DataPrepStockDataProvider
from app.infra.tasks.dispatcher import CeleryTaskDispatcher
from app.scanners.scan_orchestrator import ScanOrchestrator
from app.scanners.screener_registry import screener_registry
from app.use_cases.scanning.create_scan import CreateScanUseCase


# ── Unit of Work ─────────────────────────────────────────────────────────


def get_uow() -> Iterator[SqlUnitOfWork]:
    """Yield a SqlUnitOfWork bound to SessionLocal.

    Designed for FastAPI Depends()::

        uow: SqlUnitOfWork = Depends(get_uow)
    """
    uow = SqlUnitOfWork(SessionLocal)
    yield uow


# ── Task Dispatchers ────────────────────────────────────────────────────

_task_dispatcher: CeleryTaskDispatcher | None = None


def get_task_dispatcher() -> TaskDispatcher:
    """Return a singleton CeleryTaskDispatcher."""
    global _task_dispatcher
    if _task_dispatcher is None:
        _task_dispatcher = CeleryTaskDispatcher()
    return _task_dispatcher


# ── Use Cases ────────────────────────────────────────────────────────────


def get_create_scan_use_case() -> CreateScanUseCase:
    """Build a CreateScanUseCase wired with infrastructure adapters."""
    return CreateScanUseCase(dispatcher=get_task_dispatcher())


# ── Providers ────────────────────────────────────────────────────────────

_stock_data_provider: DataPrepStockDataProvider | None = None


def get_stock_data_provider() -> StockDataProvider:
    """Return a singleton StockDataProvider (wraps DataPreparationLayer)."""
    global _stock_data_provider
    if _stock_data_provider is None:
        _stock_data_provider = DataPrepStockDataProvider()
    return _stock_data_provider


# ── Orchestrator ────────────────────────────────────────────────────

_scan_orchestrator: ScanOrchestrator | None = None


def get_scan_orchestrator() -> ScanOrchestrator:
    """Return a singleton ScanOrchestrator wired with production dependencies."""
    global _scan_orchestrator
    if _scan_orchestrator is None:
        _scan_orchestrator = ScanOrchestrator(
            data_provider=get_stock_data_provider(),
            registry=screener_registry,
        )
    return _scan_orchestrator
