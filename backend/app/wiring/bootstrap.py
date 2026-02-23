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
from app.use_cases.scanning.get_filter_options import GetFilterOptionsUseCase
from app.use_cases.scanning.get_peers import GetPeersUseCase
from app.use_cases.scanning.get_scan_results import GetScanResultsUseCase
from app.use_cases.scanning.get_scan_symbols import GetScanSymbolsUseCase
from app.use_cases.scanning.get_single_result import GetSingleResultUseCase
from app.use_cases.scanning.export_scan_results import ExportScanResultsUseCase
from app.use_cases.scanning.run_bulk_scan import RunBulkScanUseCase
from app.use_cases.feature_store.build_daily_snapshot import (
    BuildDailyFeatureSnapshotUseCase,
)
from app.use_cases.feature_store.compare_runs import CompareFeatureRunsUseCase
from app.use_cases.feature_store.list_runs import ListFeatureRunsUseCase
from app.use_cases.scanning.explain_stock import ExplainStockUseCase
from app.use_cases.scanning.get_setup_details import GetSetupDetailsUseCase


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


def get_get_scan_results_use_case() -> GetScanResultsUseCase:
    """Build a GetScanResultsUseCase (no extra dependencies — reads via UoW)."""
    return GetScanResultsUseCase()


def get_get_scan_symbols_use_case() -> GetScanSymbolsUseCase:
    """Build a GetScanSymbolsUseCase (no extra dependencies — reads via UoW)."""
    return GetScanSymbolsUseCase()


def get_get_filter_options_use_case() -> GetFilterOptionsUseCase:
    """Build a GetFilterOptionsUseCase (no extra dependencies — reads via UoW)."""
    return GetFilterOptionsUseCase()


def get_get_single_result_use_case() -> GetSingleResultUseCase:
    """Build a GetSingleResultUseCase (no extra dependencies — reads via UoW)."""
    return GetSingleResultUseCase()


def get_get_setup_details_use_case() -> GetSetupDetailsUseCase:
    """Build a GetSetupDetailsUseCase (no extra dependencies — reads via UoW)."""
    return GetSetupDetailsUseCase()


def get_get_peers_use_case() -> GetPeersUseCase:
    """Build a GetPeersUseCase (no extra dependencies — reads via UoW)."""
    return GetPeersUseCase()


def get_export_scan_results_use_case() -> ExportScanResultsUseCase:
    """Build an ExportScanResultsUseCase (no extra dependencies — reads via UoW)."""
    return ExportScanResultsUseCase()


def get_run_bulk_scan_use_case() -> RunBulkScanUseCase:
    """Build a RunBulkScanUseCase wired with the scan orchestrator."""
    return RunBulkScanUseCase(
        scanner=get_scan_orchestrator(),
        data_provider=get_stock_data_provider(),
    )


def get_explain_stock_use_case() -> ExplainStockUseCase:
    """Build an ExplainStockUseCase (no extra dependencies — reads via UoW)."""
    return ExplainStockUseCase()


def get_list_feature_runs_use_case() -> ListFeatureRunsUseCase:
    """Build a ListFeatureRunsUseCase (no extra dependencies — reads via UoW)."""
    return ListFeatureRunsUseCase()


def get_compare_feature_runs_use_case() -> CompareFeatureRunsUseCase:
    """Build a CompareFeatureRunsUseCase (no extra dependencies — reads via UoW)."""
    return CompareFeatureRunsUseCase()


def get_build_daily_snapshot_use_case() -> BuildDailyFeatureSnapshotUseCase:
    """Build a BuildDailyFeatureSnapshotUseCase wired with the scan orchestrator."""
    return BuildDailyFeatureSnapshotUseCase(scanner=get_scan_orchestrator())


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
