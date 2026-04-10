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

from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, Iterator

from app.database import SessionLocal
from app.domain.scanning.ports import StockDataProvider, TaskDispatcher
from app.services.job_backend import JobBackend, CeleryJobBackend
from app.services.redis_pool import get_redis_client

if TYPE_CHECKING:
    from app.infra.db.uow import SqlUnitOfWork
    from app.infra.providers.stock_data import DataPrepStockDataProvider
    from app.scanners.scan_orchestrator import ScanOrchestrator
    from app.services.benchmark_cache_service import BenchmarkCacheService
    from app.services.fundamentals_cache_service import FundamentalsCacheService
    from app.services.ibd_group_rank_service import IBDGroupRankService
    from app.services.llm.groq_key_manager import GroqKeyManager
    from app.services.llm.zai_key_manager import ZAIKeyManager
    from app.services.price_cache_service import PriceCacheService
    from app.services.task_registry_service import TaskRegistryService
    from app.services.ui_snapshot_service import UISnapshotService
    from app.tasks.data_fetch_lock import DataFetchLock
    from app.use_cases.feature_store.build_daily_snapshot import (
        BuildDailyFeatureSnapshotUseCase,
    )
    from app.use_cases.feature_store.compare_runs import CompareFeatureRunsUseCase
    from app.use_cases.feature_store.list_runs import ListFeatureRunsUseCase
    from app.use_cases.scanning.create_scan import CreateScanUseCase
    from app.use_cases.scanning.explain_stock import ExplainStockUseCase
    from app.use_cases.scanning.export_scan_results import ExportScanResultsUseCase
    from app.use_cases.scanning.get_filter_options import GetFilterOptionsUseCase
    from app.use_cases.scanning.get_peers import GetPeersUseCase
    from app.use_cases.scanning.get_scan_results import GetScanResultsUseCase
    from app.use_cases.scanning.get_scan_symbols import GetScanSymbolsUseCase
    from app.use_cases.scanning.get_setup_details import GetSetupDetailsUseCase
    from app.use_cases.scanning.get_single_result import GetSingleResultUseCase
    from app.use_cases.scanning.run_bulk_scan import RunBulkScanUseCase


@dataclass(frozen=True)
class CacheBundle:
    """Shared cache service bundle for explicit dependency injection."""

    price: PriceCacheService
    fundamentals: FundamentalsCacheService
    benchmark: BenchmarkCacheService


# ── Unit of Work ─────────────────────────────────────────────────────────


def get_uow() -> Iterator[SqlUnitOfWork]:
    """Yield a SqlUnitOfWork bound to SessionLocal.

    Designed for FastAPI Depends()::

        uow: SqlUnitOfWork = Depends(get_uow)
    """
    from app.infra.db.uow import SqlUnitOfWork

    uow = SqlUnitOfWork(SessionLocal)
    yield uow


# ── Task Dispatchers ────────────────────────────────────────────────────

_task_dispatcher: TaskDispatcher | None = None
_job_backend: JobBackend | None = None
_ui_snapshot_service: UISnapshotService | None = None
_cache_bundle: CacheBundle | None = None
_group_rank_service: IBDGroupRankService | None = None
_task_registry_service: TaskRegistryService | None = None
_data_fetch_lock: DataFetchLock | None = None
_groq_key_manager: GroqKeyManager | None = None
_zai_key_manager: ZAIKeyManager | None = None
_singleton_init_lock = RLock()


def get_job_backend() -> JobBackend:
    """Return the configured async job backend."""
    global _job_backend
    if _job_backend is None:
        with _singleton_init_lock:
            if _job_backend is None:
                _job_backend = CeleryJobBackend()
    return _job_backend


def get_task_dispatcher() -> TaskDispatcher:
    """Return the Celery task dispatcher."""
    global _task_dispatcher
    if _task_dispatcher is None:
        with _singleton_init_lock:
            if _task_dispatcher is None:
                from app.infra.tasks.dispatcher import CeleryTaskDispatcher

                _task_dispatcher = CeleryTaskDispatcher()
    return _task_dispatcher


def get_ui_snapshot_service() -> UISnapshotService:
    """Return the singleton UI bootstrap snapshot publisher."""
    global _ui_snapshot_service
    if _ui_snapshot_service is None:
        with _singleton_init_lock:
            if _ui_snapshot_service is None:
                from app.services.ui_snapshot_service import UISnapshotService

                _ui_snapshot_service = UISnapshotService(session_factory=SessionLocal)
    return _ui_snapshot_service


def get_cache_bundle() -> CacheBundle:
    """Return shared cache services wired with explicit dependencies."""
    global _cache_bundle
    if _cache_bundle is None:
        with _singleton_init_lock:
            if _cache_bundle is None:
                from app.services.benchmark_cache_service import BenchmarkCacheService
                from app.services.fundamentals_cache_service import FundamentalsCacheService
                from app.services.price_cache_service import PriceCacheService

                redis_client = get_redis_client()
                _cache_bundle = CacheBundle(
                    price=PriceCacheService(
                        redis_client=redis_client,
                        session_factory=SessionLocal,
                    ),
                    fundamentals=FundamentalsCacheService(
                        redis_client=redis_client,
                        session_factory=SessionLocal,
                    ),
                    benchmark=BenchmarkCacheService(
                        redis_client=redis_client,
                        session_factory=SessionLocal,
                    ),
                )
    return _cache_bundle


def get_price_cache() -> PriceCacheService:
    return get_cache_bundle().price


def get_fundamentals_cache() -> FundamentalsCacheService:
    return get_cache_bundle().fundamentals


def get_benchmark_cache() -> BenchmarkCacheService:
    return get_cache_bundle().benchmark


def get_group_rank_service() -> IBDGroupRankService:
    """Return shared group-rank service."""
    global _group_rank_service
    if _group_rank_service is None:
        with _singleton_init_lock:
            if _group_rank_service is None:
                from app.services.ibd_group_rank_service import IBDGroupRankService

                _group_rank_service = IBDGroupRankService(
                    price_cache=get_price_cache(),
                    benchmark_cache=get_benchmark_cache(),
                )
    return _group_rank_service


def get_task_registry_service() -> TaskRegistryService:
    """Return shared task-registry service."""
    global _task_registry_service
    if _task_registry_service is None:
        with _singleton_init_lock:
            if _task_registry_service is None:
                from app.services.task_registry_service import TaskRegistryService

                _task_registry_service = TaskRegistryService()
    return _task_registry_service


def get_data_fetch_lock() -> DataFetchLock:
    """Return process-wide distributed lock instance."""
    global _data_fetch_lock
    if _data_fetch_lock is None:
        with _singleton_init_lock:
            if _data_fetch_lock is None:
                from app.tasks.data_fetch_lock import DataFetchLock

                _data_fetch_lock = DataFetchLock()
    return _data_fetch_lock


def get_groq_key_manager() -> GroqKeyManager:
    """Return process-wide Groq key manager."""
    global _groq_key_manager
    if _groq_key_manager is None:
        with _singleton_init_lock:
            if _groq_key_manager is None:
                from app.services.llm.groq_key_manager import GroqKeyManager, _get_keys_from_settings

                _groq_key_manager = GroqKeyManager(keys=_get_keys_from_settings() or [])
    return _groq_key_manager


def get_zai_key_manager() -> ZAIKeyManager:
    """Return process-wide Z.AI key manager."""
    global _zai_key_manager
    if _zai_key_manager is None:
        with _singleton_init_lock:
            if _zai_key_manager is None:
                from app.services.llm.zai_key_manager import ZAIKeyManager, _get_keys_from_settings

                _zai_key_manager = ZAIKeyManager(keys=_get_keys_from_settings() or [])
    return _zai_key_manager


# ── Use Cases ────────────────────────────────────────────────────────────


def get_create_scan_use_case() -> CreateScanUseCase:
    """Build a CreateScanUseCase wired with infrastructure adapters."""
    from app.use_cases.scanning.create_scan import CreateScanUseCase

    return CreateScanUseCase(dispatcher=get_task_dispatcher())


def get_get_scan_results_use_case() -> GetScanResultsUseCase:
    """Build a GetScanResultsUseCase (no extra dependencies — reads via UoW)."""
    from app.use_cases.scanning.get_scan_results import GetScanResultsUseCase

    return GetScanResultsUseCase()


def get_get_scan_symbols_use_case() -> GetScanSymbolsUseCase:
    """Build a GetScanSymbolsUseCase (no extra dependencies — reads via UoW)."""
    from app.use_cases.scanning.get_scan_symbols import GetScanSymbolsUseCase

    return GetScanSymbolsUseCase()


def get_get_filter_options_use_case() -> GetFilterOptionsUseCase:
    """Build a GetFilterOptionsUseCase (no extra dependencies — reads via UoW)."""
    from app.use_cases.scanning.get_filter_options import GetFilterOptionsUseCase

    return GetFilterOptionsUseCase()


def get_get_single_result_use_case() -> GetSingleResultUseCase:
    """Build a GetSingleResultUseCase (no extra dependencies — reads via UoW)."""
    from app.use_cases.scanning.get_single_result import GetSingleResultUseCase

    return GetSingleResultUseCase()


def get_get_setup_details_use_case() -> GetSetupDetailsUseCase:
    """Build a GetSetupDetailsUseCase (no extra dependencies — reads via UoW)."""
    from app.use_cases.scanning.get_setup_details import GetSetupDetailsUseCase

    return GetSetupDetailsUseCase()


def get_get_peers_use_case() -> GetPeersUseCase:
    """Build a GetPeersUseCase (no extra dependencies — reads via UoW)."""
    from app.use_cases.scanning.get_peers import GetPeersUseCase

    return GetPeersUseCase()


def get_export_scan_results_use_case() -> ExportScanResultsUseCase:
    """Build an ExportScanResultsUseCase (no extra dependencies — reads via UoW)."""
    from app.use_cases.scanning.export_scan_results import ExportScanResultsUseCase

    return ExportScanResultsUseCase()


def get_run_bulk_scan_use_case() -> RunBulkScanUseCase:
    """Build a RunBulkScanUseCase wired with the scan orchestrator."""
    from app.use_cases.scanning.run_bulk_scan import RunBulkScanUseCase

    return RunBulkScanUseCase(
        scanner=get_scan_orchestrator(),
        data_provider=get_stock_data_provider(),
    )


def get_explain_stock_use_case() -> ExplainStockUseCase:
    """Build an ExplainStockUseCase (no extra dependencies — reads via UoW)."""
    from app.use_cases.scanning.explain_stock import ExplainStockUseCase

    return ExplainStockUseCase()


def get_list_feature_runs_use_case() -> ListFeatureRunsUseCase:
    """Build a ListFeatureRunsUseCase (no extra dependencies — reads via UoW)."""
    from app.use_cases.feature_store.list_runs import ListFeatureRunsUseCase

    return ListFeatureRunsUseCase()


def get_compare_feature_runs_use_case() -> CompareFeatureRunsUseCase:
    """Build a CompareFeatureRunsUseCase (no extra dependencies — reads via UoW)."""
    from app.use_cases.feature_store.compare_runs import CompareFeatureRunsUseCase

    return CompareFeatureRunsUseCase()


def get_build_daily_snapshot_use_case() -> BuildDailyFeatureSnapshotUseCase:
    """Build a BuildDailyFeatureSnapshotUseCase wired with the scan orchestrator."""
    from app.use_cases.feature_store.build_daily_snapshot import (
        BuildDailyFeatureSnapshotUseCase,
    )

    return BuildDailyFeatureSnapshotUseCase(
        scanner=get_scan_orchestrator(),
        data_provider=get_stock_data_provider(),
    )


# ── Providers ────────────────────────────────────────────────────────────

_stock_data_provider: DataPrepStockDataProvider | None = None


def get_stock_data_provider() -> StockDataProvider:
    """Return a singleton StockDataProvider (wraps DataPreparationLayer)."""
    global _stock_data_provider
    if _stock_data_provider is None:
        with _singleton_init_lock:
            if _stock_data_provider is None:
                from app.infra.providers.stock_data import DataPrepStockDataProvider

                _stock_data_provider = DataPrepStockDataProvider(
                    cache_bundle=get_cache_bundle(),
                )
    return _stock_data_provider


# ── Orchestrator ────────────────────────────────────────────────────

_scan_orchestrator: ScanOrchestrator | None = None


def get_scan_orchestrator() -> ScanOrchestrator:
    """Return a singleton ScanOrchestrator wired with production dependencies."""
    global _scan_orchestrator
    if _scan_orchestrator is None:
        with _singleton_init_lock:
            if _scan_orchestrator is None:
                from app.scanners.scan_orchestrator import ScanOrchestrator
                from app.scanners.screener_registry import screener_registry

                _scan_orchestrator = ScanOrchestrator(
                    data_provider=get_stock_data_provider(),
                    registry=screener_registry,
                )
    return _scan_orchestrator


def _reset_singletons_for_tests() -> None:
    """Reset bootstrap singletons for test isolation."""
    global _task_dispatcher
    global _job_backend
    global _ui_snapshot_service
    global _cache_bundle
    global _group_rank_service
    global _task_registry_service
    global _data_fetch_lock
    global _groq_key_manager
    global _zai_key_manager
    global _stock_data_provider
    global _scan_orchestrator

    _task_dispatcher = None
    _job_backend = None
    _ui_snapshot_service = None
    _cache_bundle = None
    _group_rank_service = None
    _task_registry_service = None
    _data_fetch_lock = None
    _groq_key_manager = None
    _zai_key_manager = None
    _stock_data_provider = None
    _scan_orchestrator = None
