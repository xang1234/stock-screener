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

from contextvars import ContextVar, Token
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, Callable, Iterator, TypeAlias

from fastapi import Request
from sqlalchemy.orm import Session

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


SessionFactory: TypeAlias = Callable[[], Session]


@dataclass(frozen=True)
class CacheBundle:
    """Shared cache service bundle for explicit dependency injection."""

    price: PriceCacheService
    fundamentals: FundamentalsCacheService
    benchmark: BenchmarkCacheService


class RuntimeServices:
    """Process-scoped service container with lazy, lock-protected initialization."""

    def __init__(self, *, session_factory: SessionFactory = SessionLocal) -> None:
        self._session_factory = session_factory
        self._init_lock = RLock()
        self._task_dispatcher: TaskDispatcher | None = None
        self._job_backend: JobBackend | None = None
        self._ui_snapshot_service: UISnapshotService | None = None
        self._cache_bundle: CacheBundle | None = None
        self._group_rank_service: IBDGroupRankService | None = None
        self._task_registry_service: TaskRegistryService | None = None
        self._data_fetch_lock: DataFetchLock | None = None
        self._groq_key_manager: GroqKeyManager | None = None
        self._zai_key_manager: ZAIKeyManager | None = None
        self._stock_data_provider: DataPrepStockDataProvider | None = None
        self._scan_orchestrator: ScanOrchestrator | None = None

    def job_backend(self) -> JobBackend:
        if self._job_backend is None:
            with self._init_lock:
                if self._job_backend is None:
                    self._job_backend = CeleryJobBackend()
        return self._job_backend

    def task_dispatcher(self) -> TaskDispatcher:
        if self._task_dispatcher is None:
            with self._init_lock:
                if self._task_dispatcher is None:
                    from app.infra.tasks.dispatcher import CeleryTaskDispatcher

                    self._task_dispatcher = CeleryTaskDispatcher()
        return self._task_dispatcher

    def ui_snapshot_service(self) -> UISnapshotService:
        if self._ui_snapshot_service is None:
            with self._init_lock:
                if self._ui_snapshot_service is None:
                    from app.services.ui_snapshot_service import UISnapshotService

                    self._ui_snapshot_service = UISnapshotService(
                        session_factory=self._session_factory
                    )
        return self._ui_snapshot_service

    def cache_bundle(self) -> CacheBundle:
        if self._cache_bundle is None:
            with self._init_lock:
                if self._cache_bundle is None:
                    from app.services.benchmark_cache_service import BenchmarkCacheService
                    from app.services.fundamentals_cache_service import FundamentalsCacheService
                    from app.services.price_cache_service import PriceCacheService

                    redis_client = get_redis_client()
                    self._cache_bundle = CacheBundle(
                        price=PriceCacheService(
                            redis_client=redis_client,
                            session_factory=self._session_factory,
                        ),
                        fundamentals=FundamentalsCacheService(
                            redis_client=redis_client,
                            session_factory=self._session_factory,
                        ),
                        benchmark=BenchmarkCacheService(
                            redis_client=redis_client,
                            session_factory=self._session_factory,
                        ),
                    )
        return self._cache_bundle

    def group_rank_service(self) -> IBDGroupRankService:
        if self._group_rank_service is None:
            with self._init_lock:
                if self._group_rank_service is None:
                    from app.services.ibd_group_rank_service import IBDGroupRankService

                    cache_bundle = self.cache_bundle()
                    self._group_rank_service = IBDGroupRankService(
                        price_cache=cache_bundle.price,
                        benchmark_cache=cache_bundle.benchmark,
                    )
        return self._group_rank_service

    def task_registry_service(self) -> TaskRegistryService:
        if self._task_registry_service is None:
            with self._init_lock:
                if self._task_registry_service is None:
                    from app.services.task_registry_service import TaskRegistryService

                    self._task_registry_service = TaskRegistryService()
        return self._task_registry_service

    def data_fetch_lock(self) -> DataFetchLock:
        if self._data_fetch_lock is None:
            with self._init_lock:
                if self._data_fetch_lock is None:
                    from app.tasks.data_fetch_lock import DataFetchLock

                    self._data_fetch_lock = DataFetchLock()
        return self._data_fetch_lock

    def groq_key_manager(self) -> GroqKeyManager:
        if self._groq_key_manager is None:
            with self._init_lock:
                if self._groq_key_manager is None:
                    from app.services.llm.groq_key_manager import GroqKeyManager, _get_keys_from_settings

                    self._groq_key_manager = GroqKeyManager(
                        keys=_get_keys_from_settings() or []
                    )
        return self._groq_key_manager

    def zai_key_manager(self) -> ZAIKeyManager:
        if self._zai_key_manager is None:
            with self._init_lock:
                if self._zai_key_manager is None:
                    from app.services.llm.zai_key_manager import ZAIKeyManager, _get_keys_from_settings

                    self._zai_key_manager = ZAIKeyManager(
                        keys=_get_keys_from_settings() or []
                    )
        return self._zai_key_manager

    def stock_data_provider(self) -> DataPrepStockDataProvider:
        if self._stock_data_provider is None:
            with self._init_lock:
                if self._stock_data_provider is None:
                    from app.infra.providers.stock_data import DataPrepStockDataProvider

                    self._stock_data_provider = DataPrepStockDataProvider(
                        cache_bundle=self.cache_bundle(),
                    )
        return self._stock_data_provider

    def scan_orchestrator(self) -> ScanOrchestrator:
        if self._scan_orchestrator is None:
            with self._init_lock:
                if self._scan_orchestrator is None:
                    from app.scanners.scan_orchestrator import ScanOrchestrator
                    from app.scanners.screener_registry import screener_registry

                    self._scan_orchestrator = ScanOrchestrator(
                        data_provider=self.stock_data_provider(),
                        registry=screener_registry,
                    )
        return self._scan_orchestrator

    def reset_for_tests(self) -> None:
        with self._init_lock:
            self._task_dispatcher = None
            self._job_backend = None
            self._ui_snapshot_service = None
            self._cache_bundle = None
            self._group_rank_service = None
            self._task_registry_service = None
            self._data_fetch_lock = None
            self._groq_key_manager = None
            self._zai_key_manager = None
            self._stock_data_provider = None
            self._scan_orchestrator = None


_runtime_services_ctx: ContextVar[RuntimeServices | None] = ContextVar(
    "runtime_services_ctx",
    default=None,
)


def build_runtime_services(
    *,
    session_factory: SessionFactory = SessionLocal,
) -> RuntimeServices:
    """Create a new process-scoped runtime services container."""
    return RuntimeServices(session_factory=session_factory)


def set_runtime_services(runtime: RuntimeServices) -> Token[RuntimeServices | None]:
    """Bind runtime services to the current context."""
    return _runtime_services_ctx.set(runtime)


def initialize_process_runtime_services(
    *,
    session_factory: SessionFactory = SessionLocal,
    force: bool = False,
) -> RuntimeServices:
    """Ensure a runtime services container is bound in the current context."""
    current = _runtime_services_ctx.get()
    if current is not None and not force:
        return current
    runtime = build_runtime_services(session_factory=session_factory)
    _runtime_services_ctx.set(runtime)
    return runtime


def clear_runtime_services() -> None:
    """Clear runtime services from the current context."""
    _runtime_services_ctx.set(None)


def get_runtime_services(request: Request) -> RuntimeServices:
    """FastAPI dependency getter for process runtime services."""
    runtime = getattr(request.app.state, "runtime_services", None)
    if runtime is None:
        raise RuntimeError("RuntimeServices are not initialized on app.state.runtime_services")
    return runtime


def _resolve_runtime_services(request: Request | None = None) -> RuntimeServices:
    if request is not None:
        request_runtime = getattr(request.app.state, "runtime_services", None)
        if request_runtime is not None:
            return request_runtime
    context_runtime = _runtime_services_ctx.get()
    if context_runtime is not None:
        return context_runtime
    # Compatibility bridge for legacy modules that still instantiate
    # service globals at import time before explicit startup hooks run.
    return initialize_process_runtime_services()


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


def get_job_backend() -> JobBackend:
    """Return the configured async job backend."""
    return _resolve_runtime_services().job_backend()


def get_task_dispatcher() -> TaskDispatcher:
    """Return the Celery task dispatcher."""
    return _resolve_runtime_services().task_dispatcher()


def get_ui_snapshot_service() -> UISnapshotService:
    """Return the process-scoped UI bootstrap snapshot publisher."""
    return _resolve_runtime_services().ui_snapshot_service()


def get_cache_bundle() -> CacheBundle:
    """Return shared cache services wired with explicit dependencies."""
    return _resolve_runtime_services().cache_bundle()


def get_price_cache() -> PriceCacheService:
    return get_cache_bundle().price


def get_fundamentals_cache() -> FundamentalsCacheService:
    return get_cache_bundle().fundamentals


def get_benchmark_cache() -> BenchmarkCacheService:
    return get_cache_bundle().benchmark


def get_group_rank_service() -> IBDGroupRankService:
    """Return process-scoped group-rank service."""
    return _resolve_runtime_services().group_rank_service()


def get_task_registry_service() -> TaskRegistryService:
    """Return process-scoped task-registry service."""
    return _resolve_runtime_services().task_registry_service()


def get_data_fetch_lock() -> DataFetchLock:
    """Return process-scoped distributed lock instance."""
    return _resolve_runtime_services().data_fetch_lock()


def get_groq_key_manager() -> GroqKeyManager:
    """Return process-scoped Groq key manager."""
    return _resolve_runtime_services().groq_key_manager()


def get_zai_key_manager() -> ZAIKeyManager:
    """Return process-scoped Z.AI key manager."""
    return _resolve_runtime_services().zai_key_manager()


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


def get_stock_data_provider() -> StockDataProvider:
    """Return a process-scoped StockDataProvider (wraps DataPreparationLayer)."""
    return _resolve_runtime_services().stock_data_provider()


# ── Orchestrator ────────────────────────────────────────────────────


def get_scan_orchestrator() -> ScanOrchestrator:
    """Return a process-scoped ScanOrchestrator wired with production dependencies."""
    return _resolve_runtime_services().scan_orchestrator()


def _reset_singletons_for_tests() -> None:
    """Reset runtime-scoped singletons for test isolation."""
    runtime = _runtime_services_ctx.get()
    if runtime is not None:
        runtime.reset_for_tests()
    clear_runtime_services()
