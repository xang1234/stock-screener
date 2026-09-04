"""Application use-case factories composed from process runtime services."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy.orm import Session

from app.wiring.runtime_context import resolve_runtime_services

if TYPE_CHECKING:
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


class _NeverCancelled:
    def is_cancelled(self) -> bool:
        return False


class _UsOptionsSessionCalendar:
    def __init__(self, calendar_service: Any) -> None:
        self._calendar_service = calendar_service

    def is_session(self, value: date) -> bool:
        return self._calendar_service.is_trading_day("US", value)

    def sessions_ending_on(self, value: date, count: int) -> tuple[date, ...]:
        start = value - timedelta(days=count * 3 + 30)
        sessions = self._calendar_service.trading_days("US", start, value)
        if len(sessions) < count:
            raise ValueError(
                f"Only {len(sessions)} US sessions available; {count} required"
            )
        return tuple(sessions[-count:])


def get_create_scan_use_case() -> CreateScanUseCase:
    """Build the HTTP scan use case with its mandatory freshness gate."""
    from app.services.market_data_freshness import evaluate_symbol_freshness
    from app.use_cases.scanning.create_scan import CreateScanUseCase

    return CreateScanUseCase(
        dispatcher=resolve_runtime_services().task_dispatcher(),
        freshness_evaluator=evaluate_symbol_freshness,
    )


def get_create_scan_use_case_without_freshness_gate() -> CreateScanUseCase:
    """Build the internal bootstrap scan use case without a freshness gate."""
    from app.use_cases.scanning.create_scan import CreateScanUseCase

    return CreateScanUseCase(
        dispatcher=resolve_runtime_services().task_dispatcher(),
        freshness_evaluator=None,
    )


def get_get_scan_results_use_case() -> GetScanResultsUseCase:
    from app.use_cases.scanning.get_scan_results import GetScanResultsUseCase

    return GetScanResultsUseCase()


def get_get_scan_symbols_use_case() -> GetScanSymbolsUseCase:
    from app.use_cases.scanning.get_scan_symbols import GetScanSymbolsUseCase

    return GetScanSymbolsUseCase()


def get_get_filter_options_use_case() -> GetFilterOptionsUseCase:
    from app.use_cases.scanning.get_filter_options import GetFilterOptionsUseCase

    return GetFilterOptionsUseCase()


def get_get_single_result_use_case() -> GetSingleResultUseCase:
    from app.use_cases.scanning.get_single_result import GetSingleResultUseCase

    return GetSingleResultUseCase()


def get_get_setup_details_use_case() -> GetSetupDetailsUseCase:
    from app.use_cases.scanning.get_setup_details import GetSetupDetailsUseCase

    return GetSetupDetailsUseCase()


def get_get_peers_use_case() -> GetPeersUseCase:
    from app.use_cases.scanning.get_peers import GetPeersUseCase

    return GetPeersUseCase()


def get_export_scan_results_use_case() -> ExportScanResultsUseCase:
    from app.use_cases.scanning.export_scan_results import ExportScanResultsUseCase

    return ExportScanResultsUseCase()


def get_run_bulk_scan_use_case() -> RunBulkScanUseCase:
    from app.use_cases.scanning.run_bulk_scan import RunBulkScanUseCase

    runtime = resolve_runtime_services()

    return RunBulkScanUseCase(
        scanner=runtime.scan_orchestrator(),
        data_provider=runtime.stock_data_provider(),
        market_rs_reader=runtime.market_rs_reader(),
    )


def get_explain_stock_use_case() -> ExplainStockUseCase:
    from app.use_cases.scanning.explain_stock import ExplainStockUseCase

    return ExplainStockUseCase()


def get_list_feature_runs_use_case() -> ListFeatureRunsUseCase:
    from app.use_cases.feature_store.list_runs import ListFeatureRunsUseCase

    return ListFeatureRunsUseCase()


def get_compare_feature_runs_use_case() -> CompareFeatureRunsUseCase:
    from app.use_cases.feature_store.compare_runs import CompareFeatureRunsUseCase

    return CompareFeatureRunsUseCase()


def get_build_daily_snapshot_use_case() -> BuildDailyFeatureSnapshotUseCase:
    from app.services.bootstrap_cache_coverage import (
        evaluate_bootstrap_cache_coverage,
    )
    from app.use_cases.feature_store.build_daily_snapshot import (
        BuildDailyFeatureSnapshotUseCase,
    )

    runtime = resolve_runtime_services()

    return BuildDailyFeatureSnapshotUseCase(
        scanner=runtime.scan_orchestrator(),
        data_provider=runtime.stock_data_provider(),
        market_calendar=runtime.market_calendar_service(),
        market_rs_reader=runtime.market_rs_reader(),
        bootstrap_coverage_evaluator=evaluate_bootstrap_cache_coverage,
    )


def get_refresh_options_analytics_use_case(
    session: Session,
    *,
    cancellation: Any | None = None,
):
    from app.config import settings
    from app.infra.db.repositories.options_analytics_repo import (
        SqlOptionsAnalyticsRepository,
    )
    from app.infra.providers.yahoo_options import YahooOptionsProvider
    from app.infra.query.options_candidate_source import SqlOptionsCandidateSource
    from app.use_cases.options_analytics import (
        OPTIONS_ANALYTICS_CALCULATION_VERSION,
        OPTIONS_ANALYTICS_SCHEMA_VERSION,
        RefreshOptionsAnalyticsUseCase,
    )

    runtime = resolve_runtime_services()
    calendar = _UsOptionsSessionCalendar(runtime.market_calendar_service())
    requests_per_second = max(float(settings.yfinance_rate_limit), 0.01)
    provider = YahooOptionsProvider(
        rate_limiter=lambda: runtime.rate_limiter().wait(
            "yfinance:options", min_interval_s=1.0 / requests_per_second
        ),
        # The use case owns the three-attempt symbol budget.
        max_attempts=1,
    )
    return RefreshOptionsAnalyticsUseCase(
        candidate_source=SqlOptionsCandidateSource(session),
        repository=SqlOptionsAnalyticsRepository(session),
        provider=provider,
        calendar=calendar,
        clock=lambda: datetime.now(timezone.utc),
        cancellation=cancellation or _NeverCancelled(),
        calculation_version=OPTIONS_ANALYTICS_CALCULATION_VERSION,
        schema_version=OPTIONS_ANALYTICS_SCHEMA_VERSION,
        max_workers=2,
    )


def get_options_analytics_queries(session: Session):
    from app.infra.db.repositories.options_analytics_repo import (
        SqlOptionsAnalyticsRepository,
    )
    from app.use_cases.options_analytics import (
        OPTIONS_ANALYTICS_CALCULATION_VERSION,
        OptionsAnalyticsQueries,
    )

    return OptionsAnalyticsQueries(
        SqlOptionsAnalyticsRepository(session),
        calculation_version=OPTIONS_ANALYTICS_CALCULATION_VERSION,
    )


__all__ = [
    "get_build_daily_snapshot_use_case",
    "get_compare_feature_runs_use_case",
    "get_create_scan_use_case",
    "get_create_scan_use_case_without_freshness_gate",
    "get_explain_stock_use_case",
    "get_export_scan_results_use_case",
    "get_get_filter_options_use_case",
    "get_get_peers_use_case",
    "get_get_scan_results_use_case",
    "get_get_scan_symbols_use_case",
    "get_get_setup_details_use_case",
    "get_get_single_result_use_case",
    "get_list_feature_runs_use_case",
    "get_options_analytics_queries",
    "get_refresh_options_analytics_use_case",
    "get_run_bulk_scan_use_case",
]
