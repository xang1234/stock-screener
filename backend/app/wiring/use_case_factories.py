"""Application use-case factories composed from process runtime services."""

from __future__ import annotations

from datetime import timedelta, timezone
from hashlib import sha256
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

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
    import time

    from app.config import settings
    from app.domain.scanning.ports import NeverCancelledToken
    from app.infra.db.repositories.options_retention import (
        SqlOptionsRetentionRepository,
    )
    from app.infra.db.repositories.options_run_writer import SqlOptionsRunWriter
    from app.infra.db.repositories.published_options_reader import (
        SqlPublishedOptionsReader,
    )
    from app.infra.providers.yahoo_options import YahooOptionsProvider
    from app.infra.query.options_candidate_source import SqlOptionsCandidateSource
    from app.services.market_session_lag import MarketSessionWindow
    from app.services.rate_budget_policy import get_rate_budget_policy
    from app.use_cases.options_analytics import (
        OPTIONS_ANALYTICS_CALCULATION_VERSION,
        OPTIONS_ANALYTICS_SCHEMA_VERSION,
        RefreshOptionsAnalyticsUseCase,
    )

    runtime = resolve_runtime_services()
    calendar = MarketSessionWindow(runtime.market_calendar_service(), market="US")
    requests_per_second = max(float(settings.yfinance_rate_limit), 0.01)
    provider = YahooOptionsProvider(
        rate_limiter=lambda: runtime.rate_limiter().wait(
            "yfinance:options", min_interval_s=1.0 / requests_per_second
        ),
    )
    rate_budget_policy = get_rate_budget_policy()
    backoff = rate_budget_policy.get_backoff_params("yfinance", "US")

    def throttle_backoff(attempt: int) -> None:
        wait_seconds = min(
            float(backoff["base_s"]) * float(backoff["factor"]) ** (attempt - 1),
            float(backoff["max_s"]),
        )
        rate_budget_policy.record_429("yfinance", "US")
        rate_budget_policy.record_throttle_wait("yfinance", "US", wait_seconds)
        time.sleep(wait_seconds)

    run_writer = SqlOptionsRunWriter(session)
    published_reader = SqlPublishedOptionsReader(session)
    return RefreshOptionsAnalyticsUseCase(
        candidate_source=SqlOptionsCandidateSource(session),
        run_writer=run_writer,
        published_reader=published_reader,
        retention=SqlOptionsRetentionRepository(session),
        provider=provider,
        calendar=calendar,
        cancellation=cancellation or NeverCancelledToken(),
        calculation_version=OPTIONS_ANALYTICS_CALCULATION_VERSION,
        schema_version=OPTIONS_ANALYTICS_SCHEMA_VERSION,
        max_workers=2,
        throttle_backoff=throttle_backoff,
    )


def get_options_analytics_queries(session: Session):
    from app.infra.db.repositories.published_options_reader import (
        SqlPublishedOptionsReader,
    )
    from app.use_cases.options_analytics import (
        OPTIONS_ANALYTICS_CALCULATION_VERSION,
        OptionsAnalyticsQueries,
    )

    return OptionsAnalyticsQueries(
        SqlPublishedOptionsReader(session),
        calculation_version=OPTIONS_ANALYTICS_CALCULATION_VERSION,
    )


def _social_provider_factories(sessions, *, official_client=None, cooldown_gate=None):
    import httpx
    from app.config import settings
    from app.infra.providers.official_x_social_provider import OfficialXSocialProvider
    from app.infra.providers.xui_cli_social_provider import XuiCliSocialProvider
    from app.services.social_signal_runtime_gate import SharedCooldownSocialProvider
    from app.services.social_source_admin_service import SocialSourceAdminService

    def reserve_official(day, requested, daily_limit):
        with sessions() as db:
            return SocialSourceAdminService(db).reserve_official_capacity(
                day, requested, daily_limit
            )

    factories = {
        "official": lambda: OfficialXSocialProvider(
            bearer_token=settings.twitter_bearer_token,
            reservation=reserve_official,
            client=official_client or httpx.Client(timeout=30),
            daily_post_limit=settings.social_official_daily_post_limit,
            budget_timezone=settings.social_llm_budget_timezone,
        ),
        "xui": lambda: XuiCliSocialProvider(
            config_path=settings.social_xui_config_path,
            profile=settings.social_xui_profile,
        ),
    }
    if cooldown_gate is None or not hasattr(cooldown_gate, "provider_cooldown"):
        return factories
    return {
        name: (lambda name=name, factory=factory: SharedCooldownSocialProvider(
            name, factory(), cooldown_gate
        ))
        for name, factory in factories.items()
    }


def _social_provider_lease():
    from app.services.redis_pool import get_redis_client
    from app.services.social_signal_runtime_gate import RedisSocialSignalGate
    return RedisSocialSignalGate(get_redis_client())


def _social_run_id(origin, now):
    """Use the scheduled cadence slot as the idempotency identity."""
    identity_time = now
    if origin == "scheduled":
        from app.config import settings
        local = now.astimezone(ZoneInfo(settings.celery_timezone))
        eligible = [hour for hour in (0, 6, 12, 18)
                    if (hour, 17) <= (local.hour, local.minute)]
        if eligible:
            slot = local.replace(
                hour=eligible[-1], minute=17, second=0, microsecond=0
            )
        else:
            slot = (local - timedelta(days=1)).replace(
                hour=18, minute=17, second=0, microsecond=0
            )
        identity_time = slot.astimezone(timezone.utc)
    value = f"social:{origin}:{identity_time.isoformat()}".encode()
    return f"social-{sha256(value).hexdigest()[:24]}"


def get_refresh_social_signals_use_case(
    *, session_factory=None, provider_lease=None, official_client=None, llm=None
):
    """Build the private/local Social refresh path with explicit provider routing.

    The database remains authoritative for mode and provider. Provider factories
    are lazy, so an off installation does not touch X credentials or start a CLI.
    """
    from app.config import settings
    from app.database import SessionLocal
    from app.infra.db.repositories.social_refresh_support import (
        SocialScoringEvidenceReader,
        SqlConfirmationReaderFacade,
        SqlSocialRefreshCatalog,
        SqlThemeProjectionFacade,
    )
    from app.infra.db.repositories.social_signal_writer import SocialSignalWriter
    from app.services.social_extraction_service import SocialExtractionService
    from app.use_cases.social_signals.process_backlog import ProcessSocialBacklog
    from app.use_cases.social_signals.refresh import RefreshSocialSignals

    sessions = session_factory or SessionLocal

    if provider_lease is None:
        provider_lease = _social_provider_lease()

    return RefreshSocialSignals(
        catalog=SqlSocialRefreshCatalog(sessions),
        providers=_social_provider_factories(
            sessions, official_client=official_client, cooldown_gate=provider_lease
        ),
        writer=SocialSignalWriter(sessions),
        backlog=ProcessSocialBacklog(sessions, llm=llm),
        evidence_reader=SocialScoringEvidenceReader(sessions),
        theme_service=SqlThemeProjectionFacade(sessions),
        confirmation_reader=SqlConfirmationReaderFacade(
            sessions, grace_minutes=settings.social_market_close_grace_minutes
        ),
        provider_lease=provider_lease,
        run_id_factory=_social_run_id,
        input_hash=lambda post: SocialExtractionService.input_hash((post,)),
        initial_days=settings.social_initial_backfill_days,
        initial_limit=settings.social_initial_backfill_limit_per_source,
        incremental_limit=settings.social_incremental_limit_per_source,
    )


def get_validate_social_source_use_case(
    *, session_factory=None, provider_lease=None, official_client=None
):
    from datetime import datetime, timezone
    from app.database import SessionLocal
    from app.use_cases.social_signals.validate_source import (
        SqlSourceTestRegistry, ValidateSocialSource,
    )
    sessions = session_factory or SessionLocal
    provider_lease = provider_lease or _social_provider_lease()
    return ValidateSocialSource(
        registry=SqlSourceTestRegistry(sessions),
        providers=_social_provider_factories(
            sessions,
            official_client=official_client,
            cooldown_gate=provider_lease,
        ),
        provider_lease=provider_lease,
        clock=lambda: datetime.now(timezone.utc),
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
    "get_refresh_social_signals_use_case",
    "get_validate_social_source_use_case",
    "get_run_bulk_scan_use_case",
]
