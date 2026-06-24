"""Application workflow for smart market price refreshes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any

from celery.exceptions import SoftTimeLimitExceeded

from ..config import settings
from ..services.price_refresh_accounting import account_live_refresh
from ..services.price_refresh_actions import (
    PriceRefreshTerminalCompletion,
    build_terminal_completion,
)
from ..services.price_refresh_activity import (
    CeleryTaskLike,
    PriceRefreshActivityReporter,
    PriceRefreshOutcome,
)
from ..services.price_refresh_live_runner import (
    LivePriceRefreshRunner,
    PriceRefreshExecutionError,
    PriceRefreshRetryScheduler,
)
from ..services.price_refresh_execution import PriceRefreshExecutionSummary
from ..services.price_refresh_planning import (
    GitHubSeedOutcome,
    PriceRefreshMode,
    PriceRefreshPlan,
    PriceRefreshSource,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PriceRefreshMarketGateway:
    normalize_market: Callable[[str], str]
    market_tag: Callable[[str | None], str]
    log_extra: Callable[[str | None], Mapping[str, Any]]
    get_eastern_now: Callable[[], Any]
    is_trading_day: Callable[[Any], bool]
    format_market_status: Callable[[], str]
    is_market_enabled_now: Callable[[str], bool]


@dataclass(frozen=True)
class PriceRefreshWorkflowDependencies:
    session_factory: Callable[[], Any]
    price_cache_factory: Callable[[], Any]
    bulk_fetcher_factory: Callable[[], Any]
    warm_benchmarks: Callable[..., Mapping[str, Any]]
    build_refresh_plan: Callable[..., PriceRefreshPlan]
    last_completed_trading_day: Callable[[str], Any]
    activity_reporter: PriceRefreshActivityReporter
    live_runner: LivePriceRefreshRunner
    retry_scheduler: PriceRefreshRetryScheduler
    market_gateway: PriceRefreshMarketGateway
    raise_if_transient_database_error: Callable[[Exception], None]
    safe_rollback: Callable[[Any], None]
    time_window_bypass_enabled: Callable[[], bool] = lambda: False


class PriceRefreshWorkflow:
    def __init__(self, dependencies: PriceRefreshWorkflowDependencies) -> None:
        self._deps = dependencies

    def run(
        self,
        *,
        task: CeleryTaskLike,
        mode: PriceRefreshMode | str = PriceRefreshMode.AUTO,
        market: str | None = None,
        activity_lifecycle: str | None = None,
    ) -> dict[str, Any]:
        parsed_mode = PriceRefreshMode.parse(mode)
        gateway = self._deps.market_gateway
        effective_market = (
            gateway.normalize_market(market) if market is not None else "US"
        )
        activity_lifecycle = activity_lifecycle or "daily_refresh"
        log_extra = gateway.log_extra(market)

        logger.info("=" * 80)
        logger.info(
            "TASK: Smart Cache Refresh %s (mode=%s)",
            gateway.market_tag(market),
            parsed_mode.value,
            extra=log_extra,
        )
        logger.info("Market status: %s", gateway.format_market_status(), extra=log_extra)
        logger.info("Timestamp: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), extra=log_extra)
        logger.info("=" * 80)

        if market is not None and not gateway.is_market_enabled_now(
            gateway.normalize_market(market)
        ):
            logger.info("Skipping smart refresh for disabled market %s", market, extra=log_extra)
            return {
                "status": "skipped",
                "reason": f"market {effective_market} is disabled in local runtime preferences",
                "market": effective_market,
                "mode": parsed_mode.value,
                "timestamp": datetime.now().isoformat(),
            }

        if self._should_reject_full_refresh(parsed_mode, task, market):
            now_et = gateway.get_eastern_now()
            return {
                "skipped": True,
                "reason": f"Outside refresh window (weekday={now_et.weekday()}, hour={now_et.hour})",
                "mode": parsed_mode.value,
                "timestamp": datetime.now().isoformat(),
            }

        if parsed_mode is PriceRefreshMode.AUTO:
            today = gateway.get_eastern_now().date()
            if not gateway.is_trading_day(today):
                logger.info("Skipping smart refresh (auto) - %s is not a trading day", today)
                return {
                    "skipped": True,
                    "reason": "Not a trading day",
                    "date": today.isoformat(),
                    "mode": parsed_mode.value,
                }

        price_cache = self._deps.price_cache_factory()
        db = self._deps.session_factory()

        try:
            self._deps.activity_reporter.start_prices(
                db,
                task=task,
                market=effective_market,
                lifecycle=activity_lifecycle,
                message="Refreshing market prices",
            )
            refresh_plan = self._prepare_refresh(
                db,
                price_cache,
                task=task,
                mode=parsed_mode,
                market=market,
                effective_market=effective_market,
                activity_lifecycle=activity_lifecycle,
                log_extra=log_extra,
            )
            terminal_completion = self._build_terminal_completion(
                mode=parsed_mode,
                effective_market=effective_market,
                refresh_plan=refresh_plan,
            )
            if terminal_completion is not None:
                return self._complete_terminal_refresh(
                    db,
                    price_cache,
                    task=task,
                    market=market,
                    effective_market=effective_market,
                    activity_lifecycle=activity_lifecycle,
                    completion=terminal_completion,
                )

            self._warm_benchmarks(market=market)
            outcome = self._execute_live_refresh(
                db,
                price_cache,
                task=task,
                mode=parsed_mode,
                market=market,
                effective_market=effective_market,
                activity_lifecycle=activity_lifecycle,
                refresh_plan=refresh_plan,
            )
            return outcome.to_task_result()

        except PriceRefreshExecutionError as exc:
            cause = exc.cause
            self._deps.safe_rollback(db)
            if isinstance(cause, SoftTimeLimitExceeded):
                logger.error("Soft time limit exceeded in smart_refresh_cache", exc_info=True)
                self._record_refresh_failure(
                    db,
                    price_cache,
                    task=task,
                    market=market,
                    effective_market=effective_market,
                    activity_lifecycle=activity_lifecycle,
                    summary=exc.summary,
                    message="Soft time limit exceeded",
                )
                raise cause

            self._deps.raise_if_transient_database_error(cause)
            logger.error("Error in smart_refresh_cache task: %s", cause, exc_info=True)
            self._record_refresh_failure(
                db,
                price_cache,
                task=task,
                market=market,
                effective_market=effective_market,
                activity_lifecycle=activity_lifecycle,
                summary=exc.summary,
                message=str(cause),
            )
            return self._failure_task_result(parsed_mode, exc.summary, str(cause))

        except SoftTimeLimitExceeded:
            logger.error("Soft time limit exceeded in smart_refresh_cache", exc_info=True)
            self._deps.safe_rollback(db)
            self._record_refresh_failure(
                db,
                price_cache,
                task=task,
                market=market,
                effective_market=effective_market,
                activity_lifecycle=activity_lifecycle,
                summary=PriceRefreshExecutionSummary.empty(),
                message="Soft time limit exceeded",
            )
            raise
        except Exception as exc:
            self._deps.safe_rollback(db)
            self._deps.raise_if_transient_database_error(exc)
            logger.error("Error in smart_refresh_cache task: %s", exc, exc_info=True)
            summary = PriceRefreshExecutionSummary.empty()
            self._record_refresh_failure(
                db,
                price_cache,
                task=task,
                market=market,
                effective_market=effective_market,
                activity_lifecycle=activity_lifecycle,
                summary=summary,
                message=str(exc),
            )
            return self._failure_task_result(parsed_mode, summary, str(exc))
        finally:
            db.close()

    def _record_refresh_failure(
        self,
        db,
        price_cache,
        *,
        task: CeleryTaskLike,
        market: str | None,
        effective_market: str,
        activity_lifecycle: str,
        summary: PriceRefreshExecutionSummary,
        message: str,
    ) -> None:
        self._deps.activity_reporter.record_failure(
            db,
            price_cache,
            task=task,
            market=market,
            effective_market=effective_market,
            lifecycle=activity_lifecycle,
            refreshed=summary.refreshed,
            total=summary.total,
            current=summary.processed,
            message=message,
        )

    def _failure_task_result(
        self,
        mode: PriceRefreshMode,
        summary: PriceRefreshExecutionSummary,
        error: str,
    ) -> dict[str, Any]:
        return {
            "status": "failed",
            "error": error,
            "refreshed": summary.refreshed,
            "failed": summary.failed,
            "mode": mode.value,
            "timestamp": datetime.now().isoformat(),
        }

    def _prepare_refresh(
        self,
        db,
        price_cache,
        *,
        task: CeleryTaskLike,
        mode: PriceRefreshMode,
        market: str | None,
        effective_market: str,
        activity_lifecycle: str,
        log_extra: Mapping[str, Any],
    ) -> PriceRefreshPlan:
        gateway = self._deps.market_gateway
        def symbols_needing_auto_refresh(candidate_symbols: Sequence[str]) -> Sequence[str]:
            logger.info(
                "Auto refresh: %d active symbols (full universe, market cap order) %s",
                len(candidate_symbols),
                gateway.market_tag(market),
                extra=log_extra,
            )
            refresh_symbols = price_cache.get_symbols_needing_refresh(
                list(candidate_symbols),
                max_age_hours=settings.refresh_skip_hours,
            )
            skipped = len(candidate_symbols) - len(refresh_symbols)
            if skipped > 0:
                logger.info(
                    "Skipping %d recently-refreshed symbols (fresh within %sh)",
                    skipped,
                    settings.refresh_skip_hours,
            )
            return refresh_symbols

        logger.info("[1/2] Determining symbols to refresh (mode=%s)...", mode.value)
        refresh_plan = self._deps.build_refresh_plan(
            db,
            mode=mode,
            market=market,
            effective_market=effective_market,
            recently_refreshed_filter=(
                symbols_needing_auto_refresh
                if mode is PriceRefreshMode.AUTO
                else None
            ),
        )
        self._publish_github_seed_log(
            github_seed=refresh_plan.github_seed,
            refresh_plan=refresh_plan,
            effective_market=effective_market,
            all_symbols=refresh_plan.all_symbols,
            activity_lifecycle=activity_lifecycle,
            db=db,
            task=task,
            log_extra=log_extra,
        )
        self._log_live_symbol_plan(
            refresh_plan=refresh_plan,
            refresh_source=refresh_plan.source,
            symbols=refresh_plan.symbols,
            mode=mode,
            market=market,
            effective_market=effective_market,
            log_extra=log_extra,
        )
        return refresh_plan

    def _warm_benchmarks(self, *, market: str | None) -> None:
        logger.info("Warming market benchmarks before live price fetch...")
        benchmark_result = self._deps.warm_benchmarks(market=market)
        if benchmark_result.get("error"):
            logger.error("Benchmark warmup failed: %s", benchmark_result.get("error"))

    def _build_terminal_completion(
        self,
        *,
        mode: PriceRefreshMode,
        effective_market: str,
        refresh_plan: PriceRefreshPlan,
    ) -> PriceRefreshTerminalCompletion | None:
        return build_terminal_completion(
            mode=mode,
            effective_market=effective_market,
            plan=refresh_plan,
            last_completed_trading_day=self._deps.last_completed_trading_day,
        )

    def _complete_terminal_refresh(
        self,
        db,
        price_cache,
        *,
        task: CeleryTaskLike,
        market: str | None,
        effective_market: str,
        activity_lifecycle: str,
        completion: PriceRefreshTerminalCompletion,
    ) -> dict[str, Any]:
        self._deps.activity_reporter.finalize_success(
            db,
            price_cache,
            task=task,
            market=market,
            effective_market=effective_market,
            lifecycle=activity_lifecycle,
            finalization=completion.finalization,
        )
        return completion.outcome.to_task_result()

    def _execute_live_refresh(
        self,
        db,
        price_cache,
        *,
        task: CeleryTaskLike,
        mode: PriceRefreshMode,
        market: str | None,
        effective_market: str,
        activity_lifecycle: str,
        refresh_plan: PriceRefreshPlan,
    ) -> PriceRefreshOutcome:
        total = len(refresh_plan.symbols)
        bulk_fetcher = self._deps.bulk_fetcher_factory()

        self._deps.activity_reporter.publish_progress(
            db,
            price_cache,
            task=task,
            market=market,
            effective_market=effective_market,
            lifecycle=activity_lifecycle,
            current=0,
            total=total,
            percent=0,
            message="Refreshing market prices",
            refreshed=0,
            failed=0,
        )

        logger.info("[2/2] Fetching %d symbols...", total)
        execution_result = self._deps.live_runner.run(
            task=task,
            bulk_fetcher=bulk_fetcher,
            price_cache=price_cache,
            db=db,
            jobs=refresh_plan.live_refresh_jobs,
            total=total,
            batch_size=None,
            market=market,
            effective_market=effective_market,
            activity_lifecycle=activity_lifecycle,
            symbol_markets=refresh_plan.symbol_markets,
            activity_reporter=self._deps.activity_reporter,
        )

        self._deps.retry_scheduler.schedule(
            execution_result.failed_symbols,
            failure_kinds=execution_result.failure_kinds,
            effective_market=effective_market,
            symbol_markets=refresh_plan.symbol_markets,
            activity_lifecycle=activity_lifecycle,
        )
        accounting = account_live_refresh(
            refresh_plan,
            execution_result,
            effective_market=effective_market,
            last_completed_trading_day=self._deps.last_completed_trading_day,
        )

        logger.info("=" * 80)
        logger.info("Smart refresh completed (%s mode):", mode.value)
        logger.info("  Refreshed: %s", accounting.refreshed)
        logger.info("  Failed: %s", accounting.failed)
        logger.info("  Total: %s", accounting.total)
        if accounting.live_top_up_total is not None:
            logger.info(
                "  Live top-up: %s refreshed, %s failed, %s total",
                accounting.live_top_up_refreshed,
                accounting.live_top_up_failed,
                accounting.live_top_up_total,
            )
        if execution_result.failed_symbols:
            logger.info("  Failed symbols: %s...", execution_result.failed_symbols[:10])
        logger.info("=" * 80)

        self._deps.activity_reporter.finalize_success(
            db,
            price_cache,
            task=task,
            market=market,
            effective_market=effective_market,
            lifecycle=activity_lifecycle,
            finalization=accounting.to_finalization(),
        )

        return accounting.to_outcome(mode=mode)

    def _should_reject_full_refresh(
        self,
        mode: PriceRefreshMode,
        task: CeleryTaskLike,
        market: str | None,
    ) -> bool:
        if mode is not PriceRefreshMode.FULL or market is not None:
            return False
        is_manual = (
            self._deps.time_window_bypass_enabled()
            or (
                getattr(getattr(task, "request", None), "headers", None)
                and task.request.headers.get("origin") == "manual"
            )
        )
        if is_manual:
            return False
        now_et = self._deps.market_gateway.get_eastern_now()
        weekday = now_et.weekday()
        hour = now_et.hour
        in_weekday_window = weekday < 5 and 16 <= hour < 24
        in_sunday_window = weekday == 6 and 1 <= hour < 6
        if in_weekday_window or in_sunday_window:
            return False
        logger.warning(
            "Rejecting Beat-scheduled full refresh outside time window "
            "(weekday=%s, hour=%s). Likely a catchup storm.",
            weekday,
            hour,
        )
        return True

    def _publish_github_seed_log(
        self,
        *,
        github_seed: GitHubSeedOutcome | None,
        refresh_plan: PriceRefreshPlan,
        effective_market: str,
        all_symbols: Sequence[str],
        activity_lifecycle: str,
        db,
        task: CeleryTaskLike,
        log_extra: Mapping[str, Any],
    ) -> None:
        if github_seed and github_seed.stale_reason:
            logger.info(
                "GitHub daily price bundle for %s imported with stale manifest: %s",
                effective_market,
                github_seed.stale_reason,
                extra=log_extra,
            )
        if github_seed and not refresh_plan.used_github_seed:
            reason = github_seed.reason or github_seed.error
            logger.warning(
                "GitHub daily price bundle not used for %s (status=%s, reason=%s, stale_reason=%s); "
                "using live refresh policy",
                effective_market,
                github_seed.status_value,
                reason,
                github_seed.stale_reason,
                extra=log_extra,
            )
            self._deps.activity_reporter.publish_github_seed_fallback(
                db,
                task=task,
                market=effective_market,
                lifecycle=activity_lifecycle,
                total=len(all_symbols),
                status_value=github_seed.status_value,
            )

    def _log_live_symbol_plan(
        self,
        *,
        refresh_plan: PriceRefreshPlan,
        refresh_source: PriceRefreshSource,
        symbols: Sequence[str],
        mode: PriceRefreshMode,
        market: str | None,
        effective_market: str,
        log_extra: Mapping[str, Any],
    ) -> None:
        if not symbols:
            return
        if refresh_source is PriceRefreshSource.GITHUB_AND_LIVE:
            logger.info(
                "GitHub daily price bundle synced for %s; live refresh will top up %d symbols",
                effective_market,
                len(symbols),
                extra=log_extra,
            )
        elif mode is PriceRefreshMode.FULL:
            logger.info(
                "Full refresh: %d symbols (market cap order) %s",
                len(symbols),
                self._deps.market_gateway.market_tag(market),
                extra=log_extra,
            )
        elif mode in {PriceRefreshMode.BOOTSTRAP, PriceRefreshMode.DELTA}:
            logger.info(
                "Delta refresh: %d symbols %s",
                len(symbols),
                self._deps.market_gateway.market_tag(market),
                extra=log_extra,
            )
