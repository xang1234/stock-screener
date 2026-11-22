"""Side-effect runner for live price refresh batches."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .price_fetch_failures import is_retryable_price_failure_kind
from .price_refresh_activity import (
    CeleryTaskLike,
    PriceRefreshActivityReporter,
    task_id,
)
from .price_refresh_execution import (
    PriceBatchCache,
    PriceBatchFetcher,
    PriceRefreshBatchExecutionError,
    PriceRefreshBatchExecutor,
    PriceRefreshBatchOutcome,
    PriceRefreshExecutionSummary,
    SymbolFailureTracker,
)
from .price_refresh_planning import PriceRefreshJob


@dataclass(frozen=True)
class LivePriceRefreshRunnerDependencies:
    fetch_with_backoff: PriceBatchFetcher
    track_symbol_failures: SymbolFailureTracker
    data_fetch_lock_factory: Callable[[], Any]
    raise_if_transient_database_error: Callable[[Exception], None]


PriceRefreshExecutionError = PriceRefreshBatchExecutionError


class LivePriceRefreshRunner:
    def __init__(self, dependencies: LivePriceRefreshRunnerDependencies) -> None:
        self._deps = dependencies

    def run(
        self,
        *,
        task: CeleryTaskLike,
        bulk_fetcher: Any,
        price_cache: PriceBatchCache,
        db: Any,
        jobs: Sequence[PriceRefreshJob],
        total: int,
        batch_size: int | None,
        market: str | None,
        effective_market: str,
        activity_lifecycle: str,
        symbol_markets: Mapping[str, str],
        activity_reporter: PriceRefreshActivityReporter,
    ) -> PriceRefreshExecutionSummary:
        def market_for_symbol(symbol: str) -> str:
            return symbol_markets.get(str(symbol).upper(), effective_market)

        current_batch = 0
        # Mid-batch progress cannot tell successes from failures yet, so it
        # reports the totals of the batches that have finished.
        completed = PriceRefreshExecutionSummary.empty(total=total)

        def progress_callback(batch_processed: int) -> None:
            nonlocal current_batch

            # Yahoo leaves skipped symbols out of these callbacks, but the
            # finished-batch totals count them, so build on those totals.
            current_batch += batch_processed
            processed = completed.processed + current_batch

            percent = (processed / total) * 100 if total else 100.0

            activity_reporter.publish_progress(
                db,
                price_cache,
                task=task,
                market=market,
                effective_market=effective_market,
                lifecycle=activity_lifecycle,
                current=processed,
                total=total,
                percent=percent,
                message="Refreshing market prices",
                refreshed=completed.refreshed,
                failed=completed.failed,
            )

        def after_batch(
            batch: PriceRefreshBatchOutcome,
            summary: PriceRefreshExecutionSummary,
        ) -> None:
            nonlocal completed, current_batch

            completed = summary
            current_batch = 0
            percent = (summary.processed / total) * 100 if total else 100.0
            activity_reporter.publish_progress(
                db,
                price_cache,
                task=task,
                market=market,
                effective_market=effective_market,
                lifecycle=activity_lifecycle,
                current=summary.processed,
                total=total,
                percent=percent,
                message=f"Batch {batch.batch_number}/{batch.total_batches} · refreshing prices",
                refreshed=summary.refreshed,
                failed=summary.failed,
            )
            self._extend_lock(task, market=market)

        return PriceRefreshBatchExecutor(
            fetch_with_backoff=self._deps.fetch_with_backoff,
            track_symbol_failures=self._deps.track_symbol_failures,
            raise_if_transient_database_error=(
                self._deps.raise_if_transient_database_error
            ),
        ).run(
            bulk_fetcher=bulk_fetcher,
            price_cache=price_cache,
            db=db,
            jobs=jobs,
            total=total,
            batch_size=batch_size,
            market=market,
            market_for_symbol=market_for_symbol,
            progress_callback=progress_callback,
            after_batch=after_batch,
        )

    def _extend_lock(self, task: CeleryTaskLike, *, market: str | None) -> None:
        self._deps.data_fetch_lock_factory().extend_lock(
            task_id(task) or "unknown",
            300,
            market=market,
        )


@dataclass(frozen=True)
class PriceRefreshRetryScheduler:
    schedule_failed_symbol_retry: Callable[..., None]

    def schedule(
        self,
        failed_symbols: Sequence[str],
        *,
        failure_kinds: Mapping[str, str] | None = None,
        effective_market: str,
        symbol_markets: Mapping[str, str],
        activity_lifecycle: str,
    ) -> None:
        if not failed_symbols:
            return
        failure_kinds = failure_kinds or {}
        failed_symbols_by_market: dict[str, list[str]] = {}
        for symbol in failed_symbols:
            if not is_retryable_price_failure_kind(failure_kinds.get(symbol)):
                continue
            failed_symbols_by_market.setdefault(
                symbol_markets.get(str(symbol).upper(), effective_market),
                [],
            ).append(symbol)
        for retry_market, retry_symbols in failed_symbols_by_market.items():
            kwargs = {
                "symbols": retry_symbols,
                "market": retry_market,
                "attempt": 1,
            }
            if activity_lifecycle == "bootstrap":
                kwargs["countdown"] = 30
            self.schedule_failed_symbol_retry(**kwargs)
