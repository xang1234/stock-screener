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
    PriceRefreshExecutionAccumulator,
    PriceRefreshExecutionSummary,
    iter_price_refresh_batches,
)
from .price_refresh_planning import PriceRefreshJob


@dataclass(frozen=True)
class LivePriceRefreshRunnerDependencies:
    fetch_with_backoff: Callable[..., Mapping[str, Mapping[str, Any]]]
    track_symbol_failures: Callable[..., None]
    data_fetch_lock_factory: Callable[[], Any]
    raise_if_transient_database_error: Callable[[Exception], None]


class PriceRefreshExecutionError(Exception):
    def __init__(
        self,
        summary: PriceRefreshExecutionSummary,
        cause: Exception,
    ) -> None:
        super().__init__(str(cause))
        self.summary = summary
        self.cause = cause

    def __str__(self) -> str:
        return str(self.cause)


class LivePriceRefreshRunner:
    def __init__(self, dependencies: LivePriceRefreshRunnerDependencies) -> None:
        self._deps = dependencies

    def run(
        self,
        *,
        task: CeleryTaskLike,
        bulk_fetcher: Any,
        price_cache: Any,
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
        accumulator = PriceRefreshExecutionAccumulator(total=total)

        def market_for_symbol(symbol: str) -> str:
            return symbol_markets.get(str(symbol).upper(), effective_market)

        processed = 0

        def progress_callback(batch_processed: int) -> None:
            nonlocal processed

            processed += batch_processed

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
                refreshed=processed,  # adjust if you want true refreshed count
                failed=0,
            )

        def fetch_batch(
            symbols: Sequence[str],
            *,
            period: str,
            market: str | None,
            progress_callback: Callable[[int], None] | None = None,
        ):
            return self._deps.fetch_with_backoff(
                bulk_fetcher,
                list(symbols),
                period=period,
                market=market,
                progress_callback=progress_callback,
            )

        try:
            for batch in iter_price_refresh_batches(
                jobs=jobs,
                batch_size=batch_size,
                market=market,
                fetch_batch=fetch_batch,
                market_for_symbol=market_for_symbol,
                raise_if_transient_database_error=self._deps.raise_if_transient_database_error,
                progress_callback=progress_callback,
            ):
                if batch.price_data_by_symbol:
                    price_cache.store_batch_in_cache(
                        dict(batch.price_data_by_symbol),
                        also_store_db=True,
                    )
                self._deps.track_symbol_failures(
                    price_cache,
                    list(batch.successes),
                    list(batch.failures),
                    db,
                    failure_details=dict(batch.failure_details),
                )

                accumulator.add(batch)
                summary = accumulator.summary()
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
        except Exception as exc:
            raise PriceRefreshExecutionError(accumulator.summary(), exc) from exc

        return accumulator.summary()

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
