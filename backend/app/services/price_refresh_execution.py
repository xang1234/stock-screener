"""Live execution helpers for planned price refresh jobs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import logging
from typing import Any, Protocol

from celery.exceptions import SoftTimeLimitExceeded

from .price_refresh_planning import PriceRefreshJob


logger = logging.getLogger(__name__)

MappingResult = Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class PriceRefreshExecutionResult:
    refreshed: int
    failed: int
    failed_symbols: list[str]
    refreshed_by_market: Counter[str] = field(default_factory=Counter)
    failed_by_market: Counter[str] = field(default_factory=Counter)


class PriceRefreshExecutionContext(Protocol):
    def fetch_batch(
        self,
        symbols: Sequence[str],
        *,
        period: str,
        market: str | None,
    ) -> MappingResult:
        ...

    def store_prices(self, price_data_by_symbol: Mapping[str, Any]) -> None:
        ...

    def track_symbol_failures(
        self,
        successes: Sequence[str],
        failures: Sequence[str],
        *,
        failure_details: Mapping[str, str],
    ) -> None:
        ...

    def market_for_symbol(self, symbol: str) -> str:
        ...

    def publish_progress(
        self,
        current: int,
        total: int,
        percent: float,
        message: str,
        *,
        refreshed: int,
        failed: int,
    ) -> None:
        ...

    def extend_lock(self) -> None:
        ...

    def wait_between_batches(self) -> None:
        ...

    def raise_if_transient_database_error(self, exc: Exception) -> None:
        ...


def _total_batches(jobs: Sequence[PriceRefreshJob], batch_size: int) -> int:
    return sum(
        (len(job.symbols) + batch_size - 1) // batch_size
        for job in jobs
    )


def _result_for_symbol(
    batch_results: MappingResult,
    symbol: str,
) -> Mapping[str, Any] | None:
    if symbol in batch_results:
        return batch_results[symbol]
    normalized_symbol = str(symbol).upper()
    for result_symbol, result in batch_results.items():
        if str(result_symbol).upper() == normalized_symbol:
            return result
    return None


def _record_failure(
    *,
    symbol: str,
    reason: str,
    failed_symbols: list[str],
    failed_by_market: Counter[str],
    batch_failures: list[str],
    failure_details: dict[str, str],
    context: PriceRefreshExecutionContext,
) -> None:
    failed_symbols.append(symbol)
    failed_by_market[context.market_for_symbol(symbol)] += 1
    batch_failures.append(symbol)
    failure_details[symbol] = reason


def run_price_refresh_jobs(
    *,
    jobs: Sequence[PriceRefreshJob],
    total: int,
    batch_size: int,
    market: str | None,
    context: PriceRefreshExecutionContext,
) -> PriceRefreshExecutionResult:
    total_batches = _total_batches(jobs, batch_size)
    refreshed = 0
    failed = 0
    failed_symbols: list[str] = []
    refreshed_by_market: Counter[str] = Counter()
    failed_by_market: Counter[str] = Counter()
    processed_count = 0
    batch_num = 0

    for job in jobs:
        job_symbols = list(job.symbols)
        for batch_start in range(0, len(job_symbols), batch_size):
            batch_symbols = job_symbols[batch_start:batch_start + batch_size]
            batch_num += 1

            logger.info(
                "Batch %d/%d: Fetching %d symbols (%s, period=%s)",
                batch_num,
                total_batches,
                len(batch_symbols),
                job.kind,
                job.period,
            )

            batch_successes: list[str] = []
            batch_failures: list[str] = []
            failure_details: dict[str, str] = {}

            try:
                batch_results = context.fetch_batch(
                    batch_symbols,
                    period=job.period,
                    market=market,
                )
                batch_to_store = {}
                for symbol in batch_symbols:
                    data = _result_for_symbol(batch_results, symbol)
                    if data is None:
                        failed += 1
                        _record_failure(
                            symbol=symbol,
                            reason="No data returned",
                            failed_symbols=failed_symbols,
                            failed_by_market=failed_by_market,
                            batch_failures=batch_failures,
                            failure_details=failure_details,
                            context=context,
                        )
                        continue
                    if not data.get("has_error") and data.get("price_data") is not None:
                        price_df = data["price_data"]
                        if not price_df.empty:
                            batch_to_store[symbol] = price_df
                            refreshed += 1
                            refreshed_by_market[context.market_for_symbol(symbol)] += 1
                            batch_successes.append(symbol)
                        else:
                            failed += 1
                            _record_failure(
                                symbol=symbol,
                                reason="Empty data returned",
                                failed_symbols=failed_symbols,
                                failed_by_market=failed_by_market,
                                batch_failures=batch_failures,
                                failure_details=failure_details,
                                context=context,
                            )
                    else:
                        failed += 1
                        _record_failure(
                            symbol=symbol,
                            reason=str(data.get("error", "Unknown error")),
                            failed_symbols=failed_symbols,
                            failed_by_market=failed_by_market,
                            batch_failures=batch_failures,
                            failure_details=failure_details,
                            context=context,
                        )

                if batch_to_store:
                    context.store_prices(batch_to_store)

            except SoftTimeLimitExceeded:
                raise
            except Exception as exc:
                context.raise_if_transient_database_error(exc)
                logger.error("Batch %d error: %s", batch_num, exc)
                failed += len(batch_symbols)
                failed_symbols.extend(batch_symbols)
                failed_by_market.update(context.market_for_symbol(symbol) for symbol in batch_symbols)
                batch_failures.extend(batch_symbols)
                failure_details.update({symbol: str(exc) for symbol in batch_symbols})

            context.track_symbol_failures(
                batch_successes,
                batch_failures,
                failure_details=failure_details,
            )

            processed_count += len(batch_symbols)
            progress = min(processed_count, total)
            percent = (progress / total) * 100 if total else 100.0
            context.publish_progress(
                progress,
                total,
                percent,
                f"Batch {batch_num}/{total_batches} · refreshing prices",
                refreshed=refreshed,
                failed=failed,
            )
            context.extend_lock()

            if processed_count < total:
                context.wait_between_batches()

    return PriceRefreshExecutionResult(
        refreshed=refreshed,
        failed=failed,
        failed_symbols=failed_symbols,
        refreshed_by_market=refreshed_by_market,
        failed_by_market=failed_by_market,
    )
