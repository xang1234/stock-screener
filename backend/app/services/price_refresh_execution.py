"""Live execution helpers for planned price refresh jobs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import logging
from typing import Any

from celery.exceptions import SoftTimeLimitExceeded

from .price_refresh_planning import PriceRefreshJob


logger = logging.getLogger(__name__)

MappingResult = dict[str, dict[str, Any]]


@dataclass(frozen=True)
class PriceRefreshExecutionResult:
    refreshed: int
    failed: int
    failed_symbols: list[str]
    refreshed_by_market: Counter[str] = field(default_factory=Counter)
    failed_by_market: Counter[str] = field(default_factory=Counter)


def _total_batches(jobs: Sequence[PriceRefreshJob], batch_size: int) -> int:
    return sum(
        (len(job.symbols) + batch_size - 1) // batch_size
        for job in jobs
    )


def run_price_refresh_jobs(
    *,
    jobs: Sequence[PriceRefreshJob],
    total: int,
    batch_size: int,
    bulk_fetcher,
    price_cache,
    db,
    market: str | None,
    fetch_with_backoff: Callable[..., MappingResult],
    track_symbol_failures: Callable[..., None],
    market_for_symbol: Callable[[str], str],
    mark_progress: Callable[[int, int, float, str], None],
    update_task_state: Callable[[int, int, float, int, int], None],
    extend_lock: Callable[[], None],
    wait_between_batches: Callable[[], None],
    raise_if_transient_database_error: Callable[[Exception], None],
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
                batch_results = fetch_with_backoff(
                    bulk_fetcher,
                    batch_symbols,
                    period=job.period,
                    market=market,
                )
                batch_to_store = {}
                for symbol, data in batch_results.items():
                    if not data.get("has_error") and data.get("price_data") is not None:
                        price_df = data["price_data"]
                        if not price_df.empty:
                            batch_to_store[symbol] = price_df
                            refreshed += 1
                            refreshed_by_market[market_for_symbol(symbol)] += 1
                            batch_successes.append(symbol)
                        else:
                            failed += 1
                            failed_symbols.append(symbol)
                            failed_by_market[market_for_symbol(symbol)] += 1
                            batch_failures.append(symbol)
                            failure_details[symbol] = "Empty data returned"
                    else:
                        failed += 1
                        failed_symbols.append(symbol)
                        failed_by_market[market_for_symbol(symbol)] += 1
                        batch_failures.append(symbol)
                        failure_details[symbol] = data.get("error", "Unknown error")

                if batch_to_store:
                    price_cache.store_batch_in_cache(batch_to_store, also_store_db=True)

            except SoftTimeLimitExceeded:
                raise
            except Exception as exc:
                raise_if_transient_database_error(exc)
                logger.error("Batch %d error: %s", batch_num, exc)
                failed += len(batch_symbols)
                failed_symbols.extend(batch_symbols)
                failed_by_market.update(market_for_symbol(symbol) for symbol in batch_symbols)
                batch_failures.extend(batch_symbols)
                failure_details.update({symbol: str(exc) for symbol in batch_symbols})

            track_symbol_failures(
                price_cache,
                batch_successes,
                batch_failures,
                db,
                failure_details=failure_details,
            )

            processed_count += len(batch_symbols)
            progress = min(processed_count, total)
            percent = (progress / total) * 100 if total else 100.0
            update_task_state(progress, total, percent, refreshed, failed)
            mark_progress(
                progress,
                total,
                percent,
                f"Batch {batch_num}/{total_batches} · refreshing prices",
            )
            extend_lock()

            if processed_count < total:
                wait_between_batches()

    return PriceRefreshExecutionResult(
        refreshed=refreshed,
        failed=failed,
        failed_symbols=failed_symbols,
        refreshed_by_market=refreshed_by_market,
        failed_by_market=failed_by_market,
    )
