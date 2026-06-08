from __future__ import annotations

from collections import Counter
from types import SimpleNamespace


def test_run_price_refresh_jobs_batches_by_plan_period_and_reports_progress():
    from app.services.price_refresh_execution import run_price_refresh_jobs
    from app.services.price_refresh_planning import PriceRefreshJob

    fetch_calls = []
    stored_batches = []
    tracked = []
    progress_updates = []
    task_updates = []
    waits = []
    lock_extensions = []

    class _PriceCache:
        def store_batch_in_cache(self, rows, *, also_store_db):
            stored_batches.append((dict(rows), also_store_db))

    def fetch_with_backoff(fetcher, symbols, *, period, market):
        del fetcher
        fetch_calls.append((tuple(symbols), period, market))
        return {
            symbol: {
                "has_error": False,
                "price_data": SimpleNamespace(empty=False),
            }
            for symbol in symbols
        }

    result = run_price_refresh_jobs(
        jobs=(
            PriceRefreshJob(kind="stale", symbols=("A", "B"), period="7d"),
            PriceRefreshJob(kind="no_history", symbols=("C",), period="2y"),
        ),
        total=3,
        batch_size=2,
        bulk_fetcher=object(),
        price_cache=_PriceCache(),
        db=object(),
        market="US",
        fetch_with_backoff=fetch_with_backoff,
        track_symbol_failures=lambda price_cache, successes, failures, db, failure_details: tracked.append(
            (tuple(successes), tuple(failures), dict(failure_details))
        ),
        market_for_symbol=lambda symbol: "US" if symbol != "C" else "HK",
        mark_progress=lambda current, total, percent, message: progress_updates.append(
            (current, total, round(percent, 1), message)
        ),
        update_task_state=lambda current, total, percent, refreshed, failed: task_updates.append(
            (current, total, round(percent, 1), refreshed, failed)
        ),
        extend_lock=lambda: lock_extensions.append("extended"),
        wait_between_batches=lambda: waits.append("wait"),
        raise_if_transient_database_error=lambda exc: None,
    )

    assert fetch_calls == [
        (("A", "B"), "7d", "US"),
        (("C",), "2y", "US"),
    ]
    assert [set(batch.keys()) for batch, _ in stored_batches] == [{"A", "B"}, {"C"}]
    assert tracked == [
        (("A", "B"), (), {}),
        (("C",), (), {}),
    ]
    assert progress_updates == [
        (2, 3, 66.7, "Batch 1/2 · refreshing prices"),
        (3, 3, 100.0, "Batch 2/2 · refreshing prices"),
    ]
    assert task_updates[-1] == (3, 3, 100.0, 3, 0)
    assert waits == ["wait"]
    assert lock_extensions == ["extended", "extended"]
    assert result.refreshed == 3
    assert result.failed == 0
    assert result.failed_symbols == []
    assert result.refreshed_by_market == Counter({"US": 2, "HK": 1})
