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

    class _Context:
        def fetch_batch(self, symbols, *, period, market):
            fetch_calls.append((tuple(symbols), period, market))
            return {
                symbol: {
                    "has_error": False,
                    "price_data": SimpleNamespace(empty=False),
                }
                for symbol in symbols
            }

        def store_prices(self, rows):
            stored_batches.append((dict(rows), True))

        def track_symbol_failures(self, successes, failures, *, failure_details):
            tracked.append((tuple(successes), tuple(failures), dict(failure_details)))

        def market_for_symbol(self, symbol):
            return "US" if symbol != "C" else "HK"

        def publish_progress(self, current, total, percent, message, *, refreshed, failed):
            progress_updates.append((current, total, round(percent, 1), message))
            task_updates.append((current, total, round(percent, 1), refreshed, failed))

        def extend_lock(self):
            lock_extensions.append("extended")

        def wait_between_batches(self):
            waits.append("wait")

        def raise_if_transient_database_error(self, exc):
            return None

    result = run_price_refresh_jobs(
        jobs=(
            PriceRefreshJob(kind="stale", symbols=("A", "B"), period="7d"),
            PriceRefreshJob(kind="no_history", symbols=("C",), period="2y"),
        ),
        total=3,
        batch_size=2,
        market="US",
        context=_Context(),
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


def test_run_price_refresh_jobs_marks_unreturned_symbols_failed():
    from app.services.price_refresh_execution import run_price_refresh_jobs
    from app.services.price_refresh_planning import PriceRefreshJob

    tracked = []
    task_updates = []

    class _Context:
        def fetch_batch(self, symbols, *, period, market):
            del symbols, period, market
            return {
                "A": {
                    "has_error": False,
                    "price_data": SimpleNamespace(empty=False),
                }
            }

        def store_prices(self, rows):
            assert set(rows) == {"A"}

        def track_symbol_failures(self, successes, failures, *, failure_details):
            tracked.append((tuple(successes), tuple(failures), dict(failure_details)))

        def market_for_symbol(self, _symbol):
            return "US"

        def publish_progress(self, current, total, percent, message, *, refreshed, failed):
            del message
            task_updates.append((current, total, round(percent, 1), refreshed, failed))

        def extend_lock(self):
            return None

        def wait_between_batches(self):
            return None

        def raise_if_transient_database_error(self, exc):
            return None

    result = run_price_refresh_jobs(
        jobs=(PriceRefreshJob(kind="stale", symbols=("A", "B"), period="7d"),),
        total=2,
        batch_size=2,
        market="US",
        context=_Context(),
    )

    assert tracked == [
        (("A",), ("B",), {"B": "No data returned"}),
    ]
    assert task_updates[-1] == (2, 2, 100.0, 1, 1)
    assert result.refreshed == 1
    assert result.failed == 1
    assert result.failed_symbols == ["B"]
    assert result.failed_by_market == Counter({"US": 1})
