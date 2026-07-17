from __future__ import annotations

from types import SimpleNamespace

import pytest
from celery.exceptions import SoftTimeLimitExceeded


def test_live_price_refresh_runner_raises_interruption_with_partial_summary():
    from app.services.price_refresh_live_runner import (
        LivePriceRefreshRunner,
        LivePriceRefreshRunnerDependencies,
        PriceRefreshExecutionError,
    )
    from app.services.price_refresh_planning import PriceRefreshJob, PriceRefreshJobKind

    fetch_calls = []

    def fetch_with_backoff(
        _bulk_fetcher,
        symbols,
        *,
        period,
        market,
        progress_callback=None,
    ):
        del period, market, progress_callback
        fetch_calls.append(tuple(symbols))
        if len(fetch_calls) == 2:
            raise SoftTimeLimitExceeded()
        return {
            symbol: {
                "has_error": False,
                "price_data": SimpleNamespace(empty=False),
            }
            for symbol in symbols
        }

    class _FakePriceCache:
        def store_batch_in_cache(self, *_args, **_kwargs):
            pass

    class _FakeReporter:
        def publish_progress(self, *_args, **_kwargs):
            pass

    class _FakeLock:
        def extend_lock(self, *_args, **_kwargs):
            pass

    runner = LivePriceRefreshRunner(
        LivePriceRefreshRunnerDependencies(
            fetch_with_backoff=fetch_with_backoff,
            track_symbol_failures=lambda *_args, **_kwargs: None,
            data_fetch_lock_factory=lambda: _FakeLock(),
            raise_if_transient_database_error=lambda _exc: None,
        )
    )

    with pytest.raises(PriceRefreshExecutionError) as raised:
        runner.run(
            task=SimpleNamespace(request=SimpleNamespace(id="task-1")),
            bulk_fetcher=object(),
            price_cache=_FakePriceCache(),
            db=object(),
            jobs=(
                PriceRefreshJob(
                    kind=PriceRefreshJobKind.STALE,
                    symbols=("A", "B"),
                    period="7d",
                ),
            ),
            total=2,
            batch_size=1,
            market="JP",
            effective_market="JP",
            activity_lifecycle="bootstrap",
            symbol_markets={"A": "JP", "B": "JP"},
            activity_reporter=_FakeReporter(),
        )

    assert isinstance(raised.value.cause, SoftTimeLimitExceeded)
    assert raised.value.summary.total == 2
    assert raised.value.summary.processed == 1
    assert raised.value.summary.refreshed == 1
    assert raised.value.summary.failed == 0


def test_retry_scheduler_skips_permanent_no_price_data_failures():
    from app.services.price_refresh_live_runner import PriceRefreshRetryScheduler

    calls = []
    scheduler = PriceRefreshRetryScheduler(
        schedule_failed_symbol_retry=lambda **kwargs: calls.append(kwargs)
    )

    scheduler.schedule(
        ["0143.T", "7203.T"],
        failure_kinds={
            "0143.T": "no_price_data",
            "7203.T": "rate_limit",
        },
        effective_market="JP",
        symbol_markets={"0143.T": "JP", "7203.T": "JP"},
        activity_lifecycle="bootstrap",
    )

    assert calls == [
        {
            "symbols": ["7203.T"],
            "market": "JP",
            "attempt": 1,
            "countdown": 30,
        }
    ]
