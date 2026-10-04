from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy.exc import OperationalError

from app.services.failed_price_retry_runner import (
    FailedPriceRetryRunner,
    FailedPriceRetryRunnerDependencies,
)
from app.tasks.transient_database import raise_if_transient_database_error


def _price_df(close: float = 150.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [close - 1],
            "High": [close + 1],
            "Low": [close - 2],
            "Close": [close],
            "Adj Close": [close - 0.5],
            "Volume": [1_000_000],
        },
        index=pd.to_datetime([date(2026, 3, 20)]),
    )


def _success(symbols):
    frame = _price_df()
    return {
        symbol: {
            "has_error": False,
            "error": None,
            "price_data": frame,
        }
        for symbol in symbols
    }


class _NoopPriceCache:
    def store_refreshed_batch(self, *_args, **_kwargs):
        pass


def _run(
    *,
    symbols,
    fetch,
    price_cache,
    retry_calls,
    track=lambda *_args, **_kwargs: None,
    market="US",
    attempt=1,
    retry_countdown=30,
    batch_size=250,
):
    runner = FailedPriceRetryRunner(
        FailedPriceRetryRunnerDependencies(
            fetch_with_backoff=fetch,
            track_symbol_failures=track,
            raise_if_transient_database_error=raise_if_transient_database_error,
            schedule_failed_symbol_retry=(
                lambda retry_symbols, **kwargs: retry_calls.append(
                    {"symbols": retry_symbols, **kwargs}
                )
            ),
        )
    )
    return runner.run(
        price_cache=price_cache,
        bulk_fetcher=object(),
        symbols=symbols,
        market=market,
        attempt=attempt,
        retry_countdown=retry_countdown,
        batch_size=batch_size,
    )


@pytest.mark.parametrize(
    ("error_kind", "error", "expected_retry"),
    [
        ("rate_limit", "rate limited", True),
        ("no_price_data", "Provider returned no usable rows", False),
    ],
)
def test_runner_reschedules_only_retryable_failures(
    error_kind,
    error,
    expected_retry,
):
    retry_calls = []

    result = _run(
        symbols=["0143.T"],
        price_cache=_NoopPriceCache(),
        retry_calls=retry_calls,
        attempt=2,
        fetch=lambda _fetcher, symbols, **_kwargs: {
            symbol: {
                "has_error": True,
                "error": error,
                "error_kind": error_kind,
                "price_data": None,
            }
            for symbol in symbols
        },
    )

    assert result.failed_symbols == ("0143.T",)
    assert retry_calls == (
        [
            {
                "symbols": ["0143.T"],
                "market": "US",
                "attempt": 3,
                "countdown": 30,
            }
        ]
        if expected_retry
        else []
    )


def test_runner_fetches_and_persists_bounded_batches():
    timeline = []

    class PriceCache:
        def store_refreshed_batch(self, price_data, *, period, market_by_symbol):
            timeline.append(("store", tuple(price_data), period))

    def fetch(_fetcher, symbols, **_kwargs):
        timeline.append(("fetch", tuple(symbols)))
        return _success(symbols)

    result = _run(
        symbols=["AAPL", "MSFT", "NVDA"],
        fetch=fetch,
        price_cache=PriceCache(),
        retry_calls=[],
        batch_size=2,
    )

    assert timeline == [
        ("fetch", ("AAPL", "MSFT")),
        ("store", ("AAPL", "MSFT"), "2y"),
        ("fetch", ("NVDA",)),
        ("store", ("NVDA",), "2y"),
    ]
    assert result.refreshed == 3
    assert result.failed == 0


def test_runner_aggregates_mixed_results_across_batches():
    retry_calls = []

    def fetch(_fetcher, symbols, **_kwargs):
        results = _success(symbols)
        if "MSFT" in results:
            results["MSFT"] = {
                "has_error": True,
                "error": "rate limited",
                "error_kind": "rate_limit",
                "price_data": None,
            }
        if "META" in results:
            results["META"] = {
                "has_error": True,
                "error": "Provider returned no usable rows",
                "error_kind": "no_price_data",
                "price_data": None,
            }
        return results

    result = _run(
        symbols=["AAPL", "MSFT", "NVDA", "META"],
        fetch=fetch,
        price_cache=_NoopPriceCache(),
        retry_calls=retry_calls,
        batch_size=2,
    )

    assert result.refreshed == 2
    assert result.failed == 2
    assert result.failed_symbols == ("MSFT", "META")
    assert retry_calls == [
        {
            "symbols": ["MSFT"],
            "market": "US",
            "attempt": 2,
            "countdown": 30,
        }
    ]


def test_runner_reschedules_current_and_remaining_symbols_after_store_failure():
    fetched_batches = []
    retry_calls = []

    class PriceCache:
        stores = 0

        def store_refreshed_batch(self, _price_data, *, period, market_by_symbol):
            self.stores += 1
            if self.stores == 2:
                raise RuntimeError("redis unavailable")

    def fetch(_fetcher, symbols, **_kwargs):
        fetched_batches.append(tuple(symbols))
        results = _success(symbols)
        if "MSFT" in results:
            results["MSFT"] = {
                "has_error": True,
                "error": "rate limited",
                "error_kind": "rate_limit",
                "price_data": None,
            }
        return results

    result = _run(
        symbols=["AAPL", "MSFT", "NVDA", "AMZN", "META"],
        fetch=fetch,
        price_cache=PriceCache(),
        retry_calls=retry_calls,
        batch_size=2,
    )

    assert fetched_batches == [("AAPL", "MSFT"), ("NVDA", "AMZN")]
    assert result.refreshed == 1
    assert result.failed == 4
    assert result.failed_symbols == ("MSFT", "NVDA", "AMZN", "META")
    assert result.error == "redis unavailable"
    assert retry_calls[0]["symbols"] == ["MSFT", "NVDA", "AMZN", "META"]


def test_runner_propagates_soft_time_limit_from_storage():
    class PriceCache:
        def store_refreshed_batch(self, *_args, **_kwargs):
            raise SoftTimeLimitExceeded()

    with pytest.raises(SoftTimeLimitExceeded) as raised:
        _run(
            symbols=["AAPL"],
            fetch=lambda _fetcher, symbols, **_kwargs: _success(symbols),
            price_cache=PriceCache(),
            retry_calls=[],
        )

    assert raised.value.__suppress_context__ is True


def test_runner_propagates_transient_database_storage_failure():
    transient_error = OperationalError(
        "insert prices",
        {},
        Exception("database system is not yet accepting connections"),
    )

    class PriceCache:
        def store_refreshed_batch(self, *_args, **_kwargs):
            raise transient_error

    with pytest.raises(OperationalError) as raised:
        _run(
            symbols=["AAPL"],
            fetch=lambda _fetcher, symbols, **_kwargs: _success(symbols),
            price_cache=PriceCache(),
            retry_calls=[],
        )

    assert raised.value is transient_error


def test_runner_reschedules_prior_current_and_remaining_after_tracking_failure():
    retry_calls = []
    tracking_calls = 0

    def fetch(_fetcher, symbols, **_kwargs):
        results = _success(symbols)
        if "MSFT" in results:
            results["MSFT"] = {
                "has_error": True,
                "error": "rate limited",
                "error_kind": "rate_limit",
                "price_data": None,
            }
        return results

    def track(*_args, **_kwargs):
        nonlocal tracking_calls
        tracking_calls += 1
        if tracking_calls == 2:
            raise RuntimeError("failure tracking unavailable")

    result = _run(
        symbols=["AAPL", "MSFT", "NVDA", "AMZN", "META"],
        fetch=fetch,
        price_cache=_NoopPriceCache(),
        retry_calls=retry_calls,
        track=track,
        batch_size=2,
    )

    assert result.refreshed == 1
    assert result.failed == 4
    assert result.failed_symbols == ("MSFT", "NVDA", "AMZN", "META")
    assert result.error == "failure tracking unavailable"
    assert retry_calls[0]["symbols"] == ["MSFT", "NVDA", "AMZN", "META"]
