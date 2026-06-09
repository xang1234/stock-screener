from __future__ import annotations

from collections import Counter
from types import SimpleNamespace


def test_classify_price_refresh_batch_returns_shared_batch_outcome():
    from app.services.price_refresh_execution import classify_price_refresh_batch
    from app.services.price_refresh_planning import PriceRefreshJob, PriceRefreshJobKind

    job = PriceRefreshJob(
        kind=PriceRefreshJobKind.NO_HISTORY,
        symbols=("0143.T", "7203.T"),
        period="2y",
    )

    outcome = classify_price_refresh_batch(
        batch_number=1,
        total_batches=1,
        job=job,
        symbols=job.symbols,
        batch_results={
            "0143.T": {
                "has_error": True,
                "price_data": None,
                "error": "Provider returned no usable rows",
                "error_kind": "no_price_data",
            },
            "7203.T": {
                "has_error": False,
                "price_data": SimpleNamespace(empty=False),
            },
        },
        market_for_symbol=lambda _symbol: "JP",
    )

    assert outcome.job is job
    assert outcome.successes == ("7203.T",)
    assert outcome.failures == ("0143.T",)
    assert outcome.failure_kinds == {"0143.T": "no_price_data"}
    assert outcome.refreshed_by_market == Counter({"JP": 1})
    assert outcome.failed_by_market == Counter({"JP": 1})


def test_iter_price_refresh_batches_returns_batch_outcomes_without_side_effect_context():
    from app.services.price_refresh_execution import (
        iter_price_refresh_batches,
        summarize_price_refresh_batches,
    )
    from app.services.price_refresh_planning import PriceRefreshJob, PriceRefreshJobKind

    fetch_calls = []

    def fetch_batch(symbols, *, period, market):
        fetch_calls.append((tuple(symbols), period, market))
        return {
            symbol: {
                "has_error": False,
                "price_data": SimpleNamespace(empty=False),
            }
            for symbol in symbols
        }

    batches = list(iter_price_refresh_batches(
        jobs=(
            PriceRefreshJob(kind=PriceRefreshJobKind.STALE, symbols=("A", "B"), period="7d"),
            PriceRefreshJob(kind=PriceRefreshJobKind.NO_HISTORY, symbols=("C",), period="2y"),
        ),
        batch_size=2,
        market="US",
        fetch_batch=fetch_batch,
        market_for_symbol=lambda symbol: "US" if symbol != "C" else "HK",
        raise_if_transient_database_error=lambda exc: None,
    ))

    assert fetch_calls == [
        (("A", "B"), "7d", "US"),
        (("C",), "2y", "US"),
    ]
    assert [set(batch.price_data_by_symbol.keys()) for batch in batches] == [
        {"A", "B"},
        {"C"},
    ]
    assert [(batch.successes, batch.failures, batch.failure_details) for batch in batches] == [
        (("A", "B"), (), {}),
        (("C",), (), {}),
    ]
    assert [(batch.batch_number, batch.total_batches) for batch in batches] == [(1, 2), (2, 2)]

    summary = summarize_price_refresh_batches(batches)
    assert summary.refreshed == 3
    assert summary.failed == 0
    assert summary.total == 3
    assert summary.processed == 3
    assert summary.failed_symbols == []
    assert summary.refreshed_by_market == Counter({"US": 2, "HK": 1})


def test_iter_price_refresh_batches_marks_unreturned_symbols_failed():
    from app.services.price_refresh_execution import (
        iter_price_refresh_batches,
        summarize_price_refresh_batches,
    )
    from app.services.price_refresh_planning import PriceRefreshJob, PriceRefreshJobKind

    def fetch_batch(symbols, *, period, market):
        del symbols, period, market
        return {
            "A": {
                "has_error": False,
                "price_data": SimpleNamespace(empty=False),
            }
        }

    batches = list(iter_price_refresh_batches(
        jobs=(PriceRefreshJob(kind=PriceRefreshJobKind.STALE, symbols=("A", "B"), period="7d"),),
        batch_size=2,
        market="US",
        fetch_batch=fetch_batch,
        market_for_symbol=lambda _symbol: "US",
        raise_if_transient_database_error=lambda exc: None,
    ))

    assert len(batches) == 1
    assert batches[0].successes == ("A",)
    assert batches[0].failures == ("B",)
    assert batches[0].failure_details == {"B": "No data returned"}
    assert set(batches[0].price_data_by_symbol) == {"A"}

    summary = summarize_price_refresh_batches(batches)
    assert summary.refreshed == 1
    assert summary.failed == 1
    assert summary.failed_symbols == ["B"]
    assert summary.failed_by_market == Counter({"US": 1})


def test_iter_price_refresh_batches_can_delegate_provider_batching():
    from app.services.price_refresh_execution import iter_price_refresh_batches
    from app.services.price_refresh_planning import PriceRefreshJob, PriceRefreshJobKind

    fetch_calls = []

    def fetch_batch(symbols, *, period, market):
        fetch_calls.append((tuple(symbols), period, market))
        return {
            symbol: {
                "has_error": False,
                "price_data": SimpleNamespace(empty=False),
            }
            for symbol in symbols
        }

    list(iter_price_refresh_batches(
        jobs=(PriceRefreshJob(
            kind=PriceRefreshJobKind.NO_HISTORY,
            symbols=("A", "B", "C", "D"),
            period="2y",
        ),),
        batch_size=None,
        market="JP",
        fetch_batch=fetch_batch,
        market_for_symbol=lambda _symbol: "JP",
        raise_if_transient_database_error=lambda exc: None,
    ))

    assert fetch_calls == [(("A", "B", "C", "D"), "2y", "JP")]


def test_iter_price_refresh_batches_carries_failure_kinds():
    from app.services.price_refresh_execution import (
        iter_price_refresh_batches,
        summarize_price_refresh_batches,
    )
    from app.services.price_refresh_planning import PriceRefreshJob, PriceRefreshJobKind

    def fetch_batch(symbols, *, period, market):
        del symbols, period, market
        return {
            "0143.T": {
                "has_error": True,
                "price_data": None,
                "error": "YFPricesMissingError('possibly delisted; no price data found')",
                "error_kind": "no_price_data",
            },
            "7203.T": {
                "has_error": True,
                "price_data": None,
                "error": "429 Too Many Requests",
                "error_kind": "rate_limit",
            },
        }

    batches = list(iter_price_refresh_batches(
        jobs=(PriceRefreshJob(
            kind=PriceRefreshJobKind.NO_HISTORY,
            symbols=("0143.T", "7203.T"),
            period="2y",
        ),),
        batch_size=None,
        market="JP",
        fetch_batch=fetch_batch,
        market_for_symbol=lambda _symbol: "JP",
        raise_if_transient_database_error=lambda exc: None,
    ))

    assert batches[0].failure_kinds == {
        "0143.T": "no_price_data",
        "7203.T": "rate_limit",
    }
    assert summarize_price_refresh_batches(batches).failure_kinds == {
        "0143.T": "no_price_data",
        "7203.T": "rate_limit",
    }


def test_price_refresh_execution_summary_accumulates_batches_incrementally():
    from app.services.price_refresh_execution import (
        PriceRefreshBatchOutcome,
        PriceRefreshExecutionAccumulator,
    )
    from app.services.price_refresh_planning import PriceRefreshJob, PriceRefreshJobKind

    stale_job = PriceRefreshJob(
        kind=PriceRefreshJobKind.STALE,
        symbols=("A", "B"),
        period="7d",
    )
    missing_history_job = PriceRefreshJob(
        kind=PriceRefreshJobKind.NO_HISTORY,
        symbols=("C",),
        period="2y",
    )
    accumulator = PriceRefreshExecutionAccumulator()

    accumulator.add(
        PriceRefreshBatchOutcome(
            batch_number=1,
            total_batches=2,
            job=stale_job,
            symbols=("A", "B"),
            price_data_by_symbol={"A": object()},
            successes=("A",),
            failures=("B",),
            failure_details={"B": "No data returned"},
            refreshed_by_market=Counter({"US": 1}),
            failed_by_market=Counter({"US": 1}),
        )
    )
    accumulator.add(
        PriceRefreshBatchOutcome(
            batch_number=2,
            total_batches=2,
            job=missing_history_job,
            symbols=("C",),
            price_data_by_symbol={"C": object()},
            successes=("C",),
            failures=(),
            failure_details={},
            refreshed_by_market=Counter({"HK": 1}),
            failed_by_market=Counter(),
        )
    )

    summary = accumulator.summary()

    assert summary.processed == 3
    assert summary.refreshed == 2
    assert summary.failed == 1
    assert summary.failed_symbols == ["B"]
    assert summary.failure_kinds == {}
    assert summary.refreshed_by_market == Counter({"US": 1, "HK": 1})
    assert summary.failed_by_market == Counter({"US": 1})
