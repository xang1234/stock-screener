"""#543: the yfinance fundamentals batch loop starts no batch after the deadline."""

from __future__ import annotations

import pytest

import app.services.bulk_data_fetcher as bulk_module
from app.services.bulk_data_fetcher import BulkDataFetcher


class _NoYahoo:
    def __getattr__(self, name):
        raise AssertionError(f"yfinance.{name} used after the deadline")


def test_no_fundamentals_batch_starts_after_the_deadline(monkeypatch):
    monkeypatch.setattr(bulk_module.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(bulk_module, "yf", _NoYahoo())
    fetcher = BulkDataFetcher.__new__(BulkDataFetcher)

    results = fetcher.fetch_batch_fundamentals(
        ["A", "B", "C", "D"],
        batch_size=2,
        delay_between_batches=0,
        delay_per_ticker=0,
        deadline=30.0,
    )

    assert results == {}
