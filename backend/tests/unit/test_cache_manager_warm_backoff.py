"""warm_price_cache backs off only when failures look like provider throttling."""
from __future__ import annotations

import time
from types import SimpleNamespace

import pandas as pd

from app.services.bulk_data_fetcher import BulkDataFetcher
from app.services.cache_manager import CacheManager


def _warm(monkeypatch, failing_error, errors=None):
    symbols = [f"S{i}" for i in range(10)]
    errors = errors or {s: failing_error for s in symbols[:6]}

    def fake_fetch(_self, batch, **_kwargs):
        return {
            s: BulkDataFetcher._build_error_result(
                s, errors[s], error_kind=BulkDataFetcher._price_error_kind(errors[s])
            )
            if s in errors
            else {"has_error": False, "price_data": pd.DataFrame({"Close": [1.0]})}
            for s in batch
        }

    monkeypatch.setattr(BulkDataFetcher, "fetch_prices_in_batches", fake_fetch)
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)

    manager = CacheManager.__new__(CacheManager)
    manager.price_cache = SimpleNamespace(store_batch_in_cache=lambda *a, **k: None)
    stats = manager.warm_price_cache(symbols, batch_size=10, rate_limit=0)
    return stats, sleeps


def test_delisted_symbols_do_not_trigger_backoff(monkeypatch):
    stats, sleeps = _warm(monkeypatch, "S0: possibly delisted; no price data found")

    assert sleeps == []
    assert (stats["successful"], stats["failed"]) == (4, 6)


def test_rate_limited_symbols_trigger_backoff(monkeypatch):
    stats, sleeps = _warm(monkeypatch, "429 Too Many Requests")

    assert sleeps == [60]
    assert (stats["successful"], stats["failed"]) == (4, 6)


def test_delisted_symbols_do_not_dilute_a_rate_limited_batch(monkeypatch):
    # Every fetchable symbol was throttled; the delisted half must not hide it.
    errors = {f"S{i}": "S: possibly delisted; no price data found" for i in range(5)}
    errors |= {f"S{i}": "429 Too Many Requests" for i in range(5, 10)}
    _stats, sleeps = _warm(monkeypatch, None, errors=errors)

    assert sleeps == [60]
