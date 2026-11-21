"""warm_price_cache backs off only when failures look like provider throttling."""
from __future__ import annotations

import time
from types import SimpleNamespace

import pandas as pd

from app.services.bulk_data_fetcher import BulkDataFetcher
from app.services.cache_manager import CacheManager


def _warm(monkeypatch, failing_error):
    symbols = [f"S{i}" for i in range(10)]
    failing = set(symbols[:6])

    def fake_fetch(_self, batch, **_kwargs):
        return {
            s: BulkDataFetcher._build_error_result(
                s, failing_error, error_kind=BulkDataFetcher._price_error_kind(failing_error)
            )
            if s in failing
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
