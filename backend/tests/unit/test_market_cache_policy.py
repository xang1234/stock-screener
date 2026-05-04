"""Tests for the market-aware cache policy."""

from __future__ import annotations

from datetime import datetime, timedelta
import pickle

import pytest

from app.domain.markets import Market
from app.services.benchmark_cache_service import BenchmarkCacheService
from app.services.cache.market_cache_policy import MarketAwareCachePolicy
from app.services.fundamentals_cache_service import FundamentalsCacheService
from app.services.price_cache_service import PriceCacheService


def test_market_cache_policy_builds_market_scoped_keys_and_locks():
    policy = MarketAwareCachePolicy()

    assert policy.key("price", "aapl", market="us", parts=("recent",)) == "price:US:AAPL:recent"
    assert policy.key("fundamentals", "0700.hk", market=Market("HK")) == "fundamentals:HK:0700.HK"
    assert policy.lock_key("benchmark", "^n225", market="JP", parts=("2y",)) == "benchmark:JP:^N225:2y:lock"


def test_market_cache_policy_defaults_to_us_for_legacy_callers():
    policy = MarketAwareCachePolicy()

    assert policy.key("price", "SPY", parts=("fetch_meta",)) == "price:US:SPY:fetch_meta"


def test_market_cache_policy_centralizes_ttl_and_freshness():
    policy = MarketAwareCachePolicy()
    now = datetime(2026, 5, 4, 12, 0, 0)

    assert policy.ttl_seconds("benchmark", market="HK") == 86400
    assert policy.ttl_seconds("fundamentals", market="HK") == 604800
    assert policy.is_datetime_fresh("fundamentals", now - timedelta(days=7), now=now, market="HK") is True
    assert policy.is_datetime_fresh("fundamentals", now - timedelta(days=8), now=now, market="HK") is False


def test_market_cache_policy_rejects_unknown_market():
    policy = MarketAwareCachePolicy()

    with pytest.raises(ValueError):
        policy.key("price", "VOD.L", market="UK", parts=("recent",))


def test_cache_services_delegate_market_scoped_key_construction_to_policy():
    benchmark = BenchmarkCacheService(redis_client=None, session_factory=lambda: None)
    fundamentals = FundamentalsCacheService(redis_client=None, session_factory=lambda: None)
    price = PriceCacheService(redis_client=None, session_factory=lambda: None)

    assert benchmark._redis_data_key("^HSI", "2y", market="HK") == "benchmark:HK:^HSI:2y"
    assert fundamentals._redis_data_key("0700.HK", market="HK") == "fundamentals:HK:0700.HK"
    assert price._redis_recent_key("0700.HK", market="HK") == "price:HK:0700.HK:recent"
    assert price._redis_fetch_meta_key("0700.HK", market="HK") == "price:HK:0700.HK:fetch_meta"


class _Pipeline:
    def __init__(self, results):
        self.results = results
        self.keys = []

    def get(self, key):
        self.keys.append(key)
        return self

    def execute(self):
        return list(self.results)


class _Redis:
    def __init__(self, results):
        self.results = results
        self.pipeline_instance = None

    def pipeline(self):
        self.pipeline_instance = _Pipeline(self.results)
        return self.pipeline_instance


def test_fundamentals_bulk_get_reads_market_scoped_keys():
    payload = {"market_cap": 100, "sector": "Internet"}
    redis = _Redis([pickle.dumps(payload)])
    service = FundamentalsCacheService(redis_client=redis, session_factory=lambda: None)
    service._needs_db_enrichment = lambda fundamentals: False  # type: ignore[method-assign]

    result = service.get_many(["0700.HK"], market_by_symbol={"0700.HK": "HK"})

    assert result["0700.HK"] == payload
    assert redis.pipeline_instance.keys == ["fundamentals:HK:0700.HK"]


def test_fundamentals_bulk_db_fallback_warms_market_scoped_keys(monkeypatch):
    redis = _Redis([None])
    service = FundamentalsCacheService(redis_client=redis, session_factory=lambda: None)
    payload = {"market_cap": 100, "sector": "Internet"}
    stored = []

    monkeypatch.setattr(
        service,
        "_get_many_from_database",
        lambda symbols: {"0700.HK": (payload, datetime(2026, 5, 4, 12, 0, 0))},
    )
    monkeypatch.setattr(
        service,
        "_store_in_redis_for_market",
        lambda symbol, data, market=None: stored.append((symbol, data, market)),
    )

    result = service.get_many(["0700.HK"], market_by_symbol={"0700.HK": "HK"})

    assert result["0700.HK"] == payload
    assert stored == [("0700.HK", payload, "HK")]
