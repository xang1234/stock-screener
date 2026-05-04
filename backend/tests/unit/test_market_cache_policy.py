"""Tests for the market-aware cache policy."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import fnmatch
import json
import pickle

import pandas as pd
import pytest

from app.domain.markets import Market
from app.services.benchmark_cache_service import BenchmarkCacheService
from app.services.cache.market_cache_policy import MarketAwareCachePolicy
from app.services.cache.price_cache_freshness import PriceCacheFreshnessPolicy
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
    assert policy.is_datetime_fresh("benchmark", now - timedelta(days=1), now=now, market="HK") is True
    assert policy.is_datetime_fresh("benchmark", now - timedelta(days=2), now=now, market="HK") is False
    assert policy.ttl_seconds("fundamentals", market="HK") == 604800
    assert policy.is_datetime_fresh("fundamentals", now - timedelta(days=7), now=now, market="HK") is True
    assert policy.is_datetime_fresh("fundamentals", now - timedelta(days=8), now=now, market="HK") is False


def test_market_cache_policy_uses_fractional_utc_age_for_freshness():
    policy = MarketAwareCachePolicy()
    last_update = datetime(2026, 5, 1, 23, 30, tzinfo=timezone.utc)
    now = datetime(2026, 5, 9, 0, 31, tzinfo=timezone(timedelta(hours=1)))

    assert policy.is_datetime_fresh("fundamentals", last_update, now=now, market="HK") is False


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


class _MutableRedis:
    def __init__(self, values=None):
        self.values = values or {}
        self.deleted = []

    def get(self, key):
        return self.values.get(key)

    def delete(self, *keys):
        self.deleted.extend(keys)


class _ScanningPipeline:
    def __init__(self, redis):
        self.redis = redis
        self.keys = []

    def get(self, key):
        self.keys.append(key)
        return self

    def execute(self):
        return [self.redis.values[key] for key in self.keys]


class _ScanningRedis:
    def __init__(self, values):
        self.values = values
        self.matches = []

    def scan(self, cursor, match, count=500):
        self.matches.append(match)
        keys = [key for key in self.values if fnmatch.fnmatch(key, match)]
        return 0, keys

    def pipeline(self):
        return _ScanningPipeline(self)


class _FailingPipeline:
    def setex(self, *args, **kwargs):
        return None

    def execute(self):
        raise RuntimeError("pipeline unavailable")


class _FailingPipelineRedis:
    def pipeline(self):
        return _FailingPipeline()


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


def test_price_cache_invalidate_uses_market_scoped_keys():
    redis = _MutableRedis()
    service = PriceCacheService(redis_client=redis, session_factory=lambda: None)

    service.invalidate_cache("0700.HK", market="HK")

    assert redis.deleted == [
        "price:HK:0700.HK:recent",
        "price:HK:0700.HK:last_update",
    ]


def test_price_cache_batch_pipeline_fallback_preserves_market(monkeypatch):
    service = PriceCacheService(redis_client=_FailingPipelineRedis(), session_factory=lambda: None)
    data = pd.DataFrame(
        {"Close": [100.0]},
        index=pd.to_datetime(["2026-05-04"]),
    )
    calls = []
    monkeypatch.setattr(
        service,
        "_store_recent_in_redis",
        lambda symbol, payload, market=None: calls.append((symbol, market)),
    )

    service.store_batch_in_cache({"0700.HK": data}, also_store_db=False, market="HK")

    assert calls == [("0700.HK", "HK")]


def test_price_cache_freshness_scans_new_and_legacy_fetch_meta_keys(monkeypatch):
    import app.services.cache.price_cache_freshness as module

    redis = _ScanningRedis(
        {
            "price:HK:0700.HK:fetch_meta": json.dumps({"needs_refresh_after_close": True}),
            "price:AAPL:fetch_meta": json.dumps({"needs_refresh_after_close": True}),
        }
    )
    policy = PriceCacheFreshnessPolicy(
        logger=type("Logger", (), {"info": lambda *args, **kwargs: None, "error": lambda *args, **kwargs: None})(),
        redis_client=redis,
        fetch_meta_key_template=("price:*:*:fetch_meta", "price:*:fetch_meta"),
        get_expected_data_date=lambda: None,
        get_fetch_metadata=lambda symbol: None,
    )
    monkeypatch.setattr(module, "get_eastern_now", lambda: datetime(2026, 5, 4, 17, 0, 0))
    monkeypatch.setattr(module, "is_market_open", lambda now=None: False)

    assert policy.get_stale_intraday_symbols() == ["0700.HK", "AAPL"]
    assert redis.matches == ["price:*:*:fetch_meta", "price:*:fetch_meta"]


def test_fundamentals_cache_invalidate_uses_market_scoped_key():
    redis = _MutableRedis()
    service = FundamentalsCacheService(redis_client=redis, session_factory=lambda: None)

    service.invalidate_cache("0700.HK", market="HK")

    assert redis.deleted == ["fundamentals:HK:0700.HK"]


def test_benchmark_cache_invalidate_deletes_each_market_scope():
    redis = _MutableRedis()
    service = BenchmarkCacheService(redis_client=redis, session_factory=lambda: None)

    service.invalidate_cache(period="2y")

    assert "benchmark:HK:^HSI:2y" in redis.deleted
    assert "benchmark:IN:^NSEI:2y" in redis.deleted
    assert "benchmark:US:SPY:2y" in redis.deleted
    assert "benchmark:US:^HSI:2y" not in redis.deleted
