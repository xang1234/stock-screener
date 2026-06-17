"""Unit tests for the aggregated Daily Snapshot service helpers."""

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.api.v1.market_scan import _if_none_match_matches
from app.schemas.market_scan import DailySnapshotResponse
from app.services.daily_snapshot_service import (
    DAILY_SNAPSHOT_CACHE_TTL_SECONDS,
    DAILY_SNAPSHOT_MEMORY_CACHE_MAX_ENTRIES,
    DAILY_SNAPSHOT_SCHEMA_VERSION,
    _clear_daily_snapshot_memory_cache,
    _scan_freshness,
    daily_snapshot_cache_key,
    daily_snapshot_etag,
    get_or_build_daily_snapshot_payload,
    get_daily_snapshot_memory_cache,
    set_daily_snapshot_memory_cache,
)
from app.services.price_refresh_plan_builder import _key_market_refresh_symbols


def _norm(market):
    return str(market).upper()


class FakeRedis:
    def __init__(self, *, payload=None, ttl_seconds=10):
        self.payload = payload
        self.ttl_seconds = ttl_seconds
        self.get_calls = []
        self.setex_calls = []
        self.ttl_calls = []

    def get(self, key):
        self.get_calls.append(key)
        return self.payload

    def ttl(self, key):
        self.ttl_calls.append(key)
        if isinstance(self.ttl_seconds, Exception):
            raise self.ttl_seconds
        return self.ttl_seconds

    def setex(self, key, ttl_seconds, payload_json):
        self.setex_calls.append((key, ttl_seconds, payload_json))


class TestKeyMarketRefreshSymbols:
    def test_us_includes_aliased_data_symbols(self):
        symbols = _key_market_refresh_symbols("US", _norm)
        # TradingView display symbols must be resolved to fetchable Yahoo symbols
        assert "BTC-USD" in symbols
        assert "^VIX" in symbols
        assert "DX-Y.NYB" in symbols
        assert "SGD=X" in symbols
        assert all(market == "US" for market in symbols.values())
        # Display-only symbols never leak into the refresh plan
        assert "BITSTAMP:BTCUSD" not in symbols
        assert "TVC:VIX" not in symbols

    def test_none_market_spans_all_markets(self):
        symbols = _key_market_refresh_symbols(None, _norm)
        assert symbols["BTC-USD"] == "US"
        assert symbols["2800.HK"] == "HK"
        assert symbols["^N225"] == "JP"

    def test_unknown_market_is_empty(self):
        assert _key_market_refresh_symbols("XX", _norm) == {}


class TestSnapshotCacheHelpers:
    def setup_method(self):
        _clear_daily_snapshot_memory_cache()

    def teardown_method(self):
        _clear_daily_snapshot_memory_cache()

    def test_cache_key_is_scoped_to_market_and_scan_run(self):
        key = daily_snapshot_cache_key("us", "scan-abc")
        assert key == f"daily_snapshot:v{DAILY_SNAPSHOT_SCHEMA_VERSION}:US:scan-abc"
        # A newly published run switches the key, invalidating the old entry.
        assert daily_snapshot_cache_key("us", "scan-def") != key

    def test_cache_key_without_scan(self):
        key = daily_snapshot_cache_key("hk", None)
        assert key == f"daily_snapshot:v{DAILY_SNAPSHOT_SCHEMA_VERSION}:HK:no-scan"

    def test_etag_is_stable_and_weak(self):
        first = daily_snapshot_etag('{"a":1}')
        second = daily_snapshot_etag('{"a":1}')
        assert first == second
        assert first.startswith('W/"')
        assert daily_snapshot_etag('{"a":2}') != first

    def test_memory_cache_serves_payload_until_ttl(self):
        key = daily_snapshot_cache_key("us", "scan-abc")

        set_daily_snapshot_memory_cache(key, '{"ok":true}', ttl_seconds=10, now=100.0)

        assert get_daily_snapshot_memory_cache(key, now=109.0) == '{"ok":true}'
        assert get_daily_snapshot_memory_cache(key, now=110.0) is None

    def test_memory_cache_is_bounded(self):
        for index in range(DAILY_SNAPSHOT_MEMORY_CACHE_MAX_ENTRIES + 1):
            set_daily_snapshot_memory_cache(
                daily_snapshot_cache_key("us", f"scan-{index}"),
                f'{{"index":{index}}}',
                ttl_seconds=10_000,
                now=100.0,
            )

        assert get_daily_snapshot_memory_cache(
            daily_snapshot_cache_key("us", "scan-0"),
            now=100.0,
        ) is None
        assert get_daily_snapshot_memory_cache(
            daily_snapshot_cache_key("us", f"scan-{DAILY_SNAPSHOT_MEMORY_CACHE_MAX_ENTRIES}"),
            now=100.0,
        ) == f'{{"index":{DAILY_SNAPSHOT_MEMORY_CACHE_MAX_ENTRIES}}}'

    def test_memory_cache_hit_does_not_touch_redis(self):
        key = daily_snapshot_cache_key("us", "scan-abc")
        set_daily_snapshot_memory_cache(key, '{"ok":true}', ttl_seconds=10, now=100.0)

        def fail_if_called():
            raise AssertionError("Redis should not be requested on a memory hit")

        def fail_if_built():
            raise AssertionError("Payload should not be rebuilt on a memory hit")

        payload = get_or_build_daily_snapshot_payload(
            key,
            fail_if_built,
            redis_client_factory=fail_if_called,
            now=101.0,
        )

        assert payload == '{"ok":true}'

    def test_redis_hit_warms_memory_for_remaining_redis_ttl(self):
        key = daily_snapshot_cache_key("us", "scan-abc")
        redis = FakeRedis(payload=b'{"ok":true}', ttl_seconds=4)

        def fail_if_built():
            raise AssertionError("Payload should not be rebuilt on a Redis hit")

        payload = get_or_build_daily_snapshot_payload(
            key,
            fail_if_built,
            redis_client_factory=lambda: redis,
            now=100.0,
        )

        assert payload == '{"ok":true}'
        assert redis.get_calls == [key]
        assert redis.ttl_calls == [key]
        assert get_daily_snapshot_memory_cache(key, now=103.0) == '{"ok":true}'
        assert get_daily_snapshot_memory_cache(key, now=104.0) is None

    def test_redis_hit_without_positive_ttl_rebuilds_payload(self):
        key = daily_snapshot_cache_key("us", "scan-abc")
        redis = FakeRedis(payload='{"ok":true}', ttl_seconds=-1)

        payload = get_or_build_daily_snapshot_payload(
            key,
            lambda: '{"rebuilt":true}',
            redis_client_factory=lambda: redis,
            now=100.0,
        )

        assert payload == '{"rebuilt":true}'
        assert redis.get_calls == [key]
        assert redis.ttl_calls == [key]
        assert redis.setex_calls == [(key, DAILY_SNAPSHOT_CACHE_TTL_SECONDS, '{"rebuilt":true}')]
        assert get_daily_snapshot_memory_cache(key, now=100.0) == '{"rebuilt":true}'

    def test_redis_hit_still_serves_payload_when_ttl_read_fails(self):
        key = daily_snapshot_cache_key("us", "scan-abc")
        redis = FakeRedis(payload='{"ok":true}', ttl_seconds=RuntimeError("ttl unavailable"))

        payload = get_or_build_daily_snapshot_payload(
            key,
            lambda: '{"rebuilt":true}',
            redis_client_factory=lambda: redis,
            now=100.0,
        )

        assert payload == '{"ok":true}'
        assert get_daily_snapshot_memory_cache(key, now=100.0) is None

    def test_redis_client_factory_failure_builds_payload_and_warms_memory(self):
        key = daily_snapshot_cache_key("us", "scan-abc")

        def fail_to_create_redis():
            raise RuntimeError("redis unavailable")

        payload = get_or_build_daily_snapshot_payload(
            key,
            lambda: '{"ok":true}',
            redis_client_factory=fail_to_create_redis,
            now=100.0,
        )

        assert payload == '{"ok":true}'
        assert get_daily_snapshot_memory_cache(key, now=100.0) == '{"ok":true}'

    def test_cache_miss_builds_payload_then_updates_redis_and_memory(self):
        key = daily_snapshot_cache_key("us", "scan-abc")
        redis = FakeRedis()

        payload = get_or_build_daily_snapshot_payload(
            key,
            lambda: '{"ok":true}',
            redis_client_factory=lambda: redis,
            now=100.0,
        )

        assert payload == '{"ok":true}'
        assert redis.get_calls == [key]
        assert redis.setex_calls == [(key, DAILY_SNAPSHOT_CACHE_TTL_SECONDS, '{"ok":true}')]
        assert get_daily_snapshot_memory_cache(
            key,
            now=100.0 + DAILY_SNAPSHOT_CACHE_TTL_SECONDS - 1,
        ) == '{"ok":true}'
        assert get_daily_snapshot_memory_cache(
            key,
            now=100.0 + DAILY_SNAPSHOT_CACHE_TTL_SECONDS,
        ) is None


class TestScanFreshness:
    def test_no_scan(self):
        freshness = _scan_freshness(None)
        assert freshness == {
            "scan_id": None,
            "scan_as_of_date": None,
            "scan_published_at": None,
        }

    def test_scan_with_feature_run(self):
        run = SimpleNamespace(
            as_of_date=date(2026, 6, 10),
            published_at=datetime(2026, 6, 10, 23, 0, tzinfo=timezone.utc),
        )
        scan = SimpleNamespace(
            scan_id="abc-123",
            feature_run=run,
            completed_at=datetime(2026, 6, 11, 1, 0, tzinfo=timezone.utc),
        )
        freshness = _scan_freshness(scan)
        assert freshness["scan_id"] == "abc-123"
        assert freshness["scan_as_of_date"] == "2026-06-10"
        assert freshness["scan_published_at"].startswith("2026-06-10T23:00")

    def test_scan_without_feature_run_falls_back_to_completed_at(self):
        scan = SimpleNamespace(
            scan_id="abc-456",
            feature_run=None,
            completed_at=datetime(2026, 6, 11, 1, 0, tzinfo=timezone.utc),
        )
        freshness = _scan_freshness(scan)
        assert freshness["scan_as_of_date"] == "2026-06-11"
        assert freshness["scan_published_at"].startswith("2026-06-11T01:00")


class TestIfNoneMatchMatching:
    ETAG = 'W/"abc123"'

    def test_exact_match(self):
        assert _if_none_match_matches('W/"abc123"', self.ETAG)

    def test_comma_separated_list(self):
        assert _if_none_match_matches('W/"other", W/"abc123"', self.ETAG)

    def test_weak_comparison_ignores_weak_prefix(self):
        # RFC 7232 If-None-Match uses weak comparison on both sides.
        assert _if_none_match_matches('"abc123"', self.ETAG)

    def test_star_matches_anything(self):
        assert _if_none_match_matches("*", self.ETAG)

    def test_no_match(self):
        assert not _if_none_match_matches('W/"other"', self.ETAG)
        assert not _if_none_match_matches("", self.ETAG)
        assert not _if_none_match_matches(None, self.ETAG)


class TestDailySnapshotResponseSchema:
    @staticmethod
    def _payload():
        return {
            "schema_version": DAILY_SNAPSHOT_SCHEMA_VERSION,
            "generated_at": "2026-06-12T00:00:00+00:00",
            "market": "US",
            "market_display_name": "United States",
            "scan_id": "scan-abc",
            "freshness": {
                "scan_id": "scan-abc",
                "scan_as_of_date": "2026-06-11",
                "scan_published_at": "2026-06-11T23:00:00+00:00",
                "breadth_latest_date": "2026-06-11",
                "groups_latest_date": "2026-06-11",
            },
            "key_markets": [
                {
                    "symbol": "SPY",
                    "display_name": "S&P 500",
                    "currency": "USD",
                    "latest_close": 500.0,
                    "latest_date": "2026-06-11",
                    "change_1d": 0.5,
                    "history": [
                        {"date": "2026-06-10", "close": 497.5},
                        {"date": "2026-06-11", "close": None},
                    ],
                },
            ],
            "top_candidates": {"min_dollar_volume": 100_000_000, "rows": []},
            "leaders": {
                "criteria": {
                    "max_group_rank": 40,
                    "min_rs_rating": 80,
                    "min_dollar_volume": 100_000_000,
                },
                "rows": [],
            },
            "top_groups": [
                {
                    "industry_group": "Semiconductors",
                    "rank": 1,
                    "rank_change_1w": 2,
                    "rank_change_1m": 5,
                    "top_symbol": "NVDA",
                    "top_symbol_name": "NVIDIA",
                    "top_rs_rating": 99,
                },
            ],
        }

    def test_accepts_the_documented_payload_shape(self):
        DailySnapshotResponse.model_validate(self._payload())

    def test_rejects_payload_drift(self):
        # extra="forbid" keeps the schema in lockstep with the builder: an
        # undeclared field fails on the cache-miss path instead of silently
        # widening the public contract.
        payload = self._payload()
        payload["surprise_field"] = True
        with pytest.raises(ValidationError):
            DailySnapshotResponse.model_validate(payload)

    def test_rejects_nested_drift(self):
        payload = self._payload()
        payload["freshness"]["surprise_field"] = True
        with pytest.raises(ValidationError):
            DailySnapshotResponse.model_validate(payload)
