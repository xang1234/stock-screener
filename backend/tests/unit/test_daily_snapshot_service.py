"""Unit tests for the aggregated Daily Snapshot service helpers."""

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.api.v1.market_scan import _if_none_match_matches
from app.schemas.market_scan import DailySnapshotResponse
from app.services.daily_snapshot_service import (
    DAILY_SNAPSHOT_SCHEMA_VERSION,
    _scan_freshness,
    daily_snapshot_cache_key,
    daily_snapshot_etag,
)
from app.services.price_refresh_plan_builder import _key_market_refresh_symbols


def _norm(market):
    return str(market).upper()


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
