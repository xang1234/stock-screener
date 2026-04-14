"""Unit tests for per-market queue helpers (bead StockScreenClaude-asia.9.1)."""
from __future__ import annotations

import pytest

from app.tasks.market_queues import (
    SHARED_DATA_FETCH_QUEUE,
    SHARED_SENTINEL,
    SHARED_USER_SCANS_QUEUE,
    SUPPORTED_MARKETS,
    all_data_fetch_queues,
    all_user_scans_queues,
    data_fetch_queue_for_market,
    log_extra,
    market_tag,
    normalize_market,
    queue_for_market,
    user_scans_queue_for_market,
)


class TestNormalizeMarket:
    @pytest.mark.parametrize("raw,expected", [
        ("US", "US"), ("us", "US"), ("Us", "US"), (" us ", "US"),
        ("HK", "HK"), ("hk", "HK"),
        ("JP", "JP"), ("TW", "TW"),
        (None, SHARED_SENTINEL), ("", SHARED_SENTINEL), ("  ", SHARED_SENTINEL),
        ("SHARED", SHARED_SENTINEL), ("shared", SHARED_SENTINEL),
    ])
    def test_canonicalizes(self, raw, expected):
        assert normalize_market(raw) == expected

    @pytest.mark.parametrize("bad", ["CN", "UK", "ALL", "xx", "US_CANADA"])
    def test_rejects_unknown(self, bad):
        with pytest.raises(ValueError):
            normalize_market(bad)

    def test_rejects_non_string(self):
        with pytest.raises(ValueError):
            normalize_market(123)  # type: ignore[arg-type]


class TestQueueForMarket:
    @pytest.mark.parametrize("market,expected", [
        ("US", "data_fetch_us"),
        ("HK", "data_fetch_hk"),
        ("JP", "data_fetch_jp"),
        ("TW", "data_fetch_tw"),
        (None, "data_fetch_shared"),
        ("shared", "data_fetch_shared"),
    ])
    def test_data_fetch_queue(self, market, expected):
        assert queue_for_market(market) == expected
        assert data_fetch_queue_for_market(market) == expected

    @pytest.mark.parametrize("market,expected", [
        ("US", "user_scans_us"),
        ("HK", "user_scans_hk"),
        (None, "user_scans_shared"),
    ])
    def test_user_scans_queue(self, market, expected):
        assert user_scans_queue_for_market(market) == expected
        assert queue_for_market(market, base="user_scans") == expected

    def test_custom_base(self):
        assert queue_for_market("HK", base="custom") == "custom_hk"

    def test_unknown_market_raises(self):
        with pytest.raises(ValueError):
            queue_for_market("CN")


class TestAllQueues:
    def test_data_fetch_queues_include_all_markets_and_shared(self):
        queues = all_data_fetch_queues()
        for m in SUPPORTED_MARKETS:
            assert f"data_fetch_{m.lower()}" in queues
        assert SHARED_DATA_FETCH_QUEUE in queues
        # Ensure shared only appears once even if market set excludes it.
        assert queues.count(SHARED_DATA_FETCH_QUEUE) == 1

    def test_user_scans_queues_include_all(self):
        queues = all_user_scans_queues()
        for m in SUPPORTED_MARKETS:
            assert f"user_scans_{m.lower()}" in queues
        assert SHARED_USER_SCANS_QUEUE in queues

    def test_custom_market_subset(self):
        queues = all_data_fetch_queues(markets=["US", "HK"])
        assert queues == ["data_fetch_us", "data_fetch_hk", SHARED_DATA_FETCH_QUEUE]


class TestLoggingHelpers:
    def test_log_extra(self):
        assert log_extra("HK") == {"market": "hk"}
        assert log_extra(None) == {"market": "shared"}
        assert log_extra("us") == {"market": "us"}

    def test_market_tag(self):
        assert market_tag("HK") == "[market=hk]"
        assert market_tag(None) == "[market=shared]"
        assert market_tag("US") == "[market=us]"

    def test_log_extra_rejects_unknown(self):
        with pytest.raises(ValueError):
            log_extra("UK")
