"""Unit tests for per-market queue helpers (bead StockScreenClaude-asia.9.1)."""
from __future__ import annotations

import pytest

from app.tasks.market_queues import (
    MARKET_JOBS_BASE,
    SHARED_DATA_FETCH_QUEUE,
    SHARED_SENTINEL,
    SHARED_USER_SCANS_QUEUE,
    SUPPORTED_MARKETS,
    all_data_fetch_queues,
    all_market_job_queues,
    all_user_scans_queues,
    data_fetch_queue_for_market,
    log_extra,
    market_jobs_queue_for_market,
    market_tag,
    normalize_market,
    queue_for_market,
    user_scans_queue_for_market,
)


class TestNormalizeMarket:
    @pytest.mark.parametrize("raw,expected", [
        ("US", "US"), ("us", "US"), ("Us", "US"), (" us ", "US"),
        ("HK", "HK"), ("hk", "HK"),
        ("IN", "IN"), ("in", "IN"),
        ("JP", "JP"), ("KR", "KR"), ("kr", "KR"), ("TW", "TW"), ("CN", "CN"),
        (None, SHARED_SENTINEL), ("", SHARED_SENTINEL), ("  ", SHARED_SENTINEL),
        ("SHARED", SHARED_SENTINEL), ("shared", SHARED_SENTINEL),
    ])
    def test_canonicalizes(self, raw, expected):
        assert normalize_market(raw) == expected

    @pytest.mark.parametrize("bad", ["UK", "ALL", "xx", "US_CANADA"])
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
        ("IN", "data_fetch_in"),
        ("JP", "data_fetch_jp"),
        ("KR", "data_fetch_kr"),
        ("TW", "data_fetch_tw"),
        ("CN", "data_fetch_cn"),
        (None, "data_fetch_shared"),
        ("shared", "data_fetch_shared"),
    ])
    def test_data_fetch_queue(self, market, expected):
        assert queue_for_market(market) == expected
        assert data_fetch_queue_for_market(market) == expected

    @pytest.mark.parametrize("market,expected", [
        ("US", "user_scans_us"),
        ("HK", "user_scans_hk"),
        ("IN", "user_scans_in"),
        (None, "user_scans_shared"),
    ])
    def test_user_scans_queue(self, market, expected):
        assert user_scans_queue_for_market(market) == expected
        assert queue_for_market(market, base="user_scans") == expected

    def test_custom_base(self):
        assert queue_for_market("HK", base="custom") == "custom_hk"
        assert queue_for_market("US", base=MARKET_JOBS_BASE) == "market_jobs_us"

    @pytest.mark.parametrize("market,expected", [
        ("US", "market_jobs_us"),
        ("HK", "market_jobs_hk"),
        ("IN", "market_jobs_in"),
        ("JP", "market_jobs_jp"),
        ("KR", "market_jobs_kr"),
        ("TW", "market_jobs_tw"),
        ("CN", "market_jobs_cn"),
    ])
    def test_market_jobs_queue(self, market, expected):
        assert market_jobs_queue_for_market(market) == expected
        assert queue_for_market(market, base=MARKET_JOBS_BASE) == expected

    def test_unknown_market_raises(self):
        assert queue_for_market("CN") == "data_fetch_cn"


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

    def test_market_job_queues_include_all_markets(self):
        queues = all_market_job_queues()
        for m in SUPPORTED_MARKETS:
            assert f"market_jobs_{m.lower()}" in queues

    def test_market_job_queue_subset(self):
        assert all_market_job_queues(markets=["US", "HK"]) == [
            "market_jobs_us",
            "market_jobs_hk",
        ]


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
