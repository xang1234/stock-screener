"""Per-market rate-limiter integration (bead asia.9.2).

Verifies that ``RedisRateLimiter.wait_for_market`` resolves the right key
via RateBudgetPolicy and records throttle telemetry when it actually slept.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.services.rate_limiter import RedisRateLimiter


class TestWaitForMarket:
    @patch("app.services.rate_limiter.settings")
    def test_resolves_per_market_key_and_interval(self, mock_settings):
        mock_settings.redis_enabled = False  # force fallback path
        with patch("app.services.rate_budget_policy.get_rate_budget_policy") as get_policy:
            mock_policy = MagicMock()
            mock_policy.provider_key.return_value = "yfinance:hk"
            mock_policy.get_rate_interval.return_value = 0.0  # no wait
            get_policy.return_value = mock_policy

            limiter = RedisRateLimiter()
            with patch.object(limiter, "wait", return_value=0.0) as inner:
                limiter.wait_for_market("yfinance", "HK")

            inner.assert_called_once()
            assert inner.call_args.args[0] == "yfinance:hk"
            assert inner.call_args.kwargs["min_interval_s"] == 0.0
            mock_policy.provider_key.assert_called_with("yfinance", "HK")

    @patch("app.services.rate_limiter.settings")
    def test_records_throttle_when_actually_waited(self, mock_settings):
        mock_settings.redis_enabled = False
        with patch("app.services.rate_budget_policy.get_rate_budget_policy") as get_policy:
            mock_policy = MagicMock()
            mock_policy.provider_key.return_value = "yfinance:us"
            mock_policy.get_rate_interval.return_value = 1.0
            get_policy.return_value = mock_policy

            limiter = RedisRateLimiter()
            with patch.object(limiter, "wait", return_value=0.5):
                limiter.wait_for_market("yfinance", "US")

            mock_policy.record_throttle_wait.assert_called_once_with("yfinance", "US", 0.5)

    @patch("app.services.rate_limiter.settings")
    def test_no_throttle_recording_when_zero_wait(self, mock_settings):
        mock_settings.redis_enabled = False
        with patch("app.services.rate_budget_policy.get_rate_budget_policy") as get_policy:
            mock_policy = MagicMock()
            mock_policy.provider_key.return_value = "yfinance:us"
            mock_policy.get_rate_interval.return_value = 0.0
            get_policy.return_value = mock_policy

            limiter = RedisRateLimiter()
            with patch.object(limiter, "wait", return_value=0.0):
                limiter.wait_for_market("yfinance", "US")

            mock_policy.record_throttle_wait.assert_not_called()


class TestWaitKeyIsolation:
    """Two markets calling wait_for_market hit different Redis keys."""

    @patch("app.services.rate_limiter.settings")
    def test_us_and_hk_use_different_keys(self, mock_settings):
        mock_settings.redis_enabled = False
        with patch("app.services.rate_budget_policy.get_rate_budget_policy") as get_policy:
            real_policy = MagicMock()
            # Simulate real provider_key behavior
            real_policy.provider_key.side_effect = lambda p, m: f"{p}:{m.lower()}" if m else f"{p}:shared"
            real_policy.get_rate_interval.return_value = 0.0
            get_policy.return_value = real_policy

            limiter = RedisRateLimiter()
            keys_seen = []
            with patch.object(limiter, "wait", side_effect=lambda key, **kw: keys_seen.append(key) or 0.0):
                limiter.wait_for_market("yfinance", "US")
                limiter.wait_for_market("yfinance", "HK")

        assert keys_seen == ["yfinance:us", "yfinance:hk"]
