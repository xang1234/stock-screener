"""Unit tests for the per-market rate-budget policy (bead asia.9.2)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.services.rate_budget_policy import RateBudgetPolicy


class TestProviderKey:
    @pytest.mark.parametrize("provider,market,expected", [
        ("yfinance", "US", "yfinance:us"),
        ("yfinance", "HK", "yfinance:hk"),
        ("yfinance", "JP", "yfinance:jp"),
        ("yfinance", "TW", "yfinance:tw"),
        ("yfinance", None, "yfinance:shared"),
        ("yfinance", "shared", "yfinance:shared"),
        ("finviz", "HK", "finviz:hk"),
        ("yfinance:batch", "US", "yfinance:batch:us"),
    ])
    def test_provider_key(self, provider, market, expected):
        assert RateBudgetPolicy.provider_key(provider, market) == expected

    def test_unknown_market_raises(self):
        with pytest.raises(ValueError):
            RateBudgetPolicy.provider_key("yfinance", "CN")


class TestUniverseWeights:
    def test_equal_split_when_universe_empty(self):
        """Fresh deploy / no universe rows -> equal split across SUPPORTED_MARKETS."""
        with patch("app.services.rate_budget_policy.RateBudgetPolicy._compute_weights_from_db") as m:
            m.return_value = {"US": 0.25, "HK": 0.25, "JP": 0.25, "TW": 0.25}
            policy = RateBudgetPolicy()
            weights = policy._universe_weights(force_refresh=True)
        assert weights == {"US": 0.25, "HK": 0.25, "JP": 0.25, "TW": 0.25}
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_universe_weighted_split_normalizes(self):
        with patch("app.services.rate_budget_policy.RateBudgetPolicy._compute_weights_from_db") as m:
            # 6000 US, 2500 HK, 1000 JP, 500 TW = 10000 total
            m.return_value = {"US": 0.6, "HK": 0.25, "JP": 0.10, "TW": 0.05}
            policy = RateBudgetPolicy()
            weights = policy._universe_weights(force_refresh=True)
        assert sum(weights.values()) == pytest.approx(1.0)
        assert weights["US"] > weights["HK"] > weights["JP"]
        assert weights["TW"] >= 0.05

    def test_invalidate_cache_forces_recompute(self):
        with patch("app.services.rate_budget_policy.RateBudgetPolicy._compute_weights_from_db") as m:
            m.return_value = {"US": 0.4, "HK": 0.3, "JP": 0.2, "TW": 0.1}
            policy = RateBudgetPolicy()
            policy._universe_weights(force_refresh=True)
            policy._universe_weights()  # cached
            assert m.call_count == 1
            policy.invalidate_weights_cache()
            policy._universe_weights()  # recomputed
            assert m.call_count == 2


class TestRateInterval:
    def test_universe_weighted_default_inversely_scales(self):
        """Bigger universe weight -> smaller interval (gets more tokens/sec)."""
        with patch("app.services.rate_budget_policy.RateBudgetPolicy._compute_weights_from_db") as m:
            m.return_value = {"US": 0.6, "HK": 0.20, "JP": 0.10, "TW": 0.10}
            policy = RateBudgetPolicy()
            # global yfinance interval = 1.0s (yfinance_rate_limit=1 req/s)
            us_interval = policy.get_rate_interval("yfinance", "US")
            hk_interval = policy.get_rate_interval("yfinance", "HK")
        assert us_interval < hk_interval, "US should wait less than HK (bigger universe)"

    def test_per_market_override_wins(self):
        """Setting yfinance_rate_limit_hk overrides the universe-weighted default."""
        with patch("app.services.rate_budget_policy.settings") as mock_settings:
            mock_settings.yfinance_rate_limit = 1
            mock_settings.yfinance_rate_limit_hk = 2.0  # override: 2 req/s = 0.5s interval
            policy = RateBudgetPolicy()
            interval = policy.get_rate_interval("yfinance", "HK")
        assert interval == pytest.approx(0.5)

    def test_shared_market_uses_global_interval(self):
        with patch("app.services.rate_budget_policy.settings") as mock_settings:
            mock_settings.yfinance_rate_limit = 1
            policy = RateBudgetPolicy()
            interval = policy.get_rate_interval("yfinance", None)
        assert interval == pytest.approx(1.0)


class TestBatchSize:
    @pytest.mark.parametrize("market,expected_default", [
        ("US", 50), ("HK", 50), ("JP", 50), ("TW", 50),
    ])
    def test_default_batch_sizes(self, market, expected_default):
        with patch("app.services.rate_budget_policy.settings") as mock_settings:
            # No override: getattr returns None -> use built-in default
            mock_settings.configure_mock(**{f"yfinance_batch_size_{market.lower()}": None})
            policy = RateBudgetPolicy()
            assert policy.get_batch_size("yfinance", market) == expected_default

    def test_override_wins(self):
        with patch("app.services.rate_budget_policy.settings") as mock_settings:
            mock_settings.yfinance_batch_size_hk = 75
            policy = RateBudgetPolicy()
            assert policy.get_batch_size("yfinance", "HK") == 75


class TestBackoffParams:
    def test_default_backoff_per_market(self):
        with patch("app.services.rate_budget_policy.settings") as mock_settings:
            mock_settings.yfinance_backoff_max_s_us = None
            mock_settings.yfinance_backoff_max_s_hk = None
            policy = RateBudgetPolicy()
            us = policy.get_backoff_params("yfinance", "US")
            hk = policy.get_backoff_params("yfinance", "HK")
        assert us["max_s"] == 480
        assert hk["max_s"] == 600  # non-US gets longer cap

    def test_max_s_override_wins(self):
        with patch("app.services.rate_budget_policy.settings") as mock_settings:
            mock_settings.yfinance_backoff_max_s_hk = 900
            policy = RateBudgetPolicy()
            params = policy.get_backoff_params("yfinance", "HK")
        assert params["max_s"] == 900
        # Other defaults still present
        assert "base_s" in params and "factor" in params


class TestThrottleCounters:
    def _policy_with_redis(self, mock_client):
        return RateBudgetPolicy(redis_client_factory=lambda: mock_client)

    def test_record_429_increments_running_and_daily(self):
        mock_client = MagicMock()
        pipe = MagicMock()
        mock_client.pipeline.return_value = pipe
        policy = self._policy_with_redis(mock_client)

        policy.record_429("yfinance", "HK")

        pipe.incr.assert_any_call("ratelimit:429:yfinance:hk")
        # Daily-bucketed key has YYYYMMDD suffix; just assert it was incremented
        daily_calls = [c for c in pipe.incr.call_args_list if "ratelimit:429:yfinance:hk:" in c.args[0]]
        assert len(daily_calls) == 1
        pipe.expire.assert_called_once()
        pipe.execute.assert_called_once()

    def test_record_throttle_wait_skips_zero(self):
        mock_client = MagicMock()
        policy = self._policy_with_redis(mock_client)
        policy.record_throttle_wait("yfinance", "US", wait_s=0.0)
        mock_client.pipeline.assert_not_called()

    def test_record_throttle_wait_increments_count_and_seconds(self):
        mock_client = MagicMock()
        pipe = MagicMock()
        mock_client.pipeline.return_value = pipe
        policy = self._policy_with_redis(mock_client)

        policy.record_throttle_wait("yfinance", "JP", wait_s=2.5)

        pipe.incr.assert_called_once()
        pipe.incrbyfloat.assert_called_once()
        # incrbyfloat called with (key, 2.5)
        assert pipe.incrbyfloat.call_args.args[1] == pytest.approx(2.5)

    def test_records_silently_when_redis_unavailable(self):
        policy = RateBudgetPolicy(redis_client_factory=lambda: None)
        # Should not raise
        policy.record_429("yfinance", "US")
        policy.record_throttle_wait("yfinance", "US", wait_s=1.0)
