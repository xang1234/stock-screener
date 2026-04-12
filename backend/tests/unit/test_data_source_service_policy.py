"""Integration tests proving DataSourceService honours the routing policy.

These tests exist to lock in the acceptance criterion for
``StockScreenClaude-asia.5.1``: non-US requests must not attempt
unsupported provider calls.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.services.data_source_service import DataSourceService


def _make_service(finviz_return=None, yfinance_return=None) -> DataSourceService:
    """Build a DataSourceService with fully mocked provider dependencies."""
    finviz = MagicMock()
    finviz.get_fundamentals.return_value = finviz_return
    finviz.get_quarterly_growth.return_value = finviz_return
    yfinance = MagicMock()
    yfinance.get_fundamentals.return_value = yfinance_return or {
        "market_cap": 1_000,
        "pe_ratio": 10.0,
    }
    yfinance.get_quarterly_growth.return_value = yfinance_return or {
        "eps_growth_qq": 0.1,
    }
    eps_rating = MagicMock()
    rate_limiter = MagicMock()
    return DataSourceService(
        finviz_service=finviz,
        yfinance_service=yfinance,
        eps_rating_service=eps_rating,
        rate_limiter=rate_limiter,
        prefer_finviz=True,
    )


class TestGetFundamentalsPolicyGate:
    """Acceptance: non-US markets skip finviz, go straight to yfinance."""

    @pytest.mark.parametrize("market", ["HK", "JP", "TW"])
    def test_non_us_market_skips_finviz(self, market):
        svc = _make_service()
        result = svc.get_fundamentals("0700.HK", market=market)

        svc.finviz_service.get_fundamentals.assert_not_called()
        svc.yfinance_service.get_fundamentals.assert_called_once_with("0700.HK")
        assert result is not None
        assert result["data_source"] == "yfinance"
        assert svc.metrics["finviz_skipped_by_policy"] == 1
        assert svc.metrics["finviz_success"] == 0

    def test_us_market_still_prefers_finviz(self):
        svc = _make_service(finviz_return={"market_cap": 2_000})
        svc.get_fundamentals("AAPL", market="US")
        svc.finviz_service.get_fundamentals.assert_called_once_with("AAPL")
        assert svc.metrics["finviz_skipped_by_policy"] == 0

    def test_none_market_defaults_to_us_behaviour(self):
        svc = _make_service(finviz_return={"market_cap": 2_000})
        svc.get_fundamentals("AAPL")  # no market kwarg
        svc.finviz_service.get_fundamentals.assert_called_once_with("AAPL")
        assert svc.metrics["finviz_skipped_by_policy"] == 0


class TestGetQuarterlyGrowthPolicyGate:
    @pytest.mark.parametrize("market", ["HK", "JP", "TW"])
    def test_non_us_market_skips_finviz_growth(self, market):
        svc = _make_service()
        result = svc.get_quarterly_growth("7203.T", market=market)
        svc.finviz_service.get_quarterly_growth.assert_not_called()
        svc.yfinance_service.get_quarterly_growth.assert_called_once_with("7203.T")
        assert result is not None
        assert result["data_source"] == "yfinance"


class TestGetCombinedDataPolicyGate:
    @pytest.mark.parametrize("market", ["HK", "JP", "TW"])
    def test_non_us_market_skips_finviz_combined(self, market):
        svc = _make_service()
        result = svc.get_combined_data("2330.TW", market=market)
        svc.finviz_service.get_combined_data.assert_not_called()
        svc.yfinance_service.get_fundamentals.assert_called_once_with("2330.TW")
        svc.yfinance_service.get_quarterly_growth.assert_called_once_with("2330.TW")
        assert result is not None


class TestMetricsIncludePolicySkipCounter:
    """The policy metric is exposed so operators can observe impact."""

    def test_metrics_dict_contains_policy_skip_counter(self):
        svc = _make_service()
        assert "finviz_skipped_by_policy" in svc.get_metrics()
