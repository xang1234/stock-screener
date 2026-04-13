"""Integration tests: custom scanner honors mixed-market policy (T6.3).

Drives CustomScanner.scan_stock with synthetic StockData to verify that
cap / volume filters switch to USD-normalized columns when the scan
spans multiple markets, and fail closed when FX normalization is
missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.scanners.base_screener import StockData
from app.scanners.custom_scanner import CustomScanner


def _price_frame(days: int = 260, close: float = 100.0, volume: int = 600_000) -> pd.DataFrame:
    """Build a deterministic OHLCV frame wide enough for MA/52w logic."""
    idx = pd.date_range("2024-01-01", periods=days, freq="B")
    closes = np.full(days, close, dtype=float)
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes * 1.01,
            "Low": closes * 0.99,
            "Close": closes,
            "Volume": np.full(days, volume, dtype=float),
        },
        index=idx,
    )


def _make_stock_data(
    *,
    symbol: str,
    market: str,
    is_mixed: bool,
    fundamentals: dict,
    avg_volume_shares: int = 600_000,
) -> StockData:
    price = _price_frame(volume=avg_volume_shares)
    return StockData(
        symbol=symbol,
        price_data=price,
        benchmark_data=pd.DataFrame(),  # no RS filter in these tests
        fundamentals=fundamentals,
        market=market,
        is_mixed_market=is_mixed,
    )


class TestMarketCapPolicy:
    def test_single_market_uses_native_cap(self):
        """HK single-market scan: threshold interpreted as HKD against native cap."""
        scanner = CustomScanner()
        data = _make_stock_data(
            symbol="0700.HK",
            market="HK",
            is_mixed=False,
            fundamentals={"market_cap": 1_000_000_000, "market_cap_usd": 128_000_000},
        )
        # Threshold 500M HKD — native cap 1B HKD passes.
        result = scanner.scan_stock("0700.HK", data, {"custom_filters": {"market_cap_min": 500_000_000}})
        assert result.details["filter_results"]["market_cap"]["passes"] is True
        assert result.details["filter_results"]["market_cap"]["unit"] == "native"
        assert result.details["filter_results"]["market_cap"]["market_cap"] == 1_000_000_000

    def test_mixed_market_uses_usd_cap(self):
        """HK row in mixed-market scan: $200M threshold compared to USD cap $128M → fail."""
        scanner = CustomScanner()
        data = _make_stock_data(
            symbol="0700.HK",
            market="HK",
            is_mixed=True,
            fundamentals={"market_cap": 1_000_000_000, "market_cap_usd": 128_000_000},
        )
        result = scanner.scan_stock("0700.HK", data, {"custom_filters": {"market_cap_min": 200_000_000}})
        cap_result = result.details["filter_results"]["market_cap"]
        assert cap_result["passes"] is False
        assert cap_result["unit"] == "usd"
        assert cap_result["market_cap"] == 128_000_000

    def test_mixed_market_missing_usd_fails_closed(self):
        """No FX data → mixed-market cap filter fails the row rather than using HKD."""
        scanner = CustomScanner()
        data = _make_stock_data(
            symbol="0700.HK",
            market="HK",
            is_mixed=True,
            fundamentals={"market_cap": 1_000_000_000, "market_cap_usd": None},
        )
        result = scanner.scan_stock("0700.HK", data, {"custom_filters": {"market_cap_min": 1}})
        cap_result = result.details["filter_results"]["market_cap"]
        assert cap_result["passes"] is False
        assert cap_result["reason"] == "missing_market_cap_usd"


class TestVolumePolicy:
    def test_single_market_uses_share_volume(self):
        scanner = CustomScanner()
        data = _make_stock_data(
            symbol="AAPL",
            market="US",
            is_mixed=False,
            fundamentals={"adv_usd": 1},  # ignored in single-market mode
            avg_volume_shares=600_000,
        )
        result = scanner.scan_stock("AAPL", data, {"custom_filters": {"volume_min": 500_000}})
        vol = result.details["filter_results"]["volume"]
        assert vol["passes"] is True
        assert vol["unit"] == "shares"
        assert vol["avg_volume"] == 600_000

    def test_mixed_market_uses_adv_usd(self):
        """Mixed scan: $5M USD threshold against adv_usd $6.4M → pass, regardless of share volume."""
        scanner = CustomScanner()
        data = _make_stock_data(
            symbol="0700.HK",
            market="HK",
            is_mixed=True,
            fundamentals={"adv_usd": 6_400_000},
            avg_volume_shares=500_000,  # native shares — should not be used
        )
        result = scanner.scan_stock("0700.HK", data, {"custom_filters": {"volume_min": 5_000_000}})
        vol = result.details["filter_results"]["volume"]
        assert vol["passes"] is True
        assert vol["unit"] == "usd"
        assert vol["avg_volume"] == 6_400_000

    def test_mixed_market_missing_adv_usd_fails_closed(self):
        scanner = CustomScanner()
        data = _make_stock_data(
            symbol="0700.HK",
            market="HK",
            is_mixed=True,
            fundamentals={"adv_usd": None},
        )
        result = scanner.scan_stock("0700.HK", data, {"custom_filters": {"volume_min": 1}})
        vol = result.details["filter_results"]["volume"]
        assert vol["passes"] is False
        assert vol["reason"] == "missing_adv_usd"


class TestDataPreparationSetsFlag:
    """Confirm the prep layer attaches the flag once per scan."""

    def test_attach_sets_is_mixed_market_true_for_cross_market(self):
        from app.scanners.data_preparation import DataPreparationLayer

        # Stub-out dependencies; we only exercise the attach method.
        prep = DataPreparationLayer.__new__(DataPreparationLayer)
        results = {
            "AAPL": StockData(
                symbol="AAPL",
                price_data=pd.DataFrame(),
                benchmark_data=pd.DataFrame(),
                market="US",
            ),
            "0700.HK": StockData(
                symbol="0700.HK",
                price_data=pd.DataFrame(),
                benchmark_data=pd.DataFrame(),
                market="HK",
            ),
        }
        # Flag is set by _detect_and_set_mixed_market_flag (called unconditionally).
        prep._detect_and_set_mixed_market_flag(results)
        assert results["AAPL"].is_mixed_market is True
        assert results["0700.HK"].is_mixed_market is True

    def test_attach_sets_is_mixed_market_false_for_single_market(self):
        from app.scanners.data_preparation import DataPreparationLayer

        prep = DataPreparationLayer.__new__(DataPreparationLayer)
        results = {
            "AAPL": StockData(
                symbol="AAPL",
                price_data=pd.DataFrame(),
                benchmark_data=pd.DataFrame(),
                market="US",
            ),
            "MSFT": StockData(
                symbol="MSFT",
                price_data=pd.DataFrame(),
                benchmark_data=pd.DataFrame(),
                market=None,  # legacy — treated as US
            ),
        }
        prep._detect_and_set_mixed_market_flag(results)
        assert results["AAPL"].is_mixed_market is False
        assert results["MSFT"].is_mixed_market is False
