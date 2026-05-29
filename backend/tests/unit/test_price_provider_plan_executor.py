from __future__ import annotations

from datetime import date

import pandas as pd

from app.domain.providers.data_plan import (
    DATASET_PRICES,
    PROVIDER_YFINANCE,
    ProviderDataPlan,
    ProviderPlanStep,
)
from app.services.provider_adapters.price_plan_executor import PriceProviderPlanExecutor


def _price_df(day: date, close: float) -> pd.DataFrame:
    return pd.DataFrame({"Close": [close]}, index=pd.to_datetime([day]))


class _FakeFetcher:
    def __init__(self) -> None:
        self.yfinance_calls = []
        self.krx_calls = []
        self.cn_calls = []

    @staticmethod
    def _build_error_result(symbol: str, error: str) -> dict:
        return {
            "symbol": symbol,
            "price_data": None,
            "info": None,
            "fundamentals": None,
            "has_error": True,
            "error": error,
        }

    def _fetch_yfinance_prices_in_batches(
        self,
        symbols,
        *,
        period,
        start_batch_size=None,
        market=None,
    ):
        self.yfinance_calls.append(
            {
                "symbols": list(symbols),
                "period": period,
                "start_batch_size": start_batch_size,
                "market": market,
            }
        )
        return {
            symbol: {
                "symbol": symbol,
                "price_data": _price_df(date(2026, 5, 29), 100.0),
                "info": None,
                "fundamentals": None,
                "has_error": False,
                "error": None,
            }
            for symbol in symbols
        }

    def _fetch_kr_price_batch(self, symbols, *, period):
        self.krx_calls.append({"symbols": list(symbols), "period": period})
        return {
            symbol: self._build_error_result(symbol, "KRX returned empty price data")
            for symbol in symbols
        }

    def _fetch_cn_price_batch(self, symbols, *, period):
        self.cn_calls.append({"symbols": list(symbols), "period": period})
        return {
            symbol: self._build_error_result(
                symbol,
                "CN providers returned empty price data",
            )
            for symbol in symbols
        }


def test_executor_routes_yfinance_step_and_attaches_plan_metadata() -> None:
    fetcher = _FakeFetcher()
    plan = ProviderDataPlan(
        market="KR",
        dataset=DATASET_PRICES,
        steps=(ProviderPlanStep(PROVIDER_YFINANCE, batch_size=37),),
        version="test-plan",
    )
    executor = PriceProviderPlanExecutor(
        fetcher,
        plan_resolver=lambda market, mic=None: plan,
    )

    results = executor.fetch(["005930.KS"], period="7d", market="KR")

    assert fetcher.krx_calls == []
    assert fetcher.yfinance_calls == [
        {
            "symbols": ["005930.KS"],
            "period": "7d",
            "start_batch_size": 37,
            "market": "KR",
        }
    ]
    assert results["005930.KS"]["provider_data_plan"] == plan.provenance_metadata()


def test_executor_uses_security_master_mic_override_for_cn_bjse_fallback() -> None:
    fetcher = _FakeFetcher()
    executor = PriceProviderPlanExecutor(fetcher)

    results = executor.fetch(["920118.BJ"], period="7d", market="CN")

    assert fetcher.cn_calls == [{"symbols": ["920118.BJ"], "period": "7d"}]
    assert fetcher.yfinance_calls == []
    assert results["920118.BJ"]["provider_data_plan"]["mic"] == "XBSE"
    assert results["920118.BJ"]["provider_data_plan"]["providers"] == [
        "akshare",
        "baostock",
    ]
