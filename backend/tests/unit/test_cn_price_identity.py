from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pandas as pd

import app.services.cn_market_data_service as cn_market_data_service
from app.services.bulk_data_fetcher import BulkDataFetcher
from app.services.price_cache_service import PriceCacheService


def _price_df(day: date, close: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [close - 1],
            "High": [close + 1],
            "Low": [close - 2],
            "Close": [close],
            "Adj Close": [close - 0.5],
            "Volume": [1_000_000],
        },
        index=pd.to_datetime([day]),
    )


def test_bulk_cn_price_fetch_preserves_explicit_suffix(monkeypatch):
    fetcher = BulkDataFetcher()
    calls = []

    class FakeCnPriceService:
        @staticmethod
        def daily_ohlcv_dataframe(symbol, *, period):
            calls.append({"symbol": symbol, "period": period})
            return _price_df(date(2026, 4, 29), 3333.0)

    monkeypatch.setattr(fetcher, "_get_cn_price_service", lambda: FakeCnPriceService())

    results = fetcher._fetch_cn_price_batch(["000001.SS"], period="7d")

    assert calls == [{"symbol": "000001.SS", "period": "7d"}]
    assert results["000001.SS"]["has_error"] is False
    assert results["000001.SS"]["price_data"].iloc[-1]["Close"] == 3333.0


def test_bulk_cn_price_fetch_preserves_bare_local_code(monkeypatch):
    fetcher = BulkDataFetcher()
    calls = []

    class FakeCnPriceService:
        @staticmethod
        def daily_ohlcv_dataframe(symbol, *, period):
            calls.append({"symbol": symbol, "period": period})
            return _price_df(date(2026, 4, 29), 12.0)

    monkeypatch.setattr(fetcher, "_get_cn_price_service", lambda: FakeCnPriceService())

    results = fetcher._fetch_cn_price_batch(["000001"], period="7d")

    assert calls == [{"symbol": "000001", "period": "7d"}]
    assert results["000001"]["has_error"] is False
    assert results["000001"]["price_data"].iloc[-1]["Close"] == 12.0


def test_price_cache_cn_direct_fetch_preserves_explicit_suffix(monkeypatch):
    service = PriceCacheService(redis_client=None, session_factory=MagicMock())
    calls = []

    class FakeCnMarketDataService:
        @staticmethod
        def daily_ohlcv_dataframe(symbol, *, period):
            calls.append({"symbol": symbol, "period": period})
            return _price_df(date(2026, 4, 29), 3333.0)

    monkeypatch.setattr(
        cn_market_data_service,
        "CnMarketDataService",
        lambda: FakeCnMarketDataService(),
    )

    result = service._fetch_cn_historical_data("000001.SS", period="7d")

    assert calls == [{"symbol": "000001.SS", "period": "7d"}]
    assert result is not None
    assert result.iloc[-1]["Close"] == 3333.0


def test_price_cache_cn_direct_fetch_preserves_bare_local_code(monkeypatch):
    service = PriceCacheService(redis_client=None, session_factory=MagicMock())
    calls = []

    class FakeCnMarketDataService:
        @staticmethod
        def daily_ohlcv_dataframe(symbol, *, period):
            calls.append({"symbol": symbol, "period": period})
            return _price_df(date(2026, 4, 29), 12.0)

    monkeypatch.setattr(
        cn_market_data_service,
        "CnMarketDataService",
        lambda: FakeCnMarketDataService(),
    )

    result = service._fetch_cn_historical_data("000001", period="7d")

    assert calls == [{"symbol": "000001", "period": "7d"}]
    assert result is not None
    assert result.iloc[-1]["Close"] == 12.0
