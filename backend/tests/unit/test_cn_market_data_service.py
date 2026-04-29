from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.services.cn_market_data_service import (
    CnDependencyError,
    CnMarketDataService,
)


def test_cn_market_data_service_maps_akshare_spot_rows_to_listing_rows():
    frame = pd.DataFrame(
        [
            {
                "代码": "600519",
                "名称": "贵州茅台",
                "总市值": 2_500_000_000_000,
                "流通市值": 2_400_000_000_000,
                "市盈率-动态": 28.4,
                "市净率": 8.1,
                "所属行业": "酿酒行业",
            },
            {
                "代码": "000001",
                "名称": "平安银行",
                "总市值": 210_000_000_000,
                "所属行业": "银行",
            },
            {
                "代码": "920118",
                "名称": "太湖雪",
                "所属行业": "纺织服装",
            },
        ]
    )

    class FakeAkshare:
        @staticmethod
        def stock_zh_a_spot_em():
            return frame

    service = CnMarketDataService(akshare_module=FakeAkshare())

    rows = service.listing_rows(as_of=date(2026, 4, 30))

    assert [row["symbol"] for row in rows] == ["600519.SS", "000001.SZ", "920118.BJ"]
    assert rows[0]["exchange"] == "SSE"
    assert rows[0]["board"] == "SSE_MAIN"
    assert rows[0]["industry"] == "酿酒行业"
    assert rows[0]["pe_ratio"] == 28.4
    assert rows[1]["exchange"] == "SZSE"
    assert rows[2]["exchange"] == "BSE"


def test_cn_market_data_service_maps_akshare_ohlcv_to_yfinance_shape():
    frame = pd.DataFrame(
        [
            {"日期": "2026-04-28", "开盘": 100.0, "最高": 105.0, "最低": 99.0, "收盘": 104.0, "成交量": 123456},
            {"日期": "2026-04-29", "开盘": 104.0, "最高": 106.0, "最低": 103.0, "收盘": 105.0, "成交量": 234567},
        ]
    )
    calls = []

    class FakeAkshare:
        @staticmethod
        def stock_zh_a_hist(**kwargs):
            calls.append(kwargs)
            return frame

    service = CnMarketDataService(akshare_module=FakeAkshare())

    result = service.daily_ohlcv_dataframe("600519", period="1mo", end=date(2026, 4, 30))

    assert calls[0]["symbol"] == "600519"
    assert calls[0]["period"] == "daily"
    assert result is not None
    assert list(result.columns) == ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    assert result.index[0].date().isoformat() == "2026-04-28"
    assert result.iloc[-1]["Close"] == 105.0


def test_cn_market_data_service_raises_dependency_error_when_akshare_missing(monkeypatch):
    import app.services.cn_market_data_service as module

    def fake_import(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(module.importlib, "import_module", fake_import)
    service = CnMarketDataService()

    with pytest.raises(CnDependencyError, match="akshare is required"):
        service.listing_rows()
