from __future__ import annotations

from datetime import date
import signal
import time

import pandas as pd
import pytest
import requests

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
    assert rows[0]["sector"] == "Consumer Staples"
    assert rows[0]["industry"] == "酿酒行业"
    assert rows[0]["pe_ratio"] == 28.4
    assert rows[1]["exchange"] == "SZSE"
    assert rows[1]["sector"] == "Financials"
    assert rows[2]["exchange"] == "BJSE"
    assert rows[2]["sector"] == "Consumer Discretionary"


def test_cn_market_data_service_preserves_zero_listing_numeric_fields():
    frame = pd.DataFrame(
        [
            {
                "代码": "600519",
                "名称": "贵州茅台",
                "最新价": 0,
                "总市值": 0,
                "流通市值": 0,
                "市盈率-动态": 0,
                "市净率": 0,
                "股息率": 0,
                "所属行业": "酿酒行业",
            },
        ]
    )

    class FakeAkshare:
        @staticmethod
        def stock_zh_a_spot_em():
            return frame

    service = CnMarketDataService(akshare_module=FakeAkshare())

    row = service.listing_rows(as_of=date(2026, 4, 30))[0]

    assert row["market_cap"] == 0
    assert row["float_market_cap"] == 0
    assert row["shares_outstanding"] is None
    assert row["pe_ratio"] == 0
    assert row["price_to_book"] == 0
    assert row["dividend_yield"] == 0


def test_cn_market_data_service_falls_back_to_code_name_list_when_spot_fails():
    class FallbackAkshare:
        @staticmethod
        def stock_zh_a_spot_em():
            raise requests.exceptions.ConnectionError("eastmoney disconnected")

        @staticmethod
        def stock_info_a_code_name():
            return pd.DataFrame(
                [
                    {"code": "600519", "name": "贵州茅台"},
                    {"code": "000001", "name": "平安银行"},
                    {"code": "920118", "name": "太湖雪"},
                ]
            )

    service = CnMarketDataService(akshare_module=FallbackAkshare(), timeout_seconds=1)

    rows = service.listing_rows(as_of=date(2026, 4, 30))

    assert [row["symbol"] for row in rows] == ["600519.SS", "000001.SZ", "920118.BJ"]
    assert rows[0]["name"] == "贵州茅台"
    assert rows[0]["market_cap"] is None
    assert rows[1]["exchange"] == "SZSE"
    assert rows[2]["exchange"] == "BJSE"


def test_cn_market_data_service_falls_back_to_code_name_list_when_spot_is_empty():
    class FallbackAkshare:
        @staticmethod
        def stock_zh_a_spot_em():
            return pd.DataFrame()

        @staticmethod
        def stock_info_a_code_name():
            return pd.DataFrame([{"code": 688001, "name": "华兴源创"}])

    service = CnMarketDataService(akshare_module=FallbackAkshare(), timeout_seconds=1)

    rows = service.listing_rows(as_of=date(2026, 4, 30))

    assert rows[0]["symbol"] == "688001.SS"
    assert rows[0]["board"] == "SSE_STAR"


def test_cn_market_data_service_falls_back_to_code_name_list_when_spot_times_out():
    class FallbackAkshare:
        @staticmethod
        def stock_zh_a_spot_em():
            time.sleep(5)
            return pd.DataFrame([{"代码": "600519", "名称": "贵州茅台"}])

        @staticmethod
        def stock_info_a_code_name():
            return pd.DataFrame([{"code": "000001", "name": "平安银行"}])

    service = CnMarketDataService(akshare_module=FallbackAkshare(), timeout_seconds=1)
    started_at = time.monotonic()

    rows = service.listing_rows(as_of=date(2026, 4, 30))

    assert time.monotonic() - started_at < 3
    assert rows[0]["symbol"] == "000001.SZ"
    assert rows[0]["name"] == "平安银行"


def test_cn_timeout_helper_restores_existing_signal_timer():
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        pytest.skip("SIGALRM timers are unavailable on this platform")

    import app.services.cn_market_data_service as module

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def temporary_handler(signum, frame):
        del signum, frame

    try:
        signal.signal(signal.SIGALRM, temporary_handler)
        signal.setitimer(signal.ITIMER_REAL, 10.0)

        result = module._call_with_timeout(
            lambda: "ok",
            timeout_seconds=1,
            operation_name="test fetch",
        )

        restored_delay, restored_interval = signal.getitimer(signal.ITIMER_REAL)
        assert result == "ok"
        assert restored_delay > 8.0
        assert restored_interval == 0.0
        assert signal.getsignal(signal.SIGALRM) is temporary_handler
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0 or previous_timer[1] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def test_cn_market_data_service_times_out_listing_fetch():
    class SlowAkshare:
        @staticmethod
        def stock_zh_a_spot_em():
            time.sleep(5)
            return pd.DataFrame([{"代码": "600519", "名称": "贵州茅台"}])

    service = CnMarketDataService(akshare_module=SlowAkshare(), timeout_seconds=1)
    started_at = time.monotonic()

    with pytest.raises(requests.exceptions.Timeout, match="CN A-share listing fetch timed out"):
        service.listing_rows(as_of=date(2026, 4, 30))

    assert time.monotonic() - started_at < 3


def test_cn_market_data_service_skips_baostock_ohlcv_for_beijing_codes():
    class FakeBaoStock:
        def login(self):  # pragma: no cover - should not be called
            raise AssertionError("BaoStock should not be queried for Beijing Stock Exchange codes")

    service = CnMarketDataService(akshare_module=object(), baostock_module=FakeBaoStock())

    assert service._daily_ohlcv_from_baostock("920118", start="20260401", end="20260430") == []


def test_cn_market_data_service_maps_akshare_ohlcv_to_yfinance_shape():
    frame = pd.DataFrame(
        [
            {"日期": "2026-04-28", "开盘": 0.0, "最高": 0.0, "最低": 0.0, "收盘": 0.0, "成交量": 0, "成交额": 0},
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
    assert result.iloc[0]["Open"] == 0
    assert result.iloc[0]["Volume"] == 0
    assert result.iloc[-1]["Close"] == 105.0


def test_cn_market_data_service_raises_dependency_error_when_akshare_missing(monkeypatch):
    import app.services.cn_market_data_service as module

    def fake_import(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(module.importlib, "import_module", fake_import)
    service = CnMarketDataService()

    with pytest.raises(CnDependencyError, match="akshare is required"):
        service.listing_rows()
