from __future__ import annotations

import io
import zipfile
from datetime import date

import pandas as pd

from app.services.kr_market_data_service import KrxMarketDataService, OpenDartFundamentalsService


class _FakeResponse:
    def __init__(self, *, content: bytes = b"", payload: dict | None = None) -> None:
        self.content = content
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, *, params: dict, timeout: int) -> _FakeResponse:
        self.calls.append((url, params))
        if url.endswith("corpCode.xml"):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, mode="w") as archive:
                archive.writestr(
                    "CORPCODE.xml",
                    """
                    <result>
                      <list>
                        <corp_code>00126380</corp_code>
                        <corp_name>Samsung Electronics</corp_name>
                        <stock_code>005930</stock_code>
                      </list>
                    </result>
                    """,
                )
            return _FakeResponse(content=buffer.getvalue())
        return _FakeResponse(
            payload={
                "status": "000",
                "list": [
                    {"account_nm": "Revenue", "thstrm_amount": "100,000"},
                    {"account_nm": "Operating income", "thstrm_amount": "12,000"},
                    {"account_nm": "Net income", "thstrm_amount": "9,000"},
                    {"account_nm": "Gross profit", "thstrm_amount": "20,000"},
                    {"account_nm": "Total assets", "thstrm_amount": "300,000"},
                    {"account_nm": "Total liabilities", "thstrm_amount": "120,000"},
                    {"account_nm": "Total equity", "thstrm_amount": "180,000"},
                    {"account_nm": "Current assets", "thstrm_amount": "80,000"},
                    {"account_nm": "Current liabilities", "thstrm_amount": "40,000"},
                ],
            }
        )


def test_opendart_returns_empty_when_api_key_missing() -> None:
    service = OpenDartFundamentalsService(api_key="")

    assert service.get_statement_fundamentals("005930") == {}


def test_opendart_maps_statement_rows_to_existing_fundamental_fields() -> None:
    session = _FakeSession()
    service = OpenDartFundamentalsService(api_key="token", session=session)

    fields = service.get_statement_fundamentals("5930", business_year=2025)

    assert fields["revenue_current"] == 100000
    assert fields["operating_margin"] == 12.0
    assert fields["profit_margin"] == 9.0
    assert fields["gross_margin"] == 20.0
    assert fields["roe"] == 5.0
    assert fields["roa"] == 3.0
    assert fields["debt_to_equity"] == 0.6666666666666666
    assert fields["current_ratio"] == 2.0
    assert fields["recent_quarter_date"] == "2025-FY"
    assert session.calls[1][1]["corp_code"] == "00126380"


def test_krx_daily_ohlcv_dataframe_uses_canonical_price_shape() -> None:
    class _FakeStockModule:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []

        def get_market_ohlcv_by_date(self, start: str, end: str, local_code: str):
            self.calls.append((start, end, local_code))
            return pd.DataFrame(
                {
                    "시가": [100.0],
                    "고가": [110.0],
                    "저가": [95.0],
                    "종가": [105.0],
                    "거래량": [12345],
                    "거래대금": [1_296_225],
                },
                index=pd.to_datetime(["2026-04-29"]),
            )

    stock_module = _FakeStockModule()
    service = KrxMarketDataService(stock_module=stock_module)

    frame = service.daily_ohlcv_dataframe("005930", period="7d", end=date(2026, 4, 29))

    assert stock_module.calls == [("20260422", "20260429", "005930")]
    assert frame is not None
    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
    assert frame.loc[pd.Timestamp("2026-04-29"), "Close"] == 105.0


def test_krx_core_fundamentals_caches_whole_market_frames() -> None:
    class _FakeStockModule:
        def __init__(self) -> None:
            self.market_cap_calls: list[tuple[str, str]] = []
            self.fundamental_calls: list[tuple[str, str]] = []

        def get_market_cap(self, as_of: str, *, market: str):
            self.market_cap_calls.append((as_of, market))
            return pd.DataFrame(
                {
                    "시가총액": [530_000_000_000_000, 17_000_000_000_000],
                    "상장주식수": [5_969_782_550, 123_456_789],
                },
                index=["005930", "091990"],
            )

        def get_market_fundamental(self, as_of: str, *, market: str):
            self.fundamental_calls.append((as_of, market))
            return pd.DataFrame(
                {
                    "PER": [12.5, 28.0],
                    "PBR": [1.4, 4.2],
                    "EPS": [5600, 900],
                    "BPS": [73000, 8000],
                    "DIV": [2.1, 0.0],
                },
                index=["005930", "091990"],
            )

    stock_module = _FakeStockModule()
    service = KrxMarketDataService(stock_module=stock_module)

    samsung = service.core_fundamentals("005930", as_of=date(2026, 4, 29))
    celltrion = service.core_fundamentals("091990", as_of=date(2026, 4, 29))

    assert samsung["market_cap"] == 530_000_000_000_000
    assert celltrion["pe_ratio"] == 28.0
    assert stock_module.market_cap_calls == [("20260429", "ALL")]
    assert stock_module.fundamental_calls == [("20260429", "ALL")]


def test_krx_listing_rows_falls_back_to_current_listing_finder_when_daily_tickers_empty() -> None:
    class _FakeStockModule:
        def __init__(self) -> None:
            self.ticker_calls: list[tuple[str, str]] = []
            self.name_calls: list[str] = []

        def get_market_ticker_list(self, as_of: str, *, market: str):
            self.ticker_calls.append((as_of, market))
            return []

        def get_market_ticker_name(self, ticker: str):
            self.name_calls.append(ticker)
            return f"daily {ticker}"

    class _FakeListingSource:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def fetch(self, market_code: str):
            self.calls.append(market_code)
            return pd.DataFrame(
                [
                    {
                        "full_code": "KR7005930003",
                        "short_code": "005930",
                        "codeName": "Samsung Electronics",
                        "marketCode": "STK",
                    },
                    {
                        "full_code": "KR7091990002",
                        "short_code": "091990",
                        "codeName": "Celltrion Healthcare",
                        "marketCode": "KSQ",
                    },
                    {
                        "full_code": "KR7066660001",
                        "short_code": "066660",
                        "codeName": "Wrong Board",
                        "marketCode": "KNX",
                    },
                    {
                        "full_code": "KR7000000000",
                        "short_code": "",
                        "codeName": "Blank Code",
                        "marketCode": "STK",
                    },
                ]
            )

    stock_module = _FakeStockModule()
    listing_source = _FakeListingSource()
    service = KrxMarketDataService(
        stock_module=stock_module,
        listing_source=listing_source,
    )

    rows = service.listing_rows(boards=("KOSPI", "KOSDAQ"), as_of=None)

    today_token = date.today().strftime("%Y%m%d")
    assert stock_module.ticker_calls == [
        (today_token, "KOSPI"),
        (today_token, "KOSDAQ"),
    ]
    assert stock_module.name_calls == []
    assert listing_source.calls == ["STK", "KSQ"]
    assert rows == [
        {
            "symbol": "005930.KS",
            "local_code": "005930",
            "name": "Samsung Electronics",
            "exchange": "KOSPI",
            "sector": "",
            "industry": "",
            "market_cap": None,
            "source_board": "KOSPI",
            "isin": "KR7005930003",
        },
        {
            "symbol": "091990.KQ",
            "local_code": "091990",
            "name": "Celltrion Healthcare",
            "exchange": "KOSDAQ",
            "sector": "",
            "industry": "",
            "market_cap": None,
            "source_board": "KOSDAQ",
            "isin": "KR7091990002",
        },
    ]


def test_krx_listing_rows_does_not_use_current_listing_finder_for_historical_empty_tickers() -> None:
    class _FakeStockModule:
        def __init__(self) -> None:
            self.ticker_calls: list[tuple[str, str]] = []

        def get_market_ticker_list(self, as_of: str, *, market: str):
            self.ticker_calls.append((as_of, market))
            return []

    class _FakeListingSource:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def fetch(self, market_code: str):
            self.calls.append(market_code)
            return pd.DataFrame(
                [
                    {
                        "full_code": "KR7005930003",
                        "short_code": "005930",
                        "codeName": "Samsung Electronics",
                        "marketCode": "STK",
                    },
                ]
            )

    stock_module = _FakeStockModule()
    listing_source = _FakeListingSource()
    service = KrxMarketDataService(
        stock_module=stock_module,
        listing_source=listing_source,
    )

    rows = service.listing_rows(boards=("KOSPI",), as_of=date(2026, 4, 1))

    assert stock_module.ticker_calls == [("20260401", "KOSPI")]
    assert listing_source.calls == []
    assert rows == []
