"""Shared catalog for Daily Snapshot key-market instruments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class KeyMarketInstrument:
    market: str
    display_symbol: str
    display_name: str
    currency: str
    price_symbol: str | None = None

    @property
    def data_symbol(self) -> str:
        return self.price_symbol or self.display_symbol

    def watchlist_item(self) -> dict[str, str]:
        return {
            "symbol": self.display_symbol,
            "display_name": self.display_name,
        }


KEY_MARKET_INSTRUMENTS_BY_MARKET: Final[dict[str, tuple[KeyMarketInstrument, ...]]] = {
    "US": (
        KeyMarketInstrument("US", "SPY", "S&P 500 ETF", "USD"),
        KeyMarketInstrument("US", "QQQ", "Nasdaq 100 ETF", "USD"),
        KeyMarketInstrument("US", "IWM", "Russell 2000 ETF", "USD"),
        KeyMarketInstrument("US", "TVC:DXY", "US Dollar Index", "USD", "DX-Y.NYB"),
        KeyMarketInstrument("US", "FX:USDSGD", "USD/SGD", "SGD", "SGD=X"),
        KeyMarketInstrument("US", "BITSTAMP:BTCUSD", "Bitcoin", "USD", "BTC-USD"),
        KeyMarketInstrument("US", "GLD", "Gold ETF", "USD"),
        KeyMarketInstrument("US", "TLT", "20+ Year Treasury ETF", "USD"),
        KeyMarketInstrument("US", "TVC:VIX", "Volatility Index", "USD", "^VIX"),
    ),
    "HK": (
        KeyMarketInstrument("HK", "^HSI", "Hang Seng Index", "HKD"),
        KeyMarketInstrument("HK", "2800.HK", "Tracker Fund Hong Kong", "HKD"),
        KeyMarketInstrument("HK", "0700.HK", "Tencent", "HKD"),
        KeyMarketInstrument("HK", "3690.HK", "Meituan", "HKD"),
        KeyMarketInstrument("HK", "0941.HK", "China Mobile", "HKD"),
    ),
    "IN": (
        KeyMarketInstrument("IN", "^NSEI", "Nifty 50", "INR"),
        KeyMarketInstrument("IN", "NIFTYBEES.NS", "Nippon India ETF Nifty 50 BeES", "INR"),
        KeyMarketInstrument("IN", "RELIANCE.NS", "Reliance Industries", "INR"),
        KeyMarketInstrument("IN", "TCS.NS", "Tata Consultancy Services", "INR"),
        KeyMarketInstrument("IN", "HDFCBANK.NS", "HDFC Bank", "INR"),
    ),
    "JP": (
        KeyMarketInstrument("JP", "^N225", "Nikkei 225", "JPY"),
        KeyMarketInstrument("JP", "1306.T", "TOPIX ETF", "JPY"),
        KeyMarketInstrument("JP", "7203.T", "Toyota", "JPY"),
        KeyMarketInstrument("JP", "6758.T", "Sony Group", "JPY"),
        KeyMarketInstrument("JP", "9984.T", "SoftBank Group", "JPY"),
    ),
    "KR": (
        KeyMarketInstrument("KR", "^KS11", "KOSPI Composite", "KRW"),
        KeyMarketInstrument("KR", "069500.KS", "KODEX 200 ETF", "KRW"),
        KeyMarketInstrument("KR", "005930.KS", "Samsung Electronics", "KRW"),
        KeyMarketInstrument("KR", "000660.KS", "SK hynix", "KRW"),
        KeyMarketInstrument("KR", "035420.KS", "NAVER", "KRW"),
    ),
    "TW": (
        KeyMarketInstrument("TW", "^TWII", "TAIEX", "TWD"),
        KeyMarketInstrument("TW", "0050.TW", "TW50 ETF", "TWD"),
        KeyMarketInstrument("TW", "2330.TW", "TSMC", "TWD"),
        KeyMarketInstrument("TW", "2317.TW", "Hon Hai", "TWD"),
        KeyMarketInstrument("TW", "2454.TW", "MediaTek", "TWD"),
    ),
    "CN": (
        KeyMarketInstrument("CN", "000300.SS", "CSI 300", "CNY"),
        KeyMarketInstrument("CN", "000001.SS", "Shanghai Composite", "CNY"),
        KeyMarketInstrument("CN", "600519.SS", "Kweichow Moutai", "CNY"),
        KeyMarketInstrument("CN", "000001.SZ", "Ping An Bank", "CNY"),
        KeyMarketInstrument("CN", "300750.SZ", "CATL", "CNY"),
    ),
    "SG": (
        KeyMarketInstrument("SG", "^STI", "Straits Times Index", "SGD"),
        KeyMarketInstrument("SG", "ES3.SI", "SPDR Straits Times Index ETF", "SGD"),
        KeyMarketInstrument("SG", "D05.SI", "DBS Group", "SGD"),
        KeyMarketInstrument("SG", "C6L.SI", "Singapore Airlines", "SGD"),
        KeyMarketInstrument("SG", "Z74.SI", "Singtel", "SGD"),
        KeyMarketInstrument("SG", "O39.SI", "OCBC Bank", "SGD"),
        KeyMarketInstrument("SG", "U11.SI", "United Overseas Bank", "SGD"),
    ),
    "CA": (
        KeyMarketInstrument("CA", "^GSPTSE", "S&P/TSX Composite", "CAD"),
        KeyMarketInstrument("CA", "XIU.TO", "iShares S&P/TSX 60 ETF", "CAD"),
        KeyMarketInstrument("CA", "RY.TO", "Royal Bank of Canada", "CAD"),
        KeyMarketInstrument("CA", "SHOP.TO", "Shopify", "CAD"),
        KeyMarketInstrument("CA", "CNR.TO", "Canadian National Railway", "CAD"),
    ),
    "DE": (
        KeyMarketInstrument("DE", "^GDAXI", "DAX", "EUR"),
        KeyMarketInstrument("DE", "EXS1.DE", "iShares DAX UCITS ETF", "EUR"),
        KeyMarketInstrument("DE", "SAP.DE", "SAP", "EUR"),
        KeyMarketInstrument("DE", "SIE.DE", "Siemens", "EUR"),
        KeyMarketInstrument("DE", "ALV.DE", "Allianz", "EUR"),
    ),
    "AU": (
        KeyMarketInstrument("AU", "^AXJO", "S&P/ASX 200", "AUD"),
        KeyMarketInstrument("AU", "IOZ.AX", "iShares Core S&P/ASX 200 ETF", "AUD"),
        KeyMarketInstrument("AU", "BHP.AX", "BHP Group", "AUD"),
        KeyMarketInstrument("AU", "CBA.AX", "Commonwealth Bank", "AUD"),
        KeyMarketInstrument("AU", "CSL.AX", "CSL", "AUD"),
    ),
    "MY": (
        KeyMarketInstrument("MY", "^KLSE", "FBM KLCI", "MYR"),
        KeyMarketInstrument("MY", "1155.KL", "Maybank", "MYR"),
        KeyMarketInstrument("MY", "1295.KL", "Public Bank", "MYR"),
        KeyMarketInstrument("MY", "1023.KL", "CIMB Group", "MYR"),
        KeyMarketInstrument("MY", "5347.KL", "Tenaga Nasional", "MYR"),
        KeyMarketInstrument("MY", "5183.KL", "Petronas Chemicals", "MYR"),
    ),
}


def _normalize_token(value: str | None) -> str:
    return str(value or "").strip().lstrip("$").upper()


def key_market_instruments(market: str | None) -> tuple[KeyMarketInstrument, ...]:
    return KEY_MARKET_INSTRUMENTS_BY_MARKET.get(_normalize_token(market), ())


def key_market_watchlist_defaults(market: str | None) -> tuple[dict[str, str], ...]:
    return tuple(instrument.watchlist_item() for instrument in key_market_instruments(market))


def key_market_price_symbols(market: str | None) -> tuple[str, ...]:
    return tuple(instrument.data_symbol for instrument in key_market_instruments(market))


def resolve_key_market_price_symbol(symbol: str | None) -> str:
    normalized = _normalize_token(symbol)
    for instruments in KEY_MARKET_INSTRUMENTS_BY_MARKET.values():
        for instrument in instruments:
            if instrument.display_symbol == normalized:
                return instrument.data_symbol
    return normalized
