"""Shared catalog for Daily Snapshot market-context instruments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class MarketContextInstrument:
    category: str
    display_symbol: str
    display_name: str
    currency: str
    price_symbol: str | None = None

    @property
    def data_symbol(self) -> str:
        return self.price_symbol or self.display_symbol


MARKET_CONTEXT_INSTRUMENTS_BY_MARKET: Final[
    dict[str, tuple[MarketContextInstrument, ...]]
] = {
    "US": (
        #
        # Market Structure
        #
        MarketContextInstrument(
            category="market_structure",
            display_symbol="SPY",
            display_name="S&P 500 ETF",
            currency="USD",
        ),
        MarketContextInstrument(
            category="market_structure",
            display_symbol="QQQ",
            display_name="Nasdaq 100 ETF",
            currency="USD",
        ),
        MarketContextInstrument(
            category="market_structure",
            display_symbol="IWM",
            display_name="Russell 2000 ETF",
            currency="USD",
        ),
        MarketContextInstrument(
            category="market_structure",
            display_symbol="RSP",
            display_name="Equal Weight S&P 500 ETF",
            currency="USD",
        ),

        #
        # Commodities
        #
        MarketContextInstrument(
            category="commodities",
            display_symbol="GLD",
            display_name="Gold ETF",
            currency="USD",
        ),
        MarketContextInstrument(
            category="commodities",
            display_symbol="SLV",
            display_name="Silver ETF",
            currency="USD",
        ),
        MarketContextInstrument(
            category="commodities",
            display_symbol="USO",
            display_name="United States Oil Fund",
            currency="USD",
        ),
        MarketContextInstrument(
            category="commodities",
            display_symbol="COPX",
            display_name="Global X Copper Miners ETF",
            currency="USD",
        ),

        #
        # Risk / Macro
        #
        MarketContextInstrument(
            category="risk_macro",
            display_symbol="TLT",
            display_name="20+ Year Treasury ETF",
            currency="USD",
        ),
        MarketContextInstrument(
            category="risk_macro",
            display_symbol="TVC:DXY",
            display_name="US Dollar Index",
            currency="USD",
            price_symbol="DX-Y.NYB",
        ),
        MarketContextInstrument(
            category="risk_macro",
            display_symbol="TVC:VIX",
            display_name="Volatility Index",
            currency="USD",
            price_symbol="^VIX",
        ),

        #
        # Economic Activity
        #
        MarketContextInstrument(
            category="economic_activity",
            display_symbol="IYT",
            display_name="Transportation ETF",
            currency="USD",
        ),

        #
        # Sector Leadership
        #
        MarketContextInstrument(
            category="sector_leadership",
            display_symbol="SOXX",
            display_name="Semiconductor ETF",
            currency="USD",
        ),
        MarketContextInstrument(
            category="sector_leadership",
            display_symbol="XLF",
            display_name="Financial ETF",
            currency="USD",
        ),
        MarketContextInstrument(
            category="sector_leadership",
            display_symbol="XLE",
            display_name="Energy ETF",
            currency="USD",
        ),  
        MarketContextInstrument(
            category="sector_leadership",
            display_symbol="XLI",
            display_name="Industrial ETF",
            currency="USD",
        ),
        MarketContextInstrument(
            category="defensive",
            display_symbol="XLU",
            display_name="Utilities ETF",
            currency="USD",
        ),     
        MarketContextInstrument(
            category="defensive",
            display_symbol="XLP",
            display_name="Consumer Staples ETF",
            currency="USD",
        ),


        #
        # Crypto
        #
        MarketContextInstrument(
            category="crypto",
            display_symbol="BITSTAMP:BTCUSD",
            display_name="Bitcoin",
            currency="USD",
            price_symbol="BTC-USD",
        ),
        MarketContextInstrument(
            category="crypto",
            display_symbol="ETH-USD",
            display_name="Ethereum",
            currency="USD",
        ),
    ),
    "HK": (
        # Market Structure
        MarketContextInstrument(
            category="market_structure",
            display_symbol="^HSI",
            display_name="Hang Seng Index",
            currency="HKD",
        ),
        MarketContextInstrument(
            category="market_structure",
            display_symbol="2800.HK",
            display_name="Tracker Fund Hong Kong",
            currency="HKD",
        ),

        # Commodities
        MarketContextInstrument(
            category="commodities",
            display_symbol="GLD",
            display_name="Gold ETF",
            currency="USD",
        ),

        # Risk / Macro
        MarketContextInstrument(
            category="risk_macro",
            display_symbol="TVC:DXY",
            display_name="US Dollar Index",
            currency="USD",
            price_symbol="DX-Y.NYB",
        ),

        # China Proxy
        MarketContextInstrument(
            category="regional_leadership",
            display_symbol="0700.HK",
            display_name="Tencent",
            currency="HKD",
        ),
        MarketContextInstrument(
            category="regional_leadership",
            display_symbol="3690.HK",
            display_name="Meituan",
            currency="HKD",
        ),
    ),

    "IN": (),
    "JP": (
        # Market Structure
        MarketContextInstrument(
            category="market_structure",
            display_symbol="^N225",
            display_name="Nikkei 225",
            currency="JPY",
        ),
        MarketContextInstrument(
            category="market_structure",
            display_symbol="1306.T",
            display_name="TOPIX ETF",
            currency="JPY",
        ),

        # Risk / Macro
        MarketContextInstrument(
            category="risk_macro",
            display_symbol="TVC:DXY",
            display_name="US Dollar Index",
            currency="USD",
            price_symbol="DX-Y.NYB",
        ),

        # Global Commodities
        MarketContextInstrument(
            category="commodities",
            display_symbol="GLD",
            display_name="Gold ETF",
            currency="USD",
        ),

        # Regional Leadership
        MarketContextInstrument(
            category="regional_leadership",
            display_symbol="7203.T",
            display_name="Toyota",
            currency="JPY",
        ),
        MarketContextInstrument(
            category="regional_leadership",
            display_symbol="6758.T",
            display_name="Sony Group",
            currency="JPY",
        ),
    ),
    "KR": (),
    "TW": (),
    "CN": (),
    "SG": (),
    "CA": (),
    "DE": (),
    "AU": (),
    "MY": (),
}


def _normalize_token(value: str | None) -> str:
    return str(value or "").strip().lstrip("$").upper()


def market_context_instruments(
    market: str | None,
) -> tuple[MarketContextInstrument, ...]:
    return MARKET_CONTEXT_INSTRUMENTS_BY_MARKET.get(
        _normalize_token(market),
        (),
    )


def market_context_price_symbols(
    market: str | None,
) -> tuple[str, ...]:
    return tuple(
        instrument.data_symbol
        for instrument in market_context_instruments(market)
    )


def market_context_data_symbol_for_display(
    display_symbol: str | None,
    *,
    market: str | None = None,
) -> str | None:
    normalized_symbol = _normalize_token(display_symbol)

    if not normalized_symbol:
        return None

    instrument_groups = (
        (market_context_instruments(market),)
        if market is not None
        else MARKET_CONTEXT_INSTRUMENTS_BY_MARKET.values()
    )

    for instruments in instrument_groups:
        for instrument in instruments:
            if instrument.display_symbol == normalized_symbol:
                return instrument.data_symbol

    return None