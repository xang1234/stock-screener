"""Stable Market facts and lookup helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .market import Market


@dataclass(frozen=True, slots=True)
class MarketProfile:
    """Stable facts about one supported Market."""

    market: Market
    label: str
    currency: str
    timezone_name: str
    calendar_id: str
    provider_calendar_id: str | None
    exchanges: tuple[str, ...]
    indexes: tuple[str, ...]
    primary_benchmark_symbol: str
    benchmark_fallback_symbol: str | None
    benchmark_primary_kind: str
    benchmark_fallback_kind: str | None


class MarketRegistry:
    """Registry of stable Market facts.

    Runtime Preferences and Market Workload are separate concepts. This module
    owns stable identity and lookup facts only.
    """

    def __init__(self, profiles: Iterable[MarketProfile]) -> None:
        self._profiles = tuple(profiles)
        self._by_code: dict[str, MarketProfile] = {}
        self._market_by_exchange: dict[str, Market] = {}
        self._market_by_index: dict[str, Market] = {}

        for profile in self._profiles:
            code = profile.market.code
            if code in self._by_code:
                raise ValueError(f"Duplicate market profile: {code}")
            self._by_code[code] = profile
            for exchange in profile.exchanges:
                normalized_exchange = exchange.upper()
                if normalized_exchange in self._market_by_exchange:
                    raise ValueError(f"Duplicate exchange alias: {normalized_exchange}")
                self._market_by_exchange[normalized_exchange] = profile.market
            for index in profile.indexes:
                normalized_index = index.upper()
                if normalized_index in self._market_by_index:
                    raise ValueError(f"Duplicate index alias: {normalized_index}")
                self._market_by_index[normalized_index] = profile.market

    def profile(self, market: Market | str) -> MarketProfile:
        resolved = market if isinstance(market, Market) else Market.from_str(market)
        return self._by_code[resolved.code]

    def profiles(self) -> tuple[MarketProfile, ...]:
        return self._profiles

    def supported_markets(self) -> tuple[Market, ...]:
        return tuple(profile.market for profile in self._profiles)

    def supported_market_codes(self) -> tuple[str, ...]:
        return tuple(profile.market.code for profile in self._profiles)

    def market_for_exchange(self, exchange: str | None) -> Market | None:
        normalized = str(exchange or "").strip().upper()
        if not normalized:
            return None
        return self._market_by_exchange.get(normalized)

    def market_for_index(self, index: str | None) -> Market | None:
        normalized = str(index or "").strip().upper()
        if not normalized:
            return None
        return self._market_by_index.get(normalized)


market_registry = MarketRegistry(
    (
        MarketProfile(
            market=Market("US"),
            label="United States",
            currency="USD",
            timezone_name="America/New_York",
            calendar_id="XNYS",
            provider_calendar_id=None,
            exchanges=("NYSE", "NASDAQ", "AMEX", "XNYS", "XNAS", "XASE"),
            indexes=("SP500",),
            primary_benchmark_symbol="SPY",
            benchmark_fallback_symbol="IVV",
            benchmark_primary_kind="etf",
            benchmark_fallback_kind="etf",
        ),
        MarketProfile(
            market=Market("HK"),
            label="Hong Kong",
            currency="HKD",
            timezone_name="Asia/Hong_Kong",
            calendar_id="XHKG",
            provider_calendar_id=None,
            exchanges=("HKEX", "SEHK", "XHKG"),
            indexes=("HSI",),
            primary_benchmark_symbol="^HSI",
            benchmark_fallback_symbol="2800.HK",
            benchmark_primary_kind="index",
            benchmark_fallback_kind="etf",
        ),
        MarketProfile(
            market=Market("IN"),
            label="India",
            currency="INR",
            timezone_name="Asia/Kolkata",
            calendar_id="XNSE",
            provider_calendar_id="NSE",
            exchanges=("NSE", "XNSE", "BSE", "XBOM"),
            indexes=("NIFTY50",),
            primary_benchmark_symbol="^NSEI",
            benchmark_fallback_symbol="NIFTYBEES.NS",
            benchmark_primary_kind="index",
            benchmark_fallback_kind="etf",
        ),
        MarketProfile(
            market=Market("JP"),
            label="Japan",
            currency="JPY",
            timezone_name="Asia/Tokyo",
            calendar_id="XTKS",
            provider_calendar_id=None,
            exchanges=("TSE", "JPX", "XTKS"),
            indexes=("NIKKEI225",),
            primary_benchmark_symbol="^N225",
            benchmark_fallback_symbol="1306.T",
            benchmark_primary_kind="index",
            benchmark_fallback_kind="etf",
        ),
        MarketProfile(
            market=Market("KR"),
            label="South Korea",
            currency="KRW",
            timezone_name="Asia/Seoul",
            calendar_id="XKRX",
            provider_calendar_id=None,
            exchanges=("KOSPI", "KOSDAQ", "KRX", "XKRX"),
            indexes=("KOSPI",),
            primary_benchmark_symbol="^KS11",
            benchmark_fallback_symbol="069500.KS",
            benchmark_primary_kind="index",
            benchmark_fallback_kind="etf",
        ),
        MarketProfile(
            market=Market("TW"),
            label="Taiwan",
            currency="TWD",
            timezone_name="Asia/Taipei",
            calendar_id="XTAI",
            provider_calendar_id=None,
            exchanges=("TWSE", "TPEX", "XTAI"),
            indexes=("TAIEX",),
            primary_benchmark_symbol="^TWII",
            benchmark_fallback_symbol="0050.TW",
            benchmark_primary_kind="index",
            benchmark_fallback_kind="etf",
        ),
        MarketProfile(
            market=Market("CN"),
            label="China",
            currency="CNY",
            timezone_name="Asia/Shanghai",
            calendar_id="XSHG",
            provider_calendar_id=None,
            exchanges=("SSE", "SHSE", "XSHG", "SZSE", "XSHE", "BJSE", "XBSE", "XBEI"),
            indexes=("CSI300",),
            primary_benchmark_symbol="000300.SS",
            benchmark_fallback_symbol="000001.SS",
            benchmark_primary_kind="index",
            benchmark_fallback_kind="index",
        ),
    )
)
