"""SecurityMaster resolver and deterministic symbol/market normalization utilities.

This service is the single source of truth for deriving canonical identity fields
(symbol/market/exchange/currency/timezone/local_code) from mixed legacy inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

_SUPPORTED_MARKETS = {"US", "HK", "JP", "TW"}

_MARKET_DEFAULTS: dict[str, tuple[str, str]] = {
    "US": ("USD", "America/New_York"),
    "HK": ("HKD", "Asia/Hong_Kong"),
    "JP": ("JPY", "Asia/Tokyo"),
    "TW": ("TWD", "Asia/Taipei"),
}

_MARKET_BY_EXCHANGE: dict[str, str] = {
    "NYSE": "US",
    "NASDAQ": "US",
    "AMEX": "US",
    "XNYS": "US",
    "XNAS": "US",
    "XASE": "US",
    "HKEX": "HK",
    "SEHK": "HK",
    "XHKG": "HK",
    "TSE": "JP",
    "JPX": "JP",
    "XTKS": "JP",
    "TWSE": "TW",
    "TPEX": "TW",
    "XTAI": "TW",
}

_MARKET_BY_SUFFIX: tuple[tuple[str, str], ...] = (
    (".HK", "HK"),
    (".TWO", "TW"),
    (".TW", "TW"),
    (".T", "JP"),
)

_SUFFIX_BY_MARKET: dict[str, str] = {
    "HK": ".HK",
    "JP": ".T",
    "TW": ".TW",
}


@dataclass(frozen=True)
class SecurityIdentity:
    """Canonical security identity fields derived by SecurityMaster."""

    normalized_symbol: str
    canonical_symbol: str
    market: str
    exchange: Optional[str]
    currency: str
    timezone: str
    local_code: str


class SecurityMasterResolver:
    """Deterministic resolver for symbol -> market/exchange/currency identity."""

    @staticmethod
    def normalize_symbol(symbol: str | None) -> str:
        """Normalize a symbol without applying market-specific transforms."""
        normalized = (symbol or "").strip().upper()
        if normalized.startswith("$"):
            normalized = normalized[1:]
        return normalized

    @staticmethod
    def normalize_exchange(exchange: str | None) -> str | None:
        normalized = (exchange or "").strip().upper()
        return normalized or None

    @staticmethod
    def normalize_market(market: str | None) -> str | None:
        normalized = (market or "").strip().upper()
        if normalized in _SUPPORTED_MARKETS:
            return normalized
        return None

    def infer_market(self, symbol: str, exchange: str | None = None) -> str:
        normalized_exchange = self.normalize_exchange(exchange)
        if normalized_exchange and normalized_exchange in _MARKET_BY_EXCHANGE:
            return _MARKET_BY_EXCHANGE[normalized_exchange]

        normalized_symbol = self.normalize_symbol(symbol)
        for suffix, market in _MARKET_BY_SUFFIX:
            if normalized_symbol.endswith(suffix):
                return market
        return "US"

    def resolve_identity(
        self,
        *,
        symbol: str,
        market: str | None = None,
        exchange: str | None = None,
        currency: str | None = None,
        timezone: str | None = None,
        local_code: str | None = None,
    ) -> SecurityIdentity:
        """Resolve deterministic identity fields for a security."""
        normalized_symbol = self.normalize_symbol(symbol)
        normalized_exchange = self.normalize_exchange(exchange)
        normalized_market = self.normalize_market(market)
        if normalized_market is None:
            normalized_market = self.infer_market(normalized_symbol, normalized_exchange)

        default_currency, default_timezone = _MARKET_DEFAULTS.get(
            normalized_market,
            _MARKET_DEFAULTS["US"],
        )

        resolved_currency = (currency or "").strip().upper() or default_currency
        resolved_timezone = (timezone or "").strip() or default_timezone

        resolved_local_code = (local_code or "").strip()
        if not resolved_local_code:
            if "." in normalized_symbol:
                resolved_local_code = normalized_symbol.split(".", 1)[0]
            else:
                resolved_local_code = normalized_symbol

        canonical_symbol = normalized_symbol
        if normalized_market != "US" and "." not in canonical_symbol and resolved_local_code:
            suffix = _SUFFIX_BY_MARKET.get(normalized_market)
            if suffix:
                canonical_symbol = f"{resolved_local_code}{suffix}"

        return SecurityIdentity(
            normalized_symbol=normalized_symbol,
            canonical_symbol=canonical_symbol,
            market=normalized_market,
            exchange=normalized_exchange,
            currency=resolved_currency,
            timezone=resolved_timezone,
            local_code=resolved_local_code,
        )


security_master_resolver = SecurityMasterResolver()
