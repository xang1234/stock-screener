"""Stable Market Catalog facts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

CATALOG_VERSION = "2026-05-09.v1"


class MarketCatalogError(ValueError):
    """Raised when a caller asks for an unsupported Market."""


@dataclass(frozen=True)
class MarketCapabilities:
    benchmark: bool
    breadth: bool
    fundamentals: bool
    group_rankings: bool
    feature_snapshot: bool
    official_universe: bool
    finviz_screening: bool


@dataclass(frozen=True)
class MarketCatalogEntry:
    code: str
    label: str
    currency: str
    timezone: str
    calendar_id: str
    exchanges: tuple[str, ...]
    indexes: tuple[str, ...]
    capabilities: MarketCapabilities

    def as_runtime_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["exchanges"] = list(self.exchanges)
        payload["indexes"] = list(self.indexes)
        return payload


class MarketCatalog:
    """Stable Market facts; mutable runtime state lives elsewhere."""

    def __init__(self, entries: Iterable[MarketCatalogEntry]) -> None:
        self._entries = tuple(entries)
        self._by_code = {entry.code: entry for entry in self._entries}

    def supported_market_codes(self) -> list[str]:
        return [entry.code for entry in self._entries]

    def get(self, market: str | None) -> MarketCatalogEntry:
        code = (market or "").strip().upper()
        try:
            return self._by_code[code]
        except KeyError as exc:
            supported = ", ".join(self.supported_market_codes())
            raise MarketCatalogError(
                f"Unsupported market {market!r}. Supported: {supported}"
            ) from exc

    def as_runtime_payload(self) -> dict[str, object]:
        return {
            "version": CATALOG_VERSION,
            "markets": [entry.as_runtime_payload() for entry in self._entries],
        }


FULL_CAPABILITIES = MarketCapabilities(
    benchmark=True,
    breadth=True,
    fundamentals=True,
    group_rankings=True,
    feature_snapshot=True,
    official_universe=True,
    finviz_screening=False,
)


MARKET_CATALOG = MarketCatalog(
    [
        MarketCatalogEntry(
            code="US",
            label="United States",
            currency="USD",
            timezone="America/New_York",
            calendar_id="XNYS",
            exchanges=("NYSE", "NASDAQ", "AMEX"),
            indexes=("SP500",),
            capabilities=MarketCapabilities(
                benchmark=True,
                breadth=True,
                fundamentals=True,
                group_rankings=True,
                feature_snapshot=True,
                official_universe=False,
                finviz_screening=True,
            ),
        ),
        MarketCatalogEntry(
            code="HK",
            label="Hong Kong",
            currency="HKD",
            timezone="Asia/Hong_Kong",
            calendar_id="XHKG",
            exchanges=("HKEX", "SEHK", "XHKG"),
            indexes=("HSI",),
            capabilities=FULL_CAPABILITIES,
        ),
        MarketCatalogEntry(
            code="IN",
            label="India",
            currency="INR",
            timezone="Asia/Kolkata",
            calendar_id="XNSE",
            exchanges=("NSE", "XNSE", "BSE", "XBOM"),
            indexes=(),
            capabilities=FULL_CAPABILITIES,
        ),
        MarketCatalogEntry(
            code="JP",
            label="Japan",
            currency="JPY",
            timezone="Asia/Tokyo",
            calendar_id="XTKS",
            exchanges=("TSE", "JPX", "XTKS"),
            indexes=("NIKKEI225",),
            capabilities=FULL_CAPABILITIES,
        ),
        MarketCatalogEntry(
            code="KR",
            label="South Korea",
            currency="KRW",
            timezone="Asia/Seoul",
            calendar_id="XKRX",
            exchanges=("KOSPI", "KOSDAQ", "KRX", "XKRX"),
            indexes=(),
            capabilities=FULL_CAPABILITIES,
        ),
        MarketCatalogEntry(
            code="TW",
            label="Taiwan",
            currency="TWD",
            timezone="Asia/Taipei",
            calendar_id="XTAI",
            exchanges=("TWSE", "TPEX", "XTAI"),
            indexes=("TAIEX",),
            capabilities=FULL_CAPABILITIES,
        ),
        MarketCatalogEntry(
            code="CN",
            label="China A-shares",
            currency="CNY",
            timezone="Asia/Shanghai",
            calendar_id="XSHG",
            exchanges=("SSE", "SZSE", "BJSE", "XSHG", "XSHE", "XBSE"),
            indexes=(),
            capabilities=FULL_CAPABILITIES,
        ),
        MarketCatalogEntry(
            code="CA",
            label="Canada",
            currency="CAD",
            timezone="America/Toronto",
            calendar_id="XTSE",
            exchanges=("TSX", "TSXV", "XTSE", "XTNX"),
            indexes=("TSX_COMPOSITE",),
            capabilities=FULL_CAPABILITIES,
        ),
    ]
)


def get_market_catalog() -> MarketCatalog:
    return MARKET_CATALOG
