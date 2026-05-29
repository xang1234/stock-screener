"""Stable Market Catalog facts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .mic import MicFacts

CATALOG_VERSION = "2026-05-17.v1"


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
    primary_mic: str
    mics: tuple[str, ...]
    supported_currencies: tuple[str, ...]
    default_currency: str
    mic_facts: tuple[MicFacts, ...]
    exchanges: tuple[str, ...]
    indexes: tuple[str, ...]
    capabilities: MarketCapabilities

    def __post_init__(self) -> None:
        if not self.mics:
            raise ValueError(f"{self.code} must declare at least one MIC")
        if len(set(self.mics)) != len(self.mics):
            raise ValueError(f"{self.code} duplicate MICs are not allowed")
        if not self.supported_currencies:
            raise ValueError(f"{self.code} must declare at least one supported currency")
        if self.primary_mic not in self.mics:
            raise ValueError(f"{self.code} primary MIC must be present in mics")
        if self.default_currency not in self.supported_currencies:
            raise ValueError(
                f"{self.code} default currency must be present in supported_currencies"
            )
        fact_mic_values = tuple(facts.mic for facts in self.mic_facts)
        fact_mics = set(fact_mic_values)
        if len(fact_mics) != len(fact_mic_values):
            raise ValueError(f"{self.code} duplicate MIC facts are not allowed")
        if fact_mics != set(self.mics):
            raise ValueError(f"{self.code} MIC facts must match mics")
        unsupported_fact_currencies = sorted(
            {facts.default_currency for facts in self.mic_facts}
            - set(self.supported_currencies)
        )
        if unsupported_fact_currencies:
            raise ValueError(
                f"{self.code} MIC default currencies must be present in "
                f"supported_currencies: {', '.join(unsupported_fact_currencies)}"
            )
        if self.default_currency != self.primary_mic_facts.default_currency:
            raise ValueError(
                f"{self.code} default currency must match primary MIC default currency"
            )

    @property
    def primary_mic_facts(self) -> MicFacts:
        return self.mic_facts_for(self.primary_mic)

    @property
    def currency(self) -> str:
        """Deprecated compatibility alias for fallback/default row currency."""
        return self.default_currency

    @property
    def timezone(self) -> str:
        """Deprecated compatibility alias for primary MIC display timezone."""
        return self.primary_mic_facts.timezone

    @property
    def display_timezone(self) -> str:
        return self.primary_mic_facts.timezone

    @property
    def calendar_id(self) -> str:
        """Deprecated compatibility alias for primary MIC calendar ID."""
        return self.primary_mic_facts.calendar_id

    @property
    def provider_calendar_id(self) -> str | None:
        return self.primary_mic_facts.provider_calendar_id

    def mic_facts_for(self, mic: str | None = None) -> MicFacts:
        target = (mic or self.primary_mic).strip().upper()
        for facts in self.mic_facts:
            if facts.mic == target:
                return facts
        supported = ", ".join(self.mics)
        raise MarketCatalogError(
            f"Unsupported MIC {mic!r} for market {self.code}. Supported: {supported}"
        )

    def as_runtime_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "label": self.label,
            "primary_mic": self.primary_mic,
            "mics": list(self.mics),
            "supported_currencies": list(self.supported_currencies),
            "default_currency": self.default_currency,
            "mic_facts": [asdict(facts) for facts in self.mic_facts],
            # Compatibility fields for existing frontend/runtime consumers.
            "currency": self.currency,
            "timezone": self.timezone,
            "calendar_id": self.calendar_id,
            "provider_calendar_id": self.provider_calendar_id,
            "exchanges": list(self.exchanges),
            "indexes": list(self.indexes),
            "capabilities": asdict(self.capabilities),
        }


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


def _mic_facts(
    mic: str,
    *,
    timezone: str,
    default_currency: str,
    calendar_id: str | None = None,
    provider_calendar_id: str | None = None,
) -> MicFacts:
    return MicFacts(
        mic=mic,
        calendar_id=calendar_id or mic,
        timezone=timezone,
        default_currency=default_currency,
        provider_calendar_id=provider_calendar_id,
    )


def _market_entry(
    *,
    code: str,
    label: str,
    primary_mic: str,
    mic_facts: tuple[MicFacts, ...],
    exchanges: tuple[str, ...],
    indexes: tuple[str, ...],
    capabilities: MarketCapabilities,
    supported_currencies: tuple[str, ...] | None = None,
    default_currency: str | None = None,
) -> MarketCatalogEntry:
    primary_mic_code = primary_mic.strip().upper()
    primary_facts = next(
        (facts for facts in mic_facts if facts.mic == primary_mic_code),
        None,
    )
    if primary_facts is None:
        raise ValueError(f"{code} primary MIC must have MIC facts")
    derived_supported_currencies = tuple(
        dict.fromkeys(facts.default_currency for facts in mic_facts)
    )
    return MarketCatalogEntry(
        code=code,
        label=label,
        primary_mic=primary_mic_code,
        mics=tuple(facts.mic for facts in mic_facts),
        supported_currencies=supported_currencies or derived_supported_currencies,
        default_currency=default_currency or primary_facts.default_currency,
        mic_facts=mic_facts,
        exchanges=exchanges,
        indexes=indexes,
        capabilities=capabilities,
    )


MARKET_CATALOG = MarketCatalog(
    [
        _market_entry(
            code="US",
            label="United States",
            primary_mic="XNYS",
            mic_facts=(
                _mic_facts(
                    "XNYS",
                    timezone="America/New_York",
                    default_currency="USD",
                ),
                _mic_facts(
                    "XNAS",
                    timezone="America/New_York",
                    default_currency="USD",
                ),
                _mic_facts(
                    "XASE",
                    timezone="America/New_York",
                    default_currency="USD",
                ),
            ),
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
        _market_entry(
            code="HK",
            label="Hong Kong",
            primary_mic="XHKG",
            mic_facts=(
                _mic_facts(
                    "XHKG",
                    timezone="Asia/Hong_Kong",
                    default_currency="HKD",
                ),
            ),
            exchanges=("HKEX", "SEHK", "XHKG"),
            indexes=("HSI",),
            capabilities=FULL_CAPABILITIES,
        ),
        _market_entry(
            code="IN",
            label="India",
            primary_mic="XNSE",
            mic_facts=(
                _mic_facts(
                    "XNSE",
                    timezone="Asia/Kolkata",
                    default_currency="INR",
                    provider_calendar_id="NSE",
                ),
                _mic_facts(
                    "XBOM",
                    timezone="Asia/Kolkata",
                    default_currency="INR",
                ),
            ),
            exchanges=("NSE", "XNSE", "BSE", "XBOM"),
            indexes=(),
            capabilities=FULL_CAPABILITIES,
        ),
        _market_entry(
            code="JP",
            label="Japan",
            primary_mic="XTKS",
            mic_facts=(
                _mic_facts(
                    "XTKS",
                    timezone="Asia/Tokyo",
                    default_currency="JPY",
                ),
            ),
            exchanges=("TSE", "JPX", "XTKS"),
            indexes=("NIKKEI225",),
            capabilities=FULL_CAPABILITIES,
        ),
        _market_entry(
            code="KR",
            label="South Korea",
            primary_mic="XKRX",
            mic_facts=(
                _mic_facts(
                    "XKRX",
                    timezone="Asia/Seoul",
                    default_currency="KRW",
                ),
            ),
            exchanges=("KOSPI", "KOSDAQ", "KRX", "XKRX"),
            indexes=(),
            capabilities=FULL_CAPABILITIES,
        ),
        _market_entry(
            code="TW",
            label="Taiwan",
            primary_mic="XTAI",
            mic_facts=(
                _mic_facts(
                    "XTAI",
                    timezone="Asia/Taipei",
                    default_currency="TWD",
                ),
            ),
            exchanges=("TWSE", "TPEX", "XTAI"),
            indexes=("TAIEX",),
            capabilities=FULL_CAPABILITIES,
        ),
        _market_entry(
            code="CN",
            label="China A-shares",
            primary_mic="XSHG",
            mic_facts=(
                _mic_facts(
                    "XSHG",
                    timezone="Asia/Shanghai",
                    default_currency="CNY",
                ),
                _mic_facts(
                    "XSHE",
                    timezone="Asia/Shanghai",
                    default_currency="CNY",
                ),
                _mic_facts(
                    "XBSE",
                    timezone="Asia/Shanghai",
                    default_currency="CNY",
                ),
            ),
            exchanges=("SSE", "SZSE", "BJSE", "XSHG", "XSHE", "XBSE"),
            indexes=(),
            capabilities=FULL_CAPABILITIES,
        ),
        _market_entry(
            code="CA",
            label="Canada",
            primary_mic="XTSE",
            mic_facts=(
                _mic_facts(
                    "XTSE",
                    timezone="America/Toronto",
                    default_currency="CAD",
                ),
                _mic_facts(
                    "XTNX",
                    timezone="America/Toronto",
                    default_currency="CAD",
                ),
            ),
            exchanges=("TSX", "TSXV", "XTSE", "XTNX"),
            indexes=("TSX_COMPOSITE",),
            capabilities=FULL_CAPABILITIES,
        ),
        _market_entry(
            code="DE",
            label="Germany",
            primary_mic="XETR",
            mic_facts=(
                _mic_facts(
                    "XETR",
                    timezone="Europe/Berlin",
                    default_currency="EUR",
                ),
                _mic_facts(
                    "XFRA",
                    timezone="Europe/Berlin",
                    default_currency="EUR",
                ),
            ),
            exchanges=("XETR", "XETRA", "XFRA", "FRA", "FWB"),
            indexes=("DAX", "MDAX", "SDAX"),
            capabilities=MarketCapabilities(
                benchmark=True,
                breadth=True,
                fundamentals=True,
                group_rankings=False,
                feature_snapshot=True,
                official_universe=True,
                finviz_screening=False,
            ),
        ),
        _market_entry(
            code="SG",
            label="Singapore",
            primary_mic="XSES",
            mic_facts=(
                _mic_facts(
                    "XSES",
                    timezone="Asia/Singapore",
                    default_currency="SGD",
                ),
            ),
            exchanges=("SGX", "SES", "XSES"),
            indexes=("STI",),
            capabilities=MarketCapabilities(
                benchmark=True,
                breadth=False,
                fundamentals=True,
                group_rankings=False,
                feature_snapshot=True,
                official_universe=True,
                finviz_screening=False,
            ),
        ),
    ]
)


def get_market_catalog() -> MarketCatalog:
    return MARKET_CATALOG
