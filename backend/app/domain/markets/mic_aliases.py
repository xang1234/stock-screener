"""Market-scoped compatibility aliases for canonical MIC lookup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .catalog import MarketCatalogError, get_market_catalog


@dataclass(frozen=True, slots=True)
class MicAliasDefinition:
    market: str
    mic: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MicAliasResolution:
    market: str
    mic: str
    alias: str


class MicAliasRegistry:
    """Resolve legacy exchange labels to MICs with explicit Market scope."""

    def __init__(self, definitions: Iterable[MicAliasDefinition]) -> None:
        self._definitions = tuple(definitions)
        self._by_market_alias: dict[tuple[str, str], MicAliasResolution] = {}
        self._global_alias_candidates: dict[str, set[tuple[str, str]]] = {}

        catalog = get_market_catalog()
        for definition in self._definitions:
            market = definition.market.strip().upper()
            mic = definition.mic.strip().upper()
            try:
                market_entry = catalog.get(market)
            except MarketCatalogError as exc:
                raise ValueError(f"Unsupported MIC alias market: {market}") from exc
            if mic not in market_entry.mics:
                raise ValueError(f"{market} MIC alias target is not in Market Catalog: {mic}")

            aliases = (mic, *definition.aliases)
            for raw_alias in aliases:
                alias = self._normalize_alias(raw_alias)
                if not alias:
                    continue
                scoped_key = (market, alias)
                existing = self._by_market_alias.get(scoped_key)
                if existing is not None and existing.mic != mic:
                    raise ValueError(
                        f"Duplicate MIC alias for {market}: {alias!r} maps to "
                        f"{existing.mic} and {mic}"
                    )
                self._by_market_alias[scoped_key] = MicAliasResolution(
                    market=market,
                    mic=mic,
                    alias=alias,
                )
                self._global_alias_candidates.setdefault(alias, set()).add((market, mic))

    def resolve(self, market: str | None, alias: str | None) -> MicAliasResolution | None:
        market_code = str(market or "").strip().upper()
        normalized_alias = self._normalize_alias(alias)
        if not market_code or not normalized_alias:
            return None
        return self._by_market_alias.get((market_code, normalized_alias))

    def resolve_global(self, alias: str | None) -> MicAliasResolution | None:
        normalized_alias = self._normalize_alias(alias)
        if not normalized_alias:
            return None
        candidates = self._global_alias_candidates.get(normalized_alias, set())
        if len(candidates) != 1:
            return None
        market, mic = next(iter(candidates))
        return MicAliasResolution(market=market, mic=mic, alias=normalized_alias)

    def market_for_alias(self, alias: str | None) -> str | None:
        resolved = self.resolve_global(alias)
        return resolved.market if resolved else None

    def aliases(self, market: str | None = None) -> tuple[str, ...]:
        market_code = str(market or "").strip().upper()
        return tuple(
            alias
            for scoped_market, alias in self._by_market_alias
            if not market_code or scoped_market == market_code
        )

    def aliases_for_mic(self, market: str | None, mic: str | None) -> tuple[str, ...]:
        market_code = str(market or "").strip().upper()
        mic_code = str(mic or "").strip().upper()
        if not market_code or not mic_code:
            return ()
        return tuple(
            alias
            for (scoped_market, alias), resolution in self._by_market_alias.items()
            if scoped_market == market_code and resolution.mic == mic_code
        )

    def is_ambiguous(self, alias: str | None) -> bool:
        normalized_alias = self._normalize_alias(alias)
        if not normalized_alias:
            return False
        return len(self._global_alias_candidates.get(normalized_alias, set())) > 1

    @staticmethod
    def _normalize_alias(value: str | None) -> str:
        return str(value or "").strip().upper()


mic_alias_registry = MicAliasRegistry(
    (
        MicAliasDefinition("US", "XNYS", ("NYSE",)),
        MicAliasDefinition("US", "XNAS", ("NASDAQ",)),
        MicAliasDefinition("US", "XASE", ("AMEX",)),
        MicAliasDefinition("HK", "XHKG", ("HKEX", "SEHK")),
        MicAliasDefinition("IN", "XNSE", ("NSE",)),
        MicAliasDefinition("IN", "XBOM", ("BSE",)),
        MicAliasDefinition("JP", "XTKS", ("TSE", "JPX")),
        MicAliasDefinition("KR", "XKRX", ("KOSPI", "KOSDAQ", "KRX")),
        MicAliasDefinition("TW", "XTAI", ("TWSE", "TPEX")),
        MicAliasDefinition("CN", "XSHG", ("SSE", "SHSE")),
        MicAliasDefinition("CN", "XSHE", ("SZSE",)),
        MicAliasDefinition("CN", "XBSE", ("BSE", "BJSE", "XBEI")),
        MicAliasDefinition("CA", "XTSE", ("TSX",)),
        MicAliasDefinition("CA", "XTNX", ("TSXV",)),
        MicAliasDefinition("DE", "XETR", ("XETRA",)),
        MicAliasDefinition("DE", "XFRA", ("FRA", "FWB")),
        MicAliasDefinition("SG", "XSES", ("SGX", "SES")),
        MicAliasDefinition("AU", "XASX", ("ASX",)),
        MicAliasDefinition("MY", "XKLS", ("KLSE", "MYX", "BURSA")),
    )
)
