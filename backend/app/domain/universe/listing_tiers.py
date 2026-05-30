"""Listing tier normalization definitions for official universe ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class ListingTierDefinition:
    key: str
    label: str
    market: str
    mic: str | None = None
    aliases: tuple[str, ...] = ()


class ListingTierRegistry:
    """Market-scoped listing tier lookup for ingestion adapters."""

    def __init__(self, definitions: Iterable[ListingTierDefinition]) -> None:
        self._definitions = tuple(definitions)
        self._by_scoped_alias: dict[tuple[str, str | None, str], str] = {}
        self._by_market_wide_alias: dict[tuple[str, str], str] = {}
        self._by_market_alias_candidates: dict[tuple[str, str], set[str]] = {}

        for definition in self._definitions:
            market = definition.market.upper()
            definition_mic = definition.mic.strip().upper() if definition.mic else None
            aliases = (definition.key, definition.label, *definition.aliases)
            for alias in aliases:
                normalized_alias = self._normalize_alias(alias)
                if not normalized_alias:
                    continue
                scoped_key = (market, definition_mic, normalized_alias)
                market_key = (market, normalized_alias)
                if (
                    scoped_key in self._by_scoped_alias
                    and self._by_scoped_alias[scoped_key] != definition.key
                ):
                    raise ValueError(
                        f"Duplicate listing tier alias for {market}: {alias!r}"
                    )
                self._by_scoped_alias[scoped_key] = definition.key
                self._by_market_alias_candidates.setdefault(market_key, set()).add(
                    definition.key
                )
                if definition_mic is None:
                    existing_market_match = self._by_market_wide_alias.get(market_key)
                    if (
                        existing_market_match is not None
                        and existing_market_match != definition.key
                    ):
                        raise ValueError(
                            f"Duplicate market-wide listing tier alias for "
                            f"{market}: {alias!r}"
                        )
                    self._by_market_wide_alias[market_key] = definition.key

    def definitions(
        self, market: str | None = None, *, mic: str | None = None
    ) -> tuple[ListingTierDefinition, ...]:
        market_code = str(market or "").strip().upper()
        mic_code = str(mic).strip().upper() if mic else None
        return tuple(
            definition
            for definition in self._definitions
            if (not market_code or definition.market == market_code)
            and (
                mic_code is None
                or (definition.mic is not None and definition.mic.upper() == mic_code)
            )
        )

    def normalize(
        self, market: str, raw: str | None, *, mic: str | None = None
    ) -> str | None:
        normalized_alias = self._normalize_alias(raw)
        if not normalized_alias:
            return None

        market_code = str(market or "").strip().upper()
        mic_code = str(mic).strip().upper() if mic else None
        if mic_code:
            scoped_match = self._by_scoped_alias.get(
                (market_code, mic_code, normalized_alias)
            )
            if scoped_match:
                return scoped_match
            return self._by_market_wide_alias.get((market_code, normalized_alias))

        candidates = self._by_market_alias_candidates.get(
            (market_code, normalized_alias),
            set(),
        )
        if len(candidates) == 1:
            return next(iter(candidates))
        return None

    @staticmethod
    def _normalize_alias(value: str | None) -> str:
        return " ".join(str(value or "").strip().upper().replace("_", " ").split())


listing_tier_registry = ListingTierRegistry(
    (
        ListingTierDefinition(
            key="main_board",
            label="Main Board",
            market="HK",
            mic="XHKG",
            aliases=("MAIN", "HKEX MAIN BOARD", "SEHK MAIN BOARD"),
        ),
        ListingTierDefinition(
            key="gem",
            label="GEM",
            market="HK",
            mic="XHKG",
            aliases=("GROWTH ENTERPRISE MARKET", "HKEX GEM"),
        ),
        ListingTierDefinition(
            key="mainboard",
            label="Mainboard",
            market="SG",
            mic="XSES",
            aliases=("MAIN", "MAIN BOARD", "SGX MAINBOARD"),
        ),
        ListingTierDefinition(
            key="catalist",
            label="Catalist",
            market="SG",
            mic="XSES",
            aliases=("SGX CATALIST",),
        ),
        ListingTierDefinition(
            key="main",
            label="Main",
            market="AU",
            mic="XASX",
            aliases=("MAIN BOARD", "ASX MAIN", "ASX"),
        ),
    )
)
