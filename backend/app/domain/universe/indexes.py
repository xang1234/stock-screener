"""Canonical index Universe definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from ..markets.market import SUPPORTED_MARKET_CODES


@dataclass(frozen=True, slots=True)
class IndexDefinition:
    key: str
    label: str
    market: str
    membership_source: str = "stock_universe_index_membership"
    aliases: tuple[str, ...] = ()


class IndexRegistry:
    """Backend-owned index definitions for Universe validation."""

    def __init__(self, definitions: Iterable[IndexDefinition]) -> None:
        normalized_definitions: list[IndexDefinition] = []
        self._by_key: dict[str, IndexDefinition] = {}
        self._by_alias: dict[str, str] = {}
        for definition in tuple(definitions):
            key = definition.key.strip().upper()
            market = definition.market.strip().upper()
            if market not in SUPPORTED_MARKET_CODES:
                raise ValueError(f"Unsupported index market: {market}")
            if key in self._by_key:
                raise ValueError(f"Duplicate index key: {key}")

            normalized = IndexDefinition(
                key=key,
                label=definition.label,
                market=market,
                membership_source=definition.membership_source,
                aliases=definition.aliases,
            )
            normalized_definitions.append(normalized)
            self._by_key[key] = normalized

            for alias in (key, definition.label, *definition.aliases):
                normalized_alias = self._normalize_alias(alias)
                if not normalized_alias:
                    continue
                existing = self._by_alias.get(normalized_alias)
                if existing is not None and existing != key:
                    raise ValueError(
                        f"Duplicate index alias {alias!r}: {existing} and {key}"
                    )
                self._by_alias[normalized_alias] = key

        self._definitions = tuple(normalized_definitions)

    def definitions(self, market: str | None = None) -> tuple[IndexDefinition, ...]:
        market_code = str(market or "").strip().upper()
        if not market_code:
            return self._definitions
        return tuple(
            definition
            for definition in self._definitions
            if definition.market == market_code
        )

    def supported_index_keys(self) -> tuple[str, ...]:
        return tuple(definition.key for definition in self._definitions)

    def get(self, value: str) -> IndexDefinition:
        normalized = self.normalize(value)
        if normalized is None:
            supported = ", ".join(self.supported_index_keys())
            raise ValueError(f"Unsupported index {value!r}. Supported: {supported}")
        return self._by_key[normalized]

    def normalize(self, value: str | None) -> str | None:
        normalized_alias = self._normalize_alias(value)
        if not normalized_alias:
            return None
        return self._by_alias.get(normalized_alias)

    def market_for(self, value: str | None) -> str | None:
        normalized = self.normalize(value)
        if normalized is None:
            return None
        return self._by_key[normalized].market

    def label_for(self, value: str | None) -> str | None:
        normalized = self.normalize(value)
        if normalized is None:
            return None
        return self._by_key[normalized].label

    def as_runtime_payload(self) -> list[dict[str, object]]:
        return [asdict(definition) for definition in self._definitions]

    @staticmethod
    def _normalize_alias(value: str | None) -> str:
        return " ".join(str(value or "").strip().upper().replace("_", " ").split())


index_registry = IndexRegistry(
    (
        IndexDefinition(
            key="SP500",
            label="S&P 500",
            market="US",
            membership_source="stock_universe.is_sp500",
            aliases=("S AND P 500", "SNP500", "SPX"),
        ),
        IndexDefinition(
            key="HSI",
            label="Hang Seng Index",
            market="HK",
            aliases=("HANG SENG",),
        ),
        IndexDefinition(
            key="NIFTY50",
            label="NIFTY 50",
            market="IN",
            aliases=("NIFTY", "NSEI"),
        ),
        IndexDefinition(
            key="NIKKEI225",
            label="Nikkei 225",
            market="JP",
            aliases=("NIKKEI", "N225"),
        ),
        IndexDefinition(
            key="KOSPI",
            label="KOSPI Composite",
            market="KR",
            aliases=("KOSPI COMPOSITE", "KS11"),
        ),
        IndexDefinition(
            key="TAIEX",
            label="TAIEX",
            market="TW",
            aliases=("TAIWAN CAPITALIZATION WEIGHTED STOCK INDEX", "TWII"),
        ),
        IndexDefinition(
            key="CSI300",
            label="CSI 300",
            market="CN",
            aliases=("CSI 300 INDEX",),
        ),
        IndexDefinition(
            key="TSX_COMPOSITE",
            label="S&P/TSX Composite",
            market="CA",
            aliases=("TSX COMPOSITE", "GSPTSE"),
        ),
        IndexDefinition(key="DAX", label="DAX", market="DE"),
        IndexDefinition(key="MDAX", label="MDAX", market="DE"),
        IndexDefinition(key="SDAX", label="SDAX", market="DE"),
        IndexDefinition(
            key="STI",
            label="Straits Times Index",
            market="SG",
            aliases=("STRAITS TIMES",),
        ),
        IndexDefinition(
            key="ASX200",
            label="S&P/ASX 200",
            market="AU",
            aliases=("ASX 200", "S&P ASX 200", "XJO", "AXJO"),
        ),
        IndexDefinition(
            key="FBMKLCI",
            label="FTSE Bursa Malaysia KLCI",
            market="MY",
            aliases=("FBM KLCI", "KLCI", "KLSE"),
        ),
    )
)
