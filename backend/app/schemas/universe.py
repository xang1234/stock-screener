"""
Universe definition schema — single source of truth for scan universes.

Replaces the loose `universe: str` pattern with a typed, validated model
used across the API, DB persistence, symbol resolution, and retention.
"""
import hashlib
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, field_validator, model_validator

from ..domain.markets.catalog import get_market_catalog
from ..domain.markets.mic_aliases import mic_alias_registry
from ..domain.universe import (
    UniverseStorageProjection,
    UniverseType,
    normalize_market_scope,
    parse_market_key_components,
    validate_legacy_exchange_scope,
)
from ..domain.universe.indexes import index_registry


def _enum_member_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.upper()).strip("_")


def _make_str_enum(name: str, values: tuple[str, ...] | list[str]) -> type[Enum]:
    members = {_enum_member_name(value): value for value in values}
    return Enum(name, members, type=str, module=__name__)


Market = _make_str_enum(
    "Market",
    get_market_catalog().supported_market_codes(),
)


Exchange = _make_str_enum(
    "Exchange",
    mic_alias_registry.aliases(),
)


IndexName = _make_str_enum(
    "IndexName",
    index_registry.supported_index_keys(),
)


class UniverseDefinition(BaseModel):
    """
    Structured definition of a scan universe.

    Validates that the right fields are populated for each universe type:
    - ALL: no extra fields
    - MARKET: market required
    - EXCHANGE: exchange required, market optional for ambiguous exchange codes
    - INDEX: index required
    - CUSTOM/TEST: symbols required (non-empty, max 500)
    """
    type: UniverseType
    market: Optional[Market] = None
    mic: Optional[str] = None
    exchange: Optional[Exchange] = None
    index: Optional[IndexName] = None
    listing_tier: Optional[str] = None
    symbols: Optional[List[str]] = None
    allow_inactive_symbols: bool = False

    @field_validator("mic", mode="before")
    @classmethod
    def normalize_mic(cls, value):
        if value is None:
            return value
        normalized = str(value).strip().upper()
        return normalized or None

    @field_validator("listing_tier", mode="before")
    @classmethod
    def normalize_listing_tier_input(cls, value):
        if value is None:
            return value
        normalized = str(value).strip()
        return normalized or None

    @field_validator("symbols", mode="before")
    @classmethod
    def normalize_symbols(cls, v):
        """Trim, uppercase, deduplicate, and remove empty symbols."""
        if v is None:
            return v
        seen = set()
        normalized = []
        for s in v:
            if not isinstance(s, str):
                continue
            s = s.strip().upper()
            if s and s not in seen:
                seen.add(s)
                normalized.append(s)
        return normalized

    @model_validator(mode="after")
    def validate_fields_for_type(self):
        """Ensure correct fields are set for the universe type."""
        t = self.type

        if t == UniverseType.ALL:
            if (
                self.exchange is not None
                or self.market is not None
                or self.mic is not None
                or self.index is not None
                or self.listing_tier is not None
                or self.symbols is not None
                or self.allow_inactive_symbols
            ):
                raise ValueError(
                    "ALL universe must not specify market, mic, exchange, index, "
                    "listing_tier, or symbols"
                )

        elif t == UniverseType.MARKET:
            if self.market is None:
                raise ValueError("MARKET universe requires 'market' field")
            if (
                self.index is not None
                or self.symbols is not None
                or self.allow_inactive_symbols
            ):
                raise ValueError("MARKET universe must not specify index or symbols")
            self._validate_market_mic_and_listing_tier()

        elif t == UniverseType.EXCHANGE:
            if self.exchange is None:
                raise ValueError("EXCHANGE universe requires 'exchange' field")
            if (
                self.mic is not None
                or self.index is not None
                or self.listing_tier is not None
                or self.symbols is not None
                or self.allow_inactive_symbols
            ):
                raise ValueError(
                    "EXCHANGE universe must not specify mic, index, listing_tier, "
                    "or symbols"
                )
            self._validate_legacy_exchange_scope()

        elif t == UniverseType.INDEX:
            if self.index is None:
                raise ValueError("INDEX universe requires 'index' field")
            if (
                self.market is not None
                or self.mic is not None
                or self.exchange is not None
                or self.listing_tier is not None
                or self.symbols is not None
                or self.allow_inactive_symbols
            ):
                raise ValueError(
                    "INDEX universe must not specify market, mic, exchange, "
                    "listing_tier, or symbols"
                )

        elif t in (UniverseType.CUSTOM, UniverseType.TEST):
            if not self.symbols:
                raise ValueError(f"{t.value.upper()} universe requires a non-empty 'symbols' list")
            if len(self.symbols) > 500:
                raise ValueError(f"Symbol list too long ({len(self.symbols)}). Maximum is 500.")
            if (
                self.market is not None
                or self.mic is not None
                or self.exchange is not None
                or self.index is not None
                or self.listing_tier is not None
            ):
                raise ValueError(
                    f"{t.value.upper()} universe must not specify market, mic, "
                    "exchange, index, or listing_tier"
                )

        return self

    def _validate_market_mic_and_listing_tier(self) -> None:
        normalized = normalize_market_scope(
            self.market.value,
            mic=self.mic,
            exchange=self.exchange.value if self.exchange else None,
            listing_tier=self.listing_tier,
        )
        self.mic = normalized.mic
        self.listing_tier = normalized.listing_tier

    def _validate_legacy_exchange_scope(self) -> None:
        validate_legacy_exchange_scope(
            self.exchange.value,
            market=self.market.value if self.market else None,
        )

    def key(self) -> str:
        """
        Canonical key for DB indexing and retention grouping.

        Returns:
            Deterministic string key:
            - "all"
            - "market:US"
            - "exchange:NYSE"
            - "index:SP500"
            - "custom:<sha256[:12]>"
            - "test:<sha256[:12]>"
        """
        if self.type == UniverseType.ALL:
            return "all"
        elif self.type == UniverseType.MARKET:
            key = f"market:{self.market.value}"
            if self.mic is not None:
                key = f"{key}:mic:{self.mic}"
            if self.listing_tier is not None:
                key = f"{key}:tier:{self.listing_tier}"
            return key
        elif self.type == UniverseType.EXCHANGE:
            if self.market is not None and self.market != Market.US:
                return f"exchange:{self.market.value}:{self.exchange.value}"
            return f"exchange:{self.exchange.value}"
        elif self.type == UniverseType.INDEX:
            return f"index:{self.index.value}"
        else:
            # CUSTOM or TEST — hash the sorted symbol list
            joined = ",".join(sorted(self.symbols))
            digest = hashlib.sha256(joined.encode()).hexdigest()[:12]
            inactive_suffix = ":inactive" if self.allow_inactive_symbols else ""
            return f"{self.type.value}:{digest}{inactive_suffix}"

    def storage_projection(self) -> UniverseStorageProjection:
        return UniverseStorageProjection(
            label=self.label(),
            key=self.key(),
            type=self.type.value,
            market=self.market.value if self.market else None,
            exchange=self.mic or (self.exchange.value if self.exchange else None),
            index=self.index.value if self.index else None,
            symbols=self.symbols,
        )

    def label(self) -> str:
        """
        Human-readable label for display.

        Returns:
            e.g. "All Stocks", "US Market", "NYSE", "S&P 500", "Custom (25 symbols)", "Test (5 symbols)"
        """
        if self.type == UniverseType.ALL:
            return "All Stocks"
        elif self.type == UniverseType.MARKET:
            market_labels = {
                Market.US: "US Market",
                Market.HK: "Hong Kong Market",
                Market.IN: "India Market",
                Market.JP: "Japan Market",
                Market.KR: "South Korea Market",
                Market.TW: "Taiwan Market",
                Market.CN: "China A-shares",
                Market.SG: "Singapore Market",
            }
            return market_labels.get(self.market, f"{self.market.value} Market")
        elif self.type == UniverseType.EXCHANGE:
            return self.exchange.value
        elif self.type == UniverseType.INDEX:
            return index_registry.label_for(self.index.value) or self.index.value
        elif self.type == UniverseType.CUSTOM:
            n = len(self.symbols)
            suffix = ", incl. inactive" if self.allow_inactive_symbols else ""
            return f"Custom ({n} {'symbol' if n == 1 else 'symbols'}{suffix})"
        else:  # TEST
            n = len(self.symbols)
            suffix = ", incl. inactive" if self.allow_inactive_symbols else ""
            return f"Test ({n} {'symbol' if n == 1 else 'symbols'}{suffix})"

    @classmethod
    def from_scan_fields(
        cls,
        *,
        universe_type: Optional[str],
        universe: Optional[str] = None,
        universe_key: Optional[str] = None,
        universe_market: Optional[str] = None,
        universe_exchange: Optional[str] = None,
        universe_index: Optional[str] = None,
        universe_symbols: Optional[List[str]] = None,
    ) -> "UniverseDefinition":
        """Reconstruct a UniverseDefinition from persisted scan metadata columns.

        Falls back to :meth:`from_legacy` when ``universe_type`` is NULL
        (pre-migration rows). Any reconstruction failure — unknown type,
        unknown market/exchange/index enum value, or a Pydantic validator
        rejection — yields an ALL definition rather than raising, because
        callers are typically building response payloads for scans that
        already exist and should not 500 on a malformed historical row.
        """
        if universe_type is None:
            try:
                return cls.from_legacy(universe or "all", universe_symbols)
            except Exception:
                return cls(type=UniverseType.ALL)

        try:
            parsed_type = UniverseType(universe_type)

            resolved_market = universe_market
            market = exchange = index = None
            mic = listing_tier = None
            symbols: Optional[List[str]] = None

            if parsed_type == UniverseType.MARKET:
                key_parts = parse_market_key_components(universe_key)
                if (
                    resolved_market is None
                    and key_parts.get("market")
                ):
                    resolved_market = key_parts["market"]
                market = Market(resolved_market) if resolved_market else None
                mic = key_parts.get("mic")
                listing_tier = key_parts.get("tier")
                if mic is None and universe_exchange:
                    exchange = Exchange(universe_exchange)
            elif parsed_type == UniverseType.EXCHANGE:
                exchange = Exchange(universe_exchange) if universe_exchange else None
            elif parsed_type == UniverseType.INDEX:
                index = IndexName(universe_index) if universe_index else None
            elif parsed_type in (UniverseType.CUSTOM, UniverseType.TEST):
                symbols = universe_symbols

            return cls(
                type=parsed_type,
                market=market,
                mic=mic,
                exchange=exchange,
                index=index,
                listing_tier=listing_tier,
                symbols=symbols,
            )
        except Exception:
            return cls(type=UniverseType.ALL)

    @classmethod
    def from_legacy(cls, universe: str, symbols: Optional[List[str]] = None) -> "UniverseDefinition":
        """
        Parse a legacy universe string into a typed UniverseDefinition.

        Supports backward compatibility with existing API calls:
        - "all" -> ALL
        - "market:us" -> MARKET:US
        - "market:hk" -> MARKET:HK
        - "market:jp" -> MARKET:JP
        - "market:tw" -> MARKET:TW
        - "market:cn" -> MARKET:CN
        - "market:sg" -> MARKET:SG
        - "nyse" -> EXCHANGE:NYSE
        - "nasdaq" -> EXCHANGE:NASDAQ
        - "amex" -> EXCHANGE:AMEX
        - "sp500" -> INDEX:SP500
        - "custom" -> CUSTOM (requires symbols)
        - "test" -> TEST (requires symbols)

        Args:
            universe: Legacy universe string
            symbols: Optional symbol list for custom/test

        Returns:
            UniverseDefinition instance
        """
        u = universe.strip().lower()

        exchange_map = {exchange.value.lower(): exchange for exchange in Exchange}
        market_map = {market.value.lower(): market for market in Market}
        valid_markets = ", ".join(market.value for market in Market)
        valid_market_universes = ", ".join(
            f"market:{market.value.lower()}" for market in Market
        )
        index_key = index_registry.normalize(u)

        if u == "all":
            return cls(type=UniverseType.ALL)
        elif u.startswith("market:"):
            raw_market = u.split(":", 1)[1].strip().lower()
            if raw_market in market_map:
                return cls(type=UniverseType.MARKET, market=market_map[raw_market])
            raise ValueError(f"Unknown market '{raw_market}'. Valid values: {valid_markets}")
        elif u in exchange_map:
            return cls(type=UniverseType.EXCHANGE, exchange=exchange_map[u])
        elif index_key is not None:
            return cls(type=UniverseType.INDEX, index=IndexName(index_key))
        elif u == "custom":
            return cls(type=UniverseType.CUSTOM, symbols=symbols or [])
        elif u == "test":
            return cls(type=UniverseType.TEST, symbols=symbols or [])
        else:
            # Unknown legacy value — treat as custom with the value as context
            # This prevents data loss during migration
            if symbols:
                return cls(type=UniverseType.CUSTOM, symbols=symbols)
            raise ValueError(
                f"Unknown universe '{universe}' and no symbols provided. "
                f"Valid values: all, {valid_market_universes}, "
                f"nyse, nasdaq, amex, sp500, custom, test"
            )
