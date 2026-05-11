"""
Universe definition schema — single source of truth for scan universes.

Replaces the loose `universe: str` pattern with a typed, validated model
used across the API, DB persistence, symbol resolution, and retention.
"""
import hashlib
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, field_validator, model_validator


class UniverseType(str, Enum):
    """Type of stock universe to scan."""
    ALL = "all"
    MARKET = "market"
    EXCHANGE = "exchange"
    INDEX = "index"
    CUSTOM = "custom"
    TEST = "test"


class Market(str, Enum):
    """Supported market scopes."""
    US = "US"
    HK = "HK"
    IN = "IN"
    JP = "JP"
    KR = "KR"
    TW = "TW"
    CN = "CN"
    CA = "CA"
    DE = "DE"


class Exchange(str, Enum):
    """Supported stock exchanges."""
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    AMEX = "AMEX"
    KOSPI = "KOSPI"
    KOSDAQ = "KOSDAQ"
    SSE = "SSE"
    SZSE = "SZSE"
    BJSE = "BJSE"
    TSX = "TSX"
    TSXV = "TSXV"


class IndexName(str, Enum):
    """Supported stock market indices.

    ``SP500`` membership is stored on ``StockUniverse.is_sp500`` (legacy).
    Asia indices (``HSI``, ``NIKKEI225``, ``TAIEX``) resolve via the
    ``stock_universe_index_membership`` table so adding future indices
    only requires data, not a schema migration. ``TAIEX`` is narrowed to
    the top-50 constituents by weight to keep the scan set comparable to
    HSI/Nikkei-225 rather than near-whole-market TW coverage.
    """
    SP500 = "SP500"
    HSI = "HSI"
    NIKKEI225 = "NIKKEI225"
    TAIEX = "TAIEX"
    DAX = "DAX"
    MDAX = "MDAX"
    SDAX = "SDAX"


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
    exchange: Optional[Exchange] = None
    index: Optional[IndexName] = None
    symbols: Optional[List[str]] = None
    allow_inactive_symbols: bool = False

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
                or self.index is not None
                or self.symbols is not None
                or self.allow_inactive_symbols
            ):
                raise ValueError("ALL universe must not specify market, exchange, index, or symbols")

        elif t == UniverseType.MARKET:
            if self.market is None:
                raise ValueError("MARKET universe requires 'market' field")
            if (
                self.exchange is not None
                or self.index is not None
                or self.symbols is not None
                or self.allow_inactive_symbols
            ):
                raise ValueError("MARKET universe must not specify exchange, index, or symbols")

        elif t == UniverseType.EXCHANGE:
            if self.exchange is None:
                raise ValueError("EXCHANGE universe requires 'exchange' field")
            if (
                self.index is not None
                or self.symbols is not None
                or self.allow_inactive_symbols
            ):
                raise ValueError("EXCHANGE universe must not specify index or symbols")

        elif t == UniverseType.INDEX:
            if self.index is None:
                raise ValueError("INDEX universe requires 'index' field")
            if (
                self.market is not None
                or self.exchange is not None
                or self.symbols is not None
                or self.allow_inactive_symbols
            ):
                raise ValueError("INDEX universe must not specify market, exchange, or symbols")

        elif t in (UniverseType.CUSTOM, UniverseType.TEST):
            if not self.symbols:
                raise ValueError(f"{t.value.upper()} universe requires a non-empty 'symbols' list")
            if len(self.symbols) > 500:
                raise ValueError(f"Symbol list too long ({len(self.symbols)}). Maximum is 500.")
            if self.market is not None or self.exchange is not None or self.index is not None:
                raise ValueError(f"{t.value.upper()} universe must not specify market, exchange, or index")

        return self

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
            return f"market:{self.market.value}"
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
            }
            return market_labels.get(self.market, f"{self.market.value} Market")
        elif self.type == UniverseType.EXCHANGE:
            return self.exchange.value
        elif self.type == UniverseType.INDEX:
            if self.index == IndexName.SP500:
                return "S&P 500"
            return self.index.value
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
            symbols: Optional[List[str]] = None

            if parsed_type == UniverseType.MARKET:
                if (
                    resolved_market is None
                    and isinstance(universe_key, str)
                    and universe_key.lower().startswith("market:")
                ):
                    resolved_market = universe_key.split(":", 1)[1].upper()
                market = Market(resolved_market) if resolved_market else None
            elif parsed_type == UniverseType.EXCHANGE:
                market = Market(resolved_market) if resolved_market else None
                exchange = Exchange(universe_exchange) if universe_exchange else None
            elif parsed_type == UniverseType.INDEX:
                index = IndexName(universe_index) if universe_index else None
            elif parsed_type in (UniverseType.CUSTOM, UniverseType.TEST):
                symbols = universe_symbols

            return cls(
                type=parsed_type,
                market=market,
                exchange=exchange,
                index=index,
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

        exchange_map = {
            "nyse": Exchange.NYSE,
            "nasdaq": Exchange.NASDAQ,
            "amex": Exchange.AMEX,
        }
        market_map = {market.value.lower(): market for market in Market}
        valid_markets = ", ".join(market.value for market in Market)
        valid_market_universes = ", ".join(
            f"market:{market.value.lower()}" for market in Market
        )

        if u == "all":
            return cls(type=UniverseType.ALL)
        elif u.startswith("market:"):
            raw_market = u.split(":", 1)[1].strip().lower()
            if raw_market in market_map:
                return cls(type=UniverseType.MARKET, market=market_map[raw_market])
            raise ValueError(f"Unknown market '{raw_market}'. Valid values: {valid_markets}")
        elif u in exchange_map:
            return cls(type=UniverseType.EXCHANGE, exchange=exchange_map[u])
        elif u == "sp500":
            return cls(type=UniverseType.INDEX, index=IndexName.SP500)
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
