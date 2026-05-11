"""Compatibility adapter for legacy scan universe request fields.

This module keeps legacy `universe` string callers functional while steering
clients to structured `universe_def` payloads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

from ..schemas.universe import Market, UniverseDefinition

# Time-bounded compatibility policy.
LEGACY_UNIVERSE_DEPRECATION_DATE = date(2026, 4, 11)
LEGACY_UNIVERSE_SUNSET_DATE = date(2026, 10, 31)
LEGACY_UNIVERSE_SUNSET_HTTP = "Sat, 31 Oct 2026 00:00:00 GMT"

# Canonical alias map used by the legacy adapter. Values represent
# the equivalent typed payload shape for migration guidance.
LEGACY_UNIVERSE_ALIAS_MAP: dict[str, dict[str, Any]] = {
    "all": {"type": "all"},
    "active": {"type": "all"},
    "nyse": {"type": "exchange", "exchange": "NYSE"},
    "nasdaq": {"type": "exchange", "exchange": "NASDAQ"},
    "amex": {"type": "exchange", "exchange": "AMEX"},
    "sp500": {"type": "index", "index": "SP500"},
    "market:us": {"type": "market", "market": "US"},
    "market:hk": {"type": "market", "market": "HK"},
    "market:jp": {"type": "market", "market": "JP"},
    "market:kr": {"type": "market", "market": "KR"},
    "market:tw": {"type": "market", "market": "TW"},
    "market:cn": {"type": "market", "market": "CN"},
    **{
        f"market:{market.value.lower()}": {"type": "market", "market": market.value}
        for market in Market
    },
    "custom": {"type": "custom"},
    "test": {"type": "test"},
}


@dataclass(frozen=True)
class UniverseCompatResolution:
    """Result of resolving typed or legacy universe request payloads."""

    universe_def: UniverseDefinition
    used_legacy: bool
    legacy_value: Optional[str] = None
    migration_hint: Optional[dict[str, Any]] = None

    def deprecation_headers(self) -> dict[str, str]:
        """HTTP headers exposing legacy-path deprecation metadata."""
        if not self.used_legacy:
            return {}

        hint = self.migration_hint or {"type": self.universe_def.type.value}
        return {
            "Deprecation": "true",
            "Sunset": LEGACY_UNIVERSE_SUNSET_HTTP,
            "X-Universe-Compat-Mode": "legacy",
            "X-Universe-Legacy-Value": self.legacy_value or "unknown",
            "X-Universe-Migration-Hint": json.dumps(hint, separators=(",", ":")),
        }

    def deprecation_log_message(self) -> str:
        """Structured deprecation guidance for server logs."""
        return (
            "Legacy scan universe string is deprecated; "
            f"legacy_value={self.legacy_value!r}, "
            f"typed_key={self.universe_def.key()!r}, "
            f"deprecation_date={LEGACY_UNIVERSE_DEPRECATION_DATE.isoformat()}, "
            f"sunset_date={LEGACY_UNIVERSE_SUNSET_DATE.isoformat()}"
        )


def resolve_scan_universe_request(
    *,
    universe_def: UniverseDefinition | dict[str, Any] | None,
    legacy_universe: str,
    legacy_symbols: list[str] | None,
) -> UniverseCompatResolution:
    """Resolve scan request universe inputs with compatibility semantics.

    Structured `universe_def` takes precedence. Legacy `universe` is parsed
    only when `universe_def` is absent.
    """
    if universe_def is not None:
        parsed = (
            universe_def
            if isinstance(universe_def, UniverseDefinition)
            else UniverseDefinition.model_validate(universe_def)
        )
        return UniverseCompatResolution(
            universe_def=parsed,
            used_legacy=False,
        )

    normalized_legacy = (legacy_universe or "all").strip().lower()
    if normalized_legacy == "active":
        normalized_legacy = "all"
    parsed = UniverseDefinition.from_legacy(normalized_legacy, legacy_symbols)
    migration_hint = LEGACY_UNIVERSE_ALIAS_MAP.get(normalized_legacy)
    if migration_hint is None:
        migration_hint = {"type": parsed.type.value}
        if parsed.market is not None:
            migration_hint["market"] = parsed.market.value
        if parsed.exchange is not None:
            migration_hint["exchange"] = parsed.exchange.value
        if parsed.index is not None:
            migration_hint["index"] = parsed.index.value
        if parsed.symbols is not None:
            migration_hint["symbols_count"] = len(parsed.symbols)

    return UniverseCompatResolution(
        universe_def=parsed,
        used_legacy=True,
        legacy_value=normalized_legacy,
        migration_hint=migration_hint,
    )
