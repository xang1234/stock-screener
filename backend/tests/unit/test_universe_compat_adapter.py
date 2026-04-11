"""Unit tests for legacy universe compatibility adapter."""

import json

from app.schemas.universe import Exchange, UniverseDefinition, UniverseType
from app.services.universe_compat_adapter import (
    LEGACY_UNIVERSE_SUNSET_HTTP,
    resolve_scan_universe_request,
)


def test_structured_universe_def_takes_precedence():
    typed = UniverseDefinition(type=UniverseType.EXCHANGE, exchange=Exchange.NASDAQ)
    resolved = resolve_scan_universe_request(
        universe_def=typed,
        legacy_universe="all",
        legacy_symbols=None,
    )
    assert resolved.universe_def is typed
    assert resolved.used_legacy is False
    assert resolved.deprecation_headers() == {}


def test_legacy_universe_maps_to_typed_definition_with_headers():
    resolved = resolve_scan_universe_request(
        universe_def=None,
        legacy_universe="nyse",
        legacy_symbols=None,
    )
    assert resolved.used_legacy is True
    assert resolved.universe_def.type == UniverseType.EXCHANGE
    assert resolved.universe_def.exchange == Exchange.NYSE
    headers = resolved.deprecation_headers()
    assert headers["Deprecation"] == "true"
    assert headers["Sunset"] == LEGACY_UNIVERSE_SUNSET_HTTP
    assert headers["X-Universe-Legacy-Value"] == "nyse"
    assert headers["X-Universe-Compat-Mode"] == "legacy"
    assert json.loads(headers["X-Universe-Migration-Hint"]) == {
        "type": "exchange",
        "exchange": "NYSE",
    }


def test_unknown_legacy_value_with_symbols_falls_back_to_custom():
    resolved = resolve_scan_universe_request(
        universe_def=None,
        legacy_universe="legacy_payload_v1",
        legacy_symbols=["AAPL", "MSFT"],
    )
    assert resolved.used_legacy is True
    assert resolved.universe_def.type == UniverseType.CUSTOM
    assert resolved.universe_def.symbols == ["AAPL", "MSFT"]
    assert resolved.migration_hint == {"type": "custom", "symbols_count": 2}


def test_active_alias_maps_to_all_for_backward_compat():
    resolved = resolve_scan_universe_request(
        universe_def=None,
        legacy_universe="active",
        legacy_symbols=None,
    )
    assert resolved.used_legacy is True
    assert resolved.universe_def.type == UniverseType.ALL
    # normalized to canonical legacy value
    assert resolved.legacy_value == "all"
    assert resolved.migration_hint == {"type": "all"}
