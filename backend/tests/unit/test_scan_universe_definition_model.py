"""Tests for Scan.get_universe_definition reconstruction semantics."""

from app.models.scan_result import Scan
from app.schemas.universe import Exchange, Market, UniverseType


def test_get_universe_definition_ignores_market_for_non_market_types():
    scan = Scan(
        universe_type=UniverseType.ALL.value,
        universe_market="US",
    )

    universe_def = scan.get_universe_definition()

    assert universe_def.type == UniverseType.ALL
    assert universe_def.market is None


def test_get_universe_definition_derives_market_from_key_for_market_type():
    scan = Scan(
        universe_type=UniverseType.MARKET.value,
        universe_key="market:jp",
        universe_market=None,
    )

    universe_def = scan.get_universe_definition()

    assert universe_def.type == UniverseType.MARKET
    assert universe_def.market == Market.JP


def test_get_universe_definition_uses_type_specific_fields_only():
    scan = Scan(
        universe_type=UniverseType.EXCHANGE.value,
        universe_exchange="NASDAQ",
        universe_market="US",
        universe_symbols=["AAPL", "MSFT"],
    )

    universe_def = scan.get_universe_definition()

    assert universe_def.type == UniverseType.EXCHANGE
    assert universe_def.exchange == Exchange.NASDAQ
    assert universe_def.market is None
    assert universe_def.symbols is None


def test_get_universe_definition_falls_back_on_malformed_row():
    # Historical row where the market column is null and the key doesn't
    # carry the market either — Pydantic would reject MARKET + market=None,
    # so reconstruction must yield ALL rather than raise.
    scan = Scan(
        universe_type=UniverseType.MARKET.value,
        universe_key="stale-legacy-key",
        universe_market=None,
    )

    universe_def = scan.get_universe_definition()

    assert universe_def.type == UniverseType.ALL


def test_get_universe_definition_falls_back_on_unknown_enum_value():
    scan = Scan(
        universe_type=UniverseType.EXCHANGE.value,
        universe_exchange="LSE",  # not in Exchange enum
    )

    universe_def = scan.get_universe_definition()

    assert universe_def.type == UniverseType.ALL
