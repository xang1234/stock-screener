"""Tests for the scan creation endpoint and command building.

Core business logic (commit ordering, dispatch failure, idempotency)
is tested in test_create_scan_use_case.py.  These tests focus on the
router layer: request → command translation and universe parsing.
"""

import pytest

from app.api.v1.scans import _build_universe_def
from app.schemas.scanning import ScanCreateRequest
from app.schemas.universe import Exchange, IndexName, UniverseDefinition, UniverseType
from app.use_cases.scanning.create_scan import CreateScanCommand


# ── _build_universe_def tests ────────────────────────────────────────────


def test_build_universe_def_prefers_structured_def():
    """Structured universe_def takes precedence over legacy field."""
    structured = UniverseDefinition(type=UniverseType.EXCHANGE, exchange=Exchange.NASDAQ)
    request = ScanCreateRequest(
        universe="all",  # should be ignored
        universe_def=structured,
    )
    result = _build_universe_def(request)
    assert result.type == UniverseType.EXCHANGE
    assert result.exchange == Exchange.NASDAQ


def test_build_universe_def_parses_legacy_all():
    request = ScanCreateRequest(universe="all")
    result = _build_universe_def(request)
    assert result.type == UniverseType.ALL
    assert result.key() == "all"
    assert result.label() == "All Stocks"


def test_build_universe_def_parses_legacy_exchange():
    request = ScanCreateRequest(universe="nyse")
    result = _build_universe_def(request)
    assert result.type == UniverseType.EXCHANGE
    assert result.exchange == Exchange.NYSE
    assert result.key() == "exchange:NYSE"


def test_build_universe_def_parses_legacy_index():
    request = ScanCreateRequest(universe="sp500")
    result = _build_universe_def(request)
    assert result.type == UniverseType.INDEX
    assert result.index == IndexName.SP500
    assert result.label() == "S&P 500"


def test_build_universe_def_parses_legacy_custom():
    request = ScanCreateRequest(
        universe="custom",
        symbols=["AAPL", "MSFT", "TSLA"],
    )
    result = _build_universe_def(request)
    assert result.type == UniverseType.CUSTOM
    assert result.symbols == ["AAPL", "MSFT", "TSLA"]
    assert result.key().startswith("custom:")


def test_build_universe_def_raises_on_invalid_legacy():
    from fastapi import HTTPException

    request = ScanCreateRequest(universe="invalid_value")
    with pytest.raises(HTTPException) as exc_info:
        _build_universe_def(request)
    assert exc_info.value.status_code == 400


# ── Command building tests ───────────────────────────────────────────────


def test_command_preserves_screener_config():
    """Verify the router can build a correct command from request fields."""
    request = ScanCreateRequest(
        universe="custom",
        symbols=["AAPL"],
        screeners=["minervini", "canslim"],
        composite_method="maximum",
        criteria={"min_rs": 80},
        idempotency_key="abc-123",
    )
    universe_def = _build_universe_def(request)
    cmd = CreateScanCommand(
        universe_def=universe_def,
        universe_label=universe_def.label(),
        universe_key=universe_def.key(),
        universe_type=universe_def.type.value,
        universe_exchange=universe_def.exchange.value if universe_def.exchange else None,
        universe_index=universe_def.index.value if universe_def.index else None,
        universe_symbols=universe_def.symbols,
        screeners=request.screeners,
        composite_method=request.composite_method,
        criteria=request.criteria,
        idempotency_key=request.idempotency_key,
    )
    assert cmd.screeners == ["minervini", "canslim"]
    assert cmd.composite_method == "maximum"
    assert cmd.criteria == {"min_rs": 80}
    assert cmd.idempotency_key == "abc-123"
    assert cmd.universe_type == "custom"
    assert cmd.universe_symbols == ["AAPL"]


def test_command_universe_metadata_for_exchange():
    """Exchange universe populates all metadata fields correctly."""
    request = ScanCreateRequest(universe="nasdaq")
    universe_def = _build_universe_def(request)
    cmd = CreateScanCommand(
        universe_def=universe_def,
        universe_label=universe_def.label(),
        universe_key=universe_def.key(),
        universe_type=universe_def.type.value,
        universe_exchange=universe_def.exchange.value if universe_def.exchange else None,
        universe_index=universe_def.index.value if universe_def.index else None,
        universe_symbols=universe_def.symbols,
    )
    assert cmd.universe_label == "NASDAQ"
    assert cmd.universe_key == "exchange:NASDAQ"
    assert cmd.universe_type == "exchange"
    assert cmd.universe_exchange == "NASDAQ"
    assert cmd.universe_index is None
