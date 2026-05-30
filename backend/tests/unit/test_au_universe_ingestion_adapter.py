from __future__ import annotations

import pytest

from app.services.au_universe_ingestion_adapter import au_universe_ingestion_adapter


def test_au_adapter_canonicalizes_asx_rows_with_metadata():
    result = au_universe_ingestion_adapter.canonicalize_rows(
        [
            {
                "symbol": "BHP",
                "name": "BHP Group",
                "exchange": "ASX",
                "sector": "Basic Materials",
                "industry": "Materials",
                "market_cap": "95000000000",
            },
            {
                "symbol": "CBA.AX",
                "name": "Commonwealth Bank",
                "exchange": "XASX",
                "sector": "Financial Services",
                "industry": "Banks",
            },
        ],
        source_name="asx_official_public_csv",
        snapshot_id="asx-2026-05-15",
        snapshot_as_of="2026-05-15",
        source_metadata={"row_counts": {"xasx": 2}},
    )

    assert result.rejected_rows == ()
    by_symbol = {row.symbol: row for row in result.canonical_rows}
    assert set(by_symbol) == {"BHP.AX", "CBA.AX"}
    assert by_symbol["BHP.AX"].market == "AU"
    assert by_symbol["BHP.AX"].mic == "XASX"
    assert by_symbol["BHP.AX"].currency == "AUD"
    assert by_symbol["BHP.AX"].timezone == "Australia/Sydney"
    assert by_symbol["BHP.AX"].local_code == "BHP"
    assert by_symbol["BHP.AX"].industry == "Materials"
    assert by_symbol["BHP.AX"].market_cap == pytest.approx(95_000_000_000.0)
    assert by_symbol["BHP.AX"].provenance.source_metadata["row_counts"] == {
        "xasx": 2
    }
    assert by_symbol["CBA.AX"].symbol == "CBA.AX"
    assert by_symbol["CBA.AX"].mic == "XASX"


def test_au_adapter_accepts_exchange_prefixed_symbols():
    result = au_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "ASX:BHP", "name": "BHP", "exchange": ""},
            {"symbol": "XASX:CBA", "name": "Commonwealth Bank", "exchange": ""},
        ],
        source_name="asx_official",
        snapshot_id="asx-2026-05-15",
    )

    assert result.rejected_rows == ()
    canonical_symbols = {row.symbol for row in result.canonical_rows}
    assert canonical_symbols == {"BHP.AX", "CBA.AX"}


def test_au_adapter_accepts_main_board_listing_tier():
    result = au_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "BHP", "name": "BHP", "exchange": "ASX", "board": "Main"},
            {
                "symbol": "CBA",
                "name": "Commonwealth Bank",
                "exchange": "ASX",
                "listing_tier": "Main",
            },
        ],
        source_name="asx_official",
        snapshot_id="asx-2026-05-15",
    )

    assert result.rejected_rows == ()
    by_symbol = {row.symbol: row for row in result.canonical_rows}
    assert by_symbol["BHP.AX"].listing_tier == "main"
    assert by_symbol["CBA.AX"].listing_tier == "main"


def test_au_adapter_rejects_invalid_exchange_symbol_missing_and_too_long():
    result = au_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "BHP", "name": "Bad Exchange", "exchange": "NYSE"},
            {"symbol": "BAD!@#", "name": "Bad Symbol", "exchange": "XASX"},
            {"symbol": "", "name": "Missing Symbol", "exchange": "XASX"},
            {"symbol": "TOOLONG", "name": "Too long", "exchange": "XASX"},
        ],
        source_name="asx_official",
        snapshot_id="asx-2026-05-15",
    )

    assert result.canonical_rows == ()
    reasons = [row.reason for row in result.rejected_rows]
    assert any("Unsupported AU exchange" in reason for reason in reasons)
    assert any("Invalid AU symbol" in reason for reason in reasons)
    assert any("Missing symbol" in reason for reason in reasons)


def test_au_adapter_deduplicates_deterministically_and_backfills_name():
    result = au_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "BHP", "name": "", "exchange": "XASX"},
            {"symbol": "BHP.AX", "name": "BHP Group", "exchange": "XASX"},
        ],
        source_name="asx_official",
        snapshot_id="asx-2026-05-15",
    )

    assert result.rejected_rows == ()
    assert len(result.canonical_rows) == 1
    canonical = result.canonical_rows[0]
    assert canonical.symbol == "BHP.AX"
    assert canonical.name == "BHP Group"


def test_au_adapter_rejects_unapproved_source():
    with pytest.raises(ValueError, match="Unapproved AU source"):
        au_universe_ingestion_adapter.canonicalize_rows(
            [{"symbol": "BHP.AX", "name": "BHP"}],
            source_name="random_third_party",
            snapshot_id="asx-2026-05-15",
        )


def test_au_adapter_market_cap_parses_suffixed_strings():
    result = au_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "BHP.AX", "name": "BHP", "market_cap": "95B"},
            {"symbol": "CBA.AX", "name": "Commonwealth Bank", "market_cap": "70.5B"},
            {"symbol": "WES.AX", "name": "Wesfarmers", "market_cap": "1,234,567,890"},
            {"symbol": "WOW.AX", "name": "Woolworths", "market_cap": None},
        ],
        source_name="asx_official",
        snapshot_id="asx-2026-05-15",
    )

    by_symbol = {row.symbol: row for row in result.canonical_rows}
    assert by_symbol["BHP.AX"].market_cap == pytest.approx(9.5e10)
    assert by_symbol["CBA.AX"].market_cap == pytest.approx(7.05e10)
    assert by_symbol["WES.AX"].market_cap == pytest.approx(1_234_567_890.0)
    assert by_symbol["WOW.AX"].market_cap is None
