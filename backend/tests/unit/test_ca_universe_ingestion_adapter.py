from __future__ import annotations

import pytest

from app.services.ca_universe_ingestion_adapter import ca_universe_ingestion_adapter


def test_ca_adapter_canonicalizes_tsx_and_tsxv_rows_with_metadata():
    result = ca_universe_ingestion_adapter.canonicalize_rows(
        [
            {
                "symbol": "RY",
                "name": "Royal Bank of Canada",
                "exchange": "TSX",
                "sector": "Financials",
                "industry": "Banks",
                "market_cap": "210000000000",
            },
            {
                "symbol": "NVA",
                "name": "Nova Mining",
                "exchange": "TSXV",
                "sector": "Materials",
                "industry": "Metals & Mining",
            },
        ],
        source_name="tmx_official",
        snapshot_id="tmx-2026-05-09",
        snapshot_as_of="2026-05-09",
        source_metadata={"row_counts": {"tsx": 1, "tsxv": 1}},
    )

    assert result.rejected_rows == ()
    assert [row.symbol for row in result.canonical_rows] == ["NVA.V", "RY.TO"]
    assert [row.exchange for row in result.canonical_rows] == ["TSXV", "TSX"]
    assert result.canonical_rows[1].currency == "CAD"
    assert result.canonical_rows[1].timezone == "America/Toronto"
    assert result.canonical_rows[1].local_code == "RY"
    assert result.canonical_rows[1].source_metadata["row_counts"] == {"tsx": 1, "tsxv": 1}


def test_ca_adapter_normalizes_tmx_dot_segments_to_yahoo_dashes():
    result = ca_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "BIP.UN", "name": "Brookfield Infrastructure Partners", "exchange": "TSX"},
            {"symbol": "BCE.PR.K", "name": "BCE Pref K", "exchange": "TSX"},
            {"symbol": "RCI.B", "name": "Rogers Class B", "exchange": "TSX"},
        ],
        source_name="tmx_official",
        snapshot_id="tmx-2026-05-09",
    )

    assert result.rejected_rows == ()
    canonical_symbols = {row.symbol for row in result.canonical_rows}
    assert canonical_symbols == {"BIP-UN.TO", "BCE-PR-K.TO", "RCI-B.TO"}


def test_ca_adapter_infers_exchange_from_yahoo_suffix():
    result = ca_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "XYZ.TO", "name": "Senior Inferred", "exchange": ""},
            {"symbol": "ABC.V", "name": "Venture Inferred", "exchange": ""},
        ],
        source_name="tmx_official",
        snapshot_id="tmx-2026-05-09",
    )

    assert result.rejected_rows == ()
    by_symbol = {row.symbol: row for row in result.canonical_rows}
    assert by_symbol["XYZ.TO"].exchange == "TSX"
    assert by_symbol["ABC.V"].exchange == "TSXV"


def test_ca_adapter_rejects_invalid_exchange_and_symbol():
    result = ca_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "XYZ", "name": "Bad Exchange", "exchange": "NYSE"},
            {"symbol": "@@@", "name": "Bad Symbol", "exchange": "TSX"},
            {"symbol": "", "name": "Missing Symbol", "exchange": "TSX"},
        ],
        source_name="tmx_official",
        snapshot_id="tmx-2026-05-09",
    )

    assert result.canonical_rows == ()
    reasons = [row.reason for row in result.rejected_rows]
    assert any("Unsupported CA exchange" in reason for reason in reasons)
    assert any("Invalid CA symbol" in reason for reason in reasons)
    assert any("Missing symbol" in reason for reason in reasons)


def test_ca_adapter_deduplicates_deterministically():
    result = ca_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "RY", "name": "", "exchange": "TSX"},
            {"symbol": "RY.TO", "name": "Royal Bank of Canada", "exchange": "TSX"},
        ],
        source_name="tmx_official",
        snapshot_id="tmx-2026-05-09",
    )

    assert len(result.canonical_rows) == 1
    assert result.canonical_rows[0].symbol == "RY.TO"
    assert result.canonical_rows[0].name == "Royal Bank of Canada"


def test_ca_adapter_keeps_same_root_on_different_exchanges_distinct():
    # Same root on TSX and TSXV should canonicalize to *different* symbols
    # (.TO vs .V) and therefore both rows must survive — no cross-exchange
    # collapse.
    result = ca_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "ABC", "name": "Senior Listing", "exchange": "TSX"},
            {"symbol": "ABC", "name": "Junior Listing", "exchange": "TSXV"},
        ],
        source_name="tmx_official",
        snapshot_id="tmx-2026-05-09",
    )

    assert result.rejected_rows == ()
    assert len(result.canonical_rows) == 2
    by_symbol = {row.symbol: row for row in result.canonical_rows}
    assert set(by_symbol) == {"ABC.TO", "ABC.V"}
    assert by_symbol["ABC.TO"].exchange == "TSX"
    assert by_symbol["ABC.V"].exchange == "TSXV"


def test_ca_adapter_rejects_unapproved_source():
    with pytest.raises(ValueError, match="Unapproved CA source"):
        ca_universe_ingestion_adapter.canonicalize_rows(
            [{"symbol": "RY", "name": "Royal Bank", "exchange": "TSX"}],
            source_name="random_scraper",
            snapshot_id="tmx-2026-05-09",
        )


@pytest.mark.parametrize(
    "raw_symbol,expected_canonical,expected_exchange",
    [
        ("TSX:RY", "RY.TO", "TSX"),
        ("XTSE:SHOP", "SHOP.TO", "TSX"),
        ("TSXV:NVA", "NVA.V", "TSXV"),
        ("XTNX:ABC", "ABC.V", "TSXV"),
    ],
)
def test_ca_adapter_strips_exchange_prefix(raw_symbol, expected_canonical, expected_exchange):
    result = ca_universe_ingestion_adapter.canonicalize_rows(
        [{"symbol": raw_symbol, "name": "Test", "exchange": ""}],
        source_name="tmx_official",
        snapshot_id="tmx-2026-05-09",
    )

    assert result.rejected_rows == ()
    assert result.canonical_rows[0].symbol == expected_canonical
    assert result.canonical_rows[0].exchange == expected_exchange


@pytest.mark.parametrize(
    "raw_symbol,expected_local_code",
    [
        ("X", "X"),                # single-char ticker
        ("ABCDEF", "ABCDEF"),      # 6-char root
        ("BCE.PR.K22", "BCE-PR-K22"),  # preferred series with digits
        ("AQN.PR.A", "AQN-PR-A"),  # preferred class
    ],
)
def test_ca_adapter_accepts_widened_regex_cases(raw_symbol, expected_local_code):
    result = ca_universe_ingestion_adapter.canonicalize_rows(
        [{"symbol": raw_symbol, "name": "Edge case", "exchange": "TSX"}],
        source_name="tmx_official",
        snapshot_id="tmx-2026-05-09",
    )

    assert result.rejected_rows == ()
    assert result.canonical_rows[0].local_code == expected_local_code


@pytest.mark.parametrize(
    "raw_symbol",
    [
        "ABCDEFG",         # 7-char root exceeds limit
        "RY-",             # trailing dash
        "RY-TOOOLONG",     # dash segment > 4 chars
        "1RY",             # numeric leading char
    ],
)
def test_ca_adapter_rejects_out_of_range_symbols(raw_symbol):
    result = ca_universe_ingestion_adapter.canonicalize_rows(
        [{"symbol": raw_symbol, "name": "Bad", "exchange": "TSX"}],
        source_name="tmx_official",
        snapshot_id="tmx-2026-05-09",
    )

    assert result.canonical_rows == ()
    assert any("Invalid CA symbol" in row.reason for row in result.rejected_rows)
