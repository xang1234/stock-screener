from __future__ import annotations

import pytest

from app.services.sg_universe_ingestion_adapter import sg_universe_ingestion_adapter


def test_sg_adapter_canonicalizes_sti_constituents_with_metadata():
    result = sg_universe_ingestion_adapter.canonicalize_rows(
        [
            {
                "symbol": "D05",
                "name": "DBS Group",
                "exchange": "SGX",
                "sector": "Banks",
                "industry": "Banks",
                "market_cap": "95000000000",
            },
            {
                "symbol": "O39.SI",
                "name": "OCBC Bank",
                "exchange": "XSES",
                "sector": "Banks",
                "industry": "Banks",
            },
            {
                "symbol": "A17U",
                "name": "CapitaLand Ascendas REIT",
                "exchange": "SES",
                "sector": "Real Estate",
            },
        ],
        source_name="sgx_official",
        snapshot_id="sgx-2026-05-15",
        snapshot_as_of="2026-05-15",
        source_metadata={"row_counts": {"xses": 3}},
    )

    assert result.rejected_rows == ()
    by_symbol = {row.symbol: row for row in result.canonical_rows}
    assert set(by_symbol) == {"D05.SI", "O39.SI", "A17U.SI"}
    assert by_symbol["D05.SI"].exchange == "XSES"
    assert by_symbol["O39.SI"].exchange == "XSES"  # alias normalized
    assert by_symbol["A17U.SI"].exchange == "XSES"  # SES alias normalized
    assert by_symbol["D05.SI"].currency == "SGD"
    assert by_symbol["D05.SI"].timezone == "Asia/Singapore"
    assert by_symbol["D05.SI"].local_code == "D05"
    assert by_symbol["A17U.SI"].local_code == "A17U"
    assert by_symbol["D05.SI"].market_cap == pytest.approx(95_000_000_000.0)
    assert by_symbol["D05.SI"].source_metadata["row_counts"] == {"xses": 3}


def test_sg_adapter_suffix_strips_exchange_prefix_in_symbol():
    result = sg_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "SGX:D05", "name": "DBS", "exchange": ""},
            {"symbol": "SES:O39", "name": "OCBC", "exchange": ""},
            {"symbol": "XSES:U11", "name": "UOB", "exchange": ""},
        ],
        source_name="sgx_official",
        snapshot_id="sgx-2026-05-15",
    )

    assert result.rejected_rows == ()
    canonical_symbols = {row.symbol for row in result.canonical_rows}
    assert canonical_symbols == {"D05.SI", "O39.SI", "U11.SI"}


def test_sg_adapter_rejects_invalid_exchange_and_symbol():
    result = sg_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "D05", "name": "Bad Exchange", "exchange": "NYSE"},
            {"symbol": "BAD!@#", "name": "Bad Symbol", "exchange": "XSES"},
            {"symbol": "", "name": "Missing Symbol", "exchange": "XSES"},
            {"symbol": "TOOLONGSYMBOL", "name": "Too long", "exchange": "XSES"},
        ],
        source_name="sgx_official",
        snapshot_id="sgx-2026-05-15",
    )

    assert result.canonical_rows == ()
    reasons = [row.reason for row in result.rejected_rows]
    assert any("Unsupported SG exchange" in reason for reason in reasons)
    assert any("Invalid SG symbol" in reason for reason in reasons)
    assert any("Missing symbol" in reason for reason in reasons)


def test_sg_adapter_deduplicates_deterministically():
    """Bare-symbol row arrives first; merge keeps its identity but backfills the name."""
    result = sg_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "D05", "name": "", "exchange": "XSES"},
            {"symbol": "D05.SI", "name": "DBS Group", "exchange": "XSES"},
        ],
        source_name="sgx_official",
        snapshot_id="sgx-2026-05-15",
    )

    assert result.rejected_rows == ()
    assert len(result.canonical_rows) == 1
    canonical = result.canonical_rows[0]
    assert canonical.symbol == "D05.SI"
    assert canonical.name == "DBS Group"


def test_sg_adapter_rejects_unapproved_source():
    with pytest.raises(ValueError, match="Unapproved SG source"):
        sg_universe_ingestion_adapter.canonicalize_rows(
            [{"symbol": "D05.SI", "name": "DBS"}],
            source_name="random_third_party",
            snapshot_id="sgx-2026-05-15",
        )


def test_sg_adapter_accepts_reit_style_alphanumeric_codes():
    """SGX REITs carry trailing-letter codes (A17U, ME8U, BUOU, T82U, C38U)."""
    result = sg_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "A17U.SI", "name": "CapitaLand Ascendas REIT"},
            {"symbol": "ME8U.SI", "name": "Mapletree Industrial Trust"},
            {"symbol": "BUOU.SI", "name": "Frasers Logistics & Commercial Trust"},
            {"symbol": "T82U.SI", "name": "Suntec REIT"},
            {"symbol": "C38U.SI", "name": "CapitaLand Integrated Commercial Trust"},
        ],
        source_name="sgx_official",
        snapshot_id="sgx-2026-05-15",
    )

    assert result.rejected_rows == ()
    assert {row.symbol for row in result.canonical_rows} == {
        "A17U.SI",
        "ME8U.SI",
        "BUOU.SI",
        "T82U.SI",
        "C38U.SI",
    }


def test_sg_adapter_market_cap_parses_suffixed_strings():
    result = sg_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "D05.SI", "name": "DBS", "market_cap": "95B"},
            {"symbol": "O39.SI", "name": "OCBC", "market_cap": "70.5B"},
            {"symbol": "U11.SI", "name": "UOB", "market_cap": "1,234,567,890"},
            {"symbol": "C6L.SI", "name": "SIA", "market_cap": None},
        ],
        source_name="sgx_official",
        snapshot_id="sgx-2026-05-15",
    )

    by_symbol = {row.symbol: row for row in result.canonical_rows}
    assert by_symbol["D05.SI"].market_cap == pytest.approx(9.5e10)
    assert by_symbol["O39.SI"].market_cap == pytest.approx(7.05e10)
    assert by_symbol["U11.SI"].market_cap == pytest.approx(1_234_567_890.0)
    assert by_symbol["C6L.SI"].market_cap is None
