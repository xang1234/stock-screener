from __future__ import annotations

import pytest

from app.services.de_universe_ingestion_adapter import de_universe_ingestion_adapter


def test_de_adapter_canonicalizes_xetra_and_frankfurt_rows_with_metadata():
    result = de_universe_ingestion_adapter.canonicalize_rows(
        [
            {
                "symbol": "SAP",
                "name": "SAP",
                "exchange": "XETR",
                "sector": "Information Technology",
                "industry": "Software",
                "market_cap": "180000000000",
            },
            {
                "symbol": "1COV",
                "name": "Covestro",
                "exchange": "XETRA",
                "sector": "Materials",
                "industry": "Chemicals",
            },
            {
                "symbol": "ALV",
                "name": "Allianz Frankfurt floor",
                "exchange": "XFRA",
            },
        ],
        source_name="dbg_official",
        snapshot_id="dbg-2026-05-09",
        snapshot_as_of="2026-05-09",
        source_metadata={"row_counts": {"xetr": 2, "xfra": 1}},
    )

    assert result.rejected_rows == ()
    by_symbol = {row.symbol: row for row in result.canonical_rows}
    assert set(by_symbol) == {"SAP.DE", "1COV.DE", "ALV.F"}
    assert by_symbol["SAP.DE"].exchange == "XETR"
    assert by_symbol["1COV.DE"].exchange == "XETR"  # XETRA alias normalized to XETR
    assert by_symbol["ALV.F"].exchange == "XFRA"
    assert by_symbol["SAP.DE"].currency == "EUR"
    assert by_symbol["SAP.DE"].timezone == "Europe/Berlin"
    assert by_symbol["SAP.DE"].local_code == "SAP"
    assert by_symbol["SAP.DE"].source_metadata["row_counts"] == {"xetr": 2, "xfra": 1}


def test_de_adapter_suffix_overrides_disagreeing_exchange_column():
    # When the symbol carries an explicit suffix, it wins over the row's
    # exchange field (Xetra-primary policy). The disagreeing exchange must
    # still be a recognized DE alias to pass validation.
    result = de_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "SAP.DE", "name": "SAP suffix-wins", "exchange": "XFRA"},
            {"symbol": "ALV.F", "name": "Allianz suffix-wins", "exchange": "XETR"},
        ],
        source_name="dbg_official",
        snapshot_id="dbg-2026-05-09",
    )

    assert result.rejected_rows == ()
    by_symbol = {row.symbol: row for row in result.canonical_rows}
    assert by_symbol["SAP.DE"].exchange == "XETR"
    assert by_symbol["ALV.F"].exchange == "XFRA"


def test_de_adapter_strips_exchange_prefix_in_symbol():
    result = de_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "XETR:SAP", "name": "SAP", "exchange": ""},
            {"symbol": "XFRA:ALV", "name": "Allianz", "exchange": ""},
            {"symbol": "FWB:BMW", "name": "BMW", "exchange": ""},
        ],
        source_name="dbg_official",
        snapshot_id="dbg-2026-05-09",
    )

    assert result.rejected_rows == ()
    by_symbol = {row.symbol: row for row in result.canonical_rows}
    assert by_symbol["SAP.DE"].exchange == "XETR"
    assert by_symbol["ALV.F"].exchange == "XFRA"
    # FWB prefix maps to XFRA (Frankfurt floor) — suffix becomes .F
    assert by_symbol["BMW.F"].exchange == "XFRA"


def test_de_adapter_rejects_invalid_exchange_and_symbol():
    result = de_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "SAP", "name": "Bad Exchange", "exchange": "NYSE"},
            {"symbol": "BAD!@#", "name": "Bad Symbol", "exchange": "XETR"},
            {"symbol": "", "name": "Missing Symbol", "exchange": "XETR"},
        ],
        source_name="dbg_official",
        snapshot_id="dbg-2026-05-09",
    )

    assert result.canonical_rows == ()
    reasons = [row.reason for row in result.rejected_rows]
    assert any("Unsupported DE exchange" in reason for reason in reasons)
    assert any("Invalid DE symbol" in reason for reason in reasons)
    assert any("Missing symbol" in reason for reason in reasons)


def test_de_adapter_deduplicates_deterministically():
    # The bare-symbol row arrives first but the .DE-suffixed row carries
    # the canonical name; merge must prefer the earlier row's identity but
    # backfill the missing name from the later one.
    result = de_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "SAP", "name": "", "exchange": "XETR"},
            {"symbol": "SAP.DE", "name": "SAP", "exchange": "XETR"},
        ],
        source_name="dbg_official",
        snapshot_id="dbg-2026-05-09",
    )

    assert result.rejected_rows == ()
    assert len(result.canonical_rows) == 1
    assert result.canonical_rows[0].symbol == "SAP.DE"
    assert result.canonical_rows[0].name == "SAP"


def test_de_adapter_keeps_xetra_and_frankfurt_listings_distinct():
    # Same root on Xetra and Frankfurt should canonicalize to different
    # symbols (.DE vs .F) and both must survive.
    result = de_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "ALV", "name": "Allianz Xetra", "exchange": "XETR"},
            {"symbol": "ALV", "name": "Allianz Frankfurt", "exchange": "XFRA"},
        ],
        source_name="dbg_official",
        snapshot_id="dbg-2026-05-09",
    )

    assert result.rejected_rows == ()
    by_symbol = {row.symbol: row for row in result.canonical_rows}
    assert set(by_symbol) == {"ALV.DE", "ALV.F"}
    assert by_symbol["ALV.DE"].exchange == "XETR"
    assert by_symbol["ALV.F"].exchange == "XFRA"


def test_de_adapter_rejects_unapproved_source():
    with pytest.raises(ValueError, match="Unapproved DE source"):
        de_universe_ingestion_adapter.canonicalize_rows(
            [{"symbol": "SAP", "name": "SAP", "exchange": "XETR"}],
            source_name="random_scraper",
            snapshot_id="dbg-2026-05-09",
        )


def test_de_adapter_canonicalize_is_deterministic_via_row_hash():
    # Two runs over the same input must produce identical row_hashes,
    # establishing the determinism guarantee the snapshot pipeline relies on.
    rows = [{"symbol": "SAP.DE", "name": "SAP", "exchange": "XETR"}]
    first = de_universe_ingestion_adapter.canonicalize_rows(
        rows, source_name="dbg_official", snapshot_id="dbg-fixed",
    )
    second = de_universe_ingestion_adapter.canonicalize_rows(
        rows, source_name="dbg_official", snapshot_id="dbg-fixed",
    )

    assert [r.row_hash for r in first.canonical_rows] == [
        r.row_hash for r in second.canonical_rows
    ]
    assert [r.lineage_hash for r in first.canonical_rows] == [
        r.lineage_hash for r in second.canonical_rows
    ]


@pytest.mark.parametrize(
    "raw_symbol,expected_local_code,expected_suffix",
    [
        ("SAP", "SAP", ".DE"),          # 3-char root, no suffix → defaults to Xetra
        ("BMW", "BMW", ".DE"),          # 3-char root
        ("1COV", "1COV", ".DE"),        # 4-char numeric prefix (Covestro)
        ("MUV2", "MUV2", ".DE"),        # 4-char with trailing digit (Munich Re)
        ("HEN3", "HEN3", ".DE"),        # preferred-share suffix digit
        ("EXS1.DE", "EXS1", ".DE"),     # ETF ticker
        ("SAP.F", "SAP", ".F"),         # explicit Frankfurt suffix
    ],
)
def test_de_adapter_accepts_local_code_shapes(raw_symbol, expected_local_code, expected_suffix):
    result = de_universe_ingestion_adapter.canonicalize_rows(
        [{"symbol": raw_symbol, "name": "Test", "exchange": "XETR" if expected_suffix == ".DE" else "XFRA"}],
        source_name="dbg_official",
        snapshot_id="dbg-2026-05-09",
    )

    assert result.rejected_rows == ()
    assert result.canonical_rows[0].local_code == expected_local_code
    assert result.canonical_rows[0].symbol == f"{expected_local_code}{expected_suffix}"


@pytest.mark.parametrize(
    "raw_symbol",
    [
        "ABCDEFGHI",  # 9-char exceeds 8-char cap
        "SAP-PR",     # dashes not in regex
        "SAP_PR",     # underscores not in regex
        "SAP@",       # punctuation other than dot rejected
    ],
)
def test_de_adapter_rejects_out_of_range_symbols(raw_symbol):
    result = de_universe_ingestion_adapter.canonicalize_rows(
        [{"symbol": raw_symbol, "name": "Bad", "exchange": "XETR"}],
        source_name="dbg_official",
        snapshot_id="dbg-2026-05-09",
    )

    assert result.canonical_rows == ()
    assert any("Invalid DE symbol" in row.reason for row in result.rejected_rows)
