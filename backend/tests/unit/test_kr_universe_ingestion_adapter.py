from __future__ import annotations

import pytest

from app.services.kr_universe_ingestion_adapter import kr_universe_ingestion_adapter


def test_kr_adapter_canonicalizes_kospi_and_kosdaq_rows_with_metadata():
    result = kr_universe_ingestion_adapter.canonicalize_rows(
        [
            {
                "symbol": "005930",
                "name": "Samsung Electronics",
                "exchange": "KOSPI",
                "sector": "Information Technology",
                "industry": "Semiconductors",
                "market_cap": "530000000000000",
            },
            {
                "symbol": "091990",
                "name": "Celltrion Healthcare",
                "exchange": "KOSDAQ",
                "sector": "Health Care",
                "industry": "Biotechnology",
            },
        ],
        source_name="krx_official",
        snapshot_id="krx-2026-04-29",
        snapshot_as_of="2026-04-29",
        source_metadata={"row_counts": {"kospi": 1, "kosdaq": 1}},
    )

    assert result.rejected_rows == ()
    assert [row.symbol for row in result.canonical_rows] == ["005930.KS", "091990.KQ"]
    assert [row.exchange for row in result.canonical_rows] == ["KOSPI", "KOSDAQ"]
    assert result.canonical_rows[0].currency == "KRW"
    assert result.canonical_rows[0].timezone == "Asia/Seoul"
    assert result.canonical_rows[0].local_code == "005930"
    assert result.canonical_rows[0].source_metadata["row_counts"] == {"kospi": 1, "kosdaq": 1}


def test_kr_adapter_rejects_konex_and_non_operating_products():
    result = kr_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "123456", "name": "Konex Co", "exchange": "KONEX"},
            {"symbol": "305720", "name": "KODEX ETF", "exchange": "KOSPI", "security_type": "ETF"},
            {"symbol": "ABC", "name": "Bad Code", "exchange": "KOSPI"},
        ],
        source_name="krx_official",
        snapshot_id="krx-2026-04-29",
    )

    assert result.canonical_rows == ()
    reasons = [row.reason for row in result.rejected_rows]
    assert any("Unsupported KR exchange" in reason for reason in reasons)
    assert any("non-operating product" in reason for reason in reasons)
    assert any("Invalid KR symbol" in reason for reason in reasons)


def test_kr_adapter_deduplicates_deterministically():
    result = kr_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "005930", "name": "", "exchange": "KOSPI"},
            {"symbol": "005930.KS", "name": "Samsung Electronics", "exchange": "KOSPI"},
        ],
        source_name="krx_official",
        snapshot_id="krx-2026-04-29",
    )

    assert len(result.canonical_rows) == 1
    assert result.canonical_rows[0].symbol == "005930.KS"
    assert result.canonical_rows[0].name == "Samsung Electronics"


def test_kr_adapter_requires_approved_sources():
    with pytest.raises(ValueError, match="Unapproved KR source"):
        kr_universe_ingestion_adapter.canonicalize_rows(
            [{"symbol": "005930", "name": "Samsung Electronics", "exchange": "KOSPI"}],
            source_name="random_blog",
            snapshot_id="snapshot",
        )
