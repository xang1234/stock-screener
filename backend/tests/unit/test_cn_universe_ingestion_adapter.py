from __future__ import annotations

import pytest

from app.services.cn_universe_ingestion_adapter import cn_universe_ingestion_adapter


def test_cn_adapter_canonicalizes_sse_szse_and_bse_rows_with_metadata():
    result = cn_universe_ingestion_adapter.canonicalize_rows(
        [
            {
                "symbol": "600519",
                "name": "Kweichow Moutai",
                "exchange": "SSE",
                "industry": "Beverage Manufacturing",
                "market_cap": "2500000000000",
            },
            {
                "symbol": "000001",
                "name": "Ping An Bank",
                "exchange": "SZSE",
                "industry": "Banking",
            },
            {
                "symbol": "920118",
                "name": "Taihu Snow",
                "exchange": "BSE",
                "industry": "Textile Manufacturing",
            },
        ],
        source_name="cn_akshare_eastmoney",
        snapshot_id="cn-a-share-2026-04-30",
        snapshot_as_of="2026-04-30",
        source_metadata={"source_count": 3},
    )

    assert result.rejected_rows == ()
    assert [row.symbol for row in result.canonical_rows] == [
        "000001.SZ",
        "600519.SS",
        "920118.BJ",
    ]
    rows_by_symbol = {row.symbol: row for row in result.canonical_rows}
    assert rows_by_symbol["600519.SS"].exchange == "SSE"
    assert rows_by_symbol["600519.SS"].board == "SSE_MAIN"
    assert rows_by_symbol["000001.SZ"].exchange == "SZSE"
    assert rows_by_symbol["000001.SZ"].board == "SZSE_MAIN"
    assert rows_by_symbol["920118.BJ"].exchange == "BSE"
    assert rows_by_symbol["920118.BJ"].board == "BSE"
    assert rows_by_symbol["600519.SS"].currency == "CNY"
    assert rows_by_symbol["600519.SS"].timezone == "Asia/Shanghai"
    assert rows_by_symbol["000001.SZ"].sector == "Financials"
    assert rows_by_symbol["600519.SS"].source_metadata["source_count"] == 3


def test_cn_adapter_infers_exchange_and_board_from_suffix_or_code_prefix():
    result = cn_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "688981.SS", "name": "SMIC"},
            {"symbol": "300750.SZ", "name": "CATL"},
            {"symbol": "920118.BJ", "name": "Taihu Snow"},
        ],
        source_name="cn_akshare_eastmoney",
        snapshot_id="cn-a-share-2026-04-30",
    )

    assert result.rejected_rows == ()
    rows_by_symbol = {row.symbol: row for row in result.canonical_rows}
    assert rows_by_symbol["688981.SS"].board == "SSE_STAR"
    assert rows_by_symbol["300750.SZ"].board == "SZSE_CHINEXT"
    assert rows_by_symbol["920118.BJ"].exchange == "BSE"


def test_cn_adapter_rejects_non_a_share_products_and_invalid_codes():
    result = cn_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "900901", "name": "B Share", "exchange": "SSE"},
            {"symbol": "510300", "name": "CSI 300 ETF", "exchange": "SSE", "security_type": "ETF"},
            {"symbol": "113000", "name": "Convertible Bond", "exchange": "SSE", "security_type": "Bond"},
            {"symbol": "ABC", "name": "Bad Code", "exchange": "SSE"},
            {"symbol": "600519", "name": "Bad Exchange", "exchange": "HKEX"},
        ],
        source_name="cn_akshare_eastmoney",
        snapshot_id="cn-a-share-2026-04-30",
    )

    assert result.canonical_rows == ()
    reasons = [row.reason for row in result.rejected_rows]
    assert any("non-A-share or non-operating product" in reason for reason in reasons)
    assert any("Invalid CN symbol" in reason for reason in reasons)
    assert any("Unsupported CN exchange" in reason for reason in reasons)


def test_cn_adapter_deduplicates_deterministically_and_prefers_complete_rows():
    result = cn_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "600519", "name": "", "exchange": "SSE"},
            {
                "symbol": "600519.SS",
                "name": "Kweichow Moutai",
                "exchange": "SSE",
                "sector": "Consumer Staples",
                "industry": "Beverage Manufacturing",
                "market_cap": "2500000000000",
            },
        ],
        source_name="cn_akshare_eastmoney",
        snapshot_id="cn-a-share-2026-04-30",
    )

    assert len(result.canonical_rows) == 1
    row = result.canonical_rows[0]
    assert row.symbol == "600519.SS"
    assert row.name == "Kweichow Moutai"
    assert row.sector == "Consumer Staples"
    assert row.industry == "Beverage Manufacturing"
    assert row.market_cap == 2_500_000_000_000.0


def test_cn_adapter_infers_sector_from_chinese_industry_when_source_sector_missing():
    result = cn_universe_ingestion_adapter.canonicalize_rows(
        [
            {"symbol": "600519", "name": "贵州茅台", "exchange": "SSE", "industry": "酿酒行业"},
            {"symbol": "300750", "name": "宁德时代", "exchange": "SZSE", "industry": "电气设备"},
        ],
        source_name="cn_akshare_eastmoney",
        snapshot_id="cn-a-share-2026-04-30",
    )

    rows_by_symbol = {row.symbol: row for row in result.canonical_rows}
    assert rows_by_symbol["600519.SS"].sector == "Consumer Staples"
    assert rows_by_symbol["300750.SZ"].sector == "Industrials"


def test_cn_adapter_requires_approved_sources():
    with pytest.raises(ValueError, match="Unapproved CN source"):
        cn_universe_ingestion_adapter.canonicalize_rows(
            [{"symbol": "600519", "name": "Kweichow Moutai", "exchange": "SSE"}],
            source_name="random_blog",
            snapshot_id="snapshot",
        )
