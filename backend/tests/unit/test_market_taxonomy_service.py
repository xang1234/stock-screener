from __future__ import annotations

import pytest

from app.services.market_taxonomy_service import MarketTaxonomyService, TaxonomyLoadError


def _write_csv(path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_market_taxonomy_service_normalizes_hk_jp_and_tw_symbols(tmp_path):
    _write_csv(tmp_path / "IBD_industry_group.csv", "AAPL,Computer-Hardware/Peripherals\n")
    _write_csv(
        tmp_path / "hk-deep.csv",
        "Symbol,EM Industry (EN),Theme (EN)\n700,Internet Services,AI Infrastructure\n",
    )
    _write_csv(
        tmp_path / "india-deep.csv",
        (
            "Symbol,Exchange,Industry (Sector),Subgroup (Theme),Sub-industry\n"
            "RELIANCE.NS,XNSE,ENERGY (STOCKS),Oil & Gas,Integrated Oil & Gas\n"
            "500325.BO,XBOM,ENERGY (STOCKS),Oil & Gas,Integrated Oil & Gas\n"
            "500002.BO,XBOM,INDUSTRIALS (STOCKS),Electrical Equipment,Industrial Electrical Equipment\n"
        ),
    )
    _write_csv(
        tmp_path / "kabutan_themes_en.csv",
        "Symbol,TSE 33-Sector,TSE 17-Sector,Theme (EN)\n7203,Transportation Equipment,Automobiles,Hybrid Vehicles\n",
    )
    _write_csv(
        tmp_path / "taiwan-deep.csv",
        "Symbol,Market,Industry (EN)\n2330,TWSE,Semiconductors\n8069,TPEX,Computer Peripherals\n",
    )
    _write_csv(
        tmp_path / "korea-deep.csv",
        (
            "Symbol,Market,Sector,Industry Group,Industry,Sub-Industry\n"
            "005930,KOSPI,Information Technology,Technology Hardware,Semiconductors,Memory Semiconductors\n"
            "091990,KOSDAQ,Health Care,Biotechnology,Biologics,Biopharmaceuticals\n"
        ),
    )
    _write_csv(
        tmp_path / "china-deep.csv",
        (
            "Symbol,Exchange,Sector,Industry Group,Industry,Sub-Industry\n"
            "600519,SSE,Consumer Staples,Food & Beverage,Beverage Manufacturing,Liquor\n"
            "000001,SZSE,Financials,Banks,Commercial Banks,Joint-stock Banks\n"
            "920118,BJSE,Consumer Discretionary,Textiles,Textile Manufacturing,Home Textiles\n"
        ),
    )

    service = MarketTaxonomyService(data_dir=tmp_path)

    hk = service.get("00700.HK", market="HK")
    assert hk is not None
    assert hk.symbol == "0700.HK"
    assert hk.industry_group == "Internet Services"
    assert hk.themes_list() == ["AI Infrastructure"]

    hk_unpadded = service.get("700", market="HK")
    assert hk_unpadded is not None
    assert hk_unpadded.symbol == "0700.HK"

    ind_nse = service.get("RELIANCE", market="IN")
    assert ind_nse is not None
    assert ind_nse.symbol == "RELIANCE.NS"
    assert ind_nse.sector == "ENERGY (STOCKS)"
    assert ind_nse.industry == "Integrated Oil & Gas"
    assert ind_nse.industry_group == "Oil & Gas"

    ind_bse_dual_listed = service.get("500325.BO", market="IN", exchange="XBOM")
    assert ind_bse_dual_listed is not None
    assert ind_bse_dual_listed.symbol == "500325.BO"
    assert ind_bse_dual_listed.industry == "Integrated Oil & Gas"
    assert ind_bse_dual_listed.industry_group == "Oil & Gas"

    ind_bse = service.get("500002", market="IN", exchange="XBOM")
    assert ind_bse is not None
    assert ind_bse.symbol == "500002.BO"
    assert ind_bse.industry_group == "Electrical Equipment"

    jp = service.get("7203.T", market="JP")
    assert jp is not None
    assert jp.symbol == "7203.T"
    assert jp.industry_group == "Transportation Equipment"
    assert jp.sector == "Automobiles"
    assert jp.themes_list() == ["Hybrid Vehicles"]

    tw = service.get("2330.TW", market="TW")
    assert tw is not None
    assert tw.symbol == "2330.TW"
    assert tw.industry_group == "Semiconductors"
    assert tw.themes_list() == []

    tpex = service.get("8069", market="TW")
    assert tpex is not None
    assert tpex.symbol == "8069.TWO"
    assert tpex.industry_group == "Computer Peripherals"
    assert tpex.themes_list() == []

    kr_kospi = service.get("005930.KS", market="KR")
    assert kr_kospi is not None
    assert kr_kospi.symbol == "005930.KS"
    assert kr_kospi.sector == "Information Technology"
    assert kr_kospi.industry_group == "Technology Hardware"
    assert kr_kospi.industry == "Semiconductors"
    assert kr_kospi.sub_industry == "Memory Semiconductors"

    kr_kosdaq = service.get("091990", market="KR", exchange="KOSDAQ")
    assert kr_kosdaq is not None
    assert kr_kosdaq.symbol == "091990.KQ"
    assert kr_kosdaq.industry_group == "Biotechnology"
    assert kr_kosdaq.sub_industry == "Biopharmaceuticals"

    kr_kosdaq_umbrella = service.get("091990", market="KR", exchange="XKRX")
    assert kr_kosdaq_umbrella is not None
    assert kr_kosdaq_umbrella.symbol == "091990.KQ"
    assert service.entry_count_for_market("KR") == 2

    cn_sse = service.get("600519.SS", market="CN")
    assert cn_sse is not None
    assert cn_sse.symbol == "600519.SS"
    assert cn_sse.sector == "Consumer Staples"
    assert cn_sse.industry_group == "Food & Beverage"
    assert cn_sse.industry == "Beverage Manufacturing"
    assert cn_sse.sub_industry == "Liquor"

    cn_szse = service.get("000001", market="CN", exchange="SZSE")
    assert cn_szse is not None
    assert cn_szse.symbol == "000001.SZ"
    assert cn_szse.industry_group == "Banks"

    cn_bse = service.get("920118", market="CN", exchange="BJSE")
    assert cn_bse is not None
    assert cn_bse.symbol == "920118.BJ"
    assert cn_bse.industry_group == "Textiles"
    assert service.entry_count_for_market("CN") == 3


def test_groups_for_market_and_symbols_for_group(tmp_path):
    """The new group-listing helpers feed IBDIndustryService for non-US markets."""
    _write_csv(tmp_path / "IBD_industry_group.csv", "AAPL,Computer-Hardware/Peripherals\n")
    _write_csv(
        tmp_path / "hk-deep.csv",
        (
            "Symbol,EM Industry (EN),Theme (EN)\n"
            "700,Internet Services,AI Infrastructure\n"
            "1,Conglomerates,\n"
            "5,Banks,\n"
            "11,Banks,\n"
        ),
    )
    # Other CSVs still need to be present so `refresh()` can load them.
    _write_csv(
        tmp_path / "india-deep.csv",
        "Symbol,Exchange,Industry (Sector),Subgroup (Theme),Sub-industry\n",
    )
    _write_csv(
        tmp_path / "kabutan_themes_en.csv",
        "Symbol,TSE 33-Sector,TSE 17-Sector,Theme (EN)\n",
    )
    _write_csv(tmp_path / "taiwan-deep.csv", "Symbol,Market,Industry (EN)\n")
    _write_csv(tmp_path / "korea-deep.csv", "Symbol,Market,Sector,Industry Group,Industry,Sub-Industry\n")
    _write_csv(tmp_path / "china-deep.csv", "Symbol,Exchange,Sector,Industry Group,Industry,Sub-Industry\n")

    service = MarketTaxonomyService(data_dir=tmp_path)

    hk_groups = service.groups_for_market("HK")
    # Deduplicated and sorted
    assert hk_groups == ["Banks", "Conglomerates", "Internet Services"]

    banks_symbols = service.symbols_for_group("HK", "Banks")
    assert banks_symbols == ["0005.HK", "0011.HK"]

    assert service.symbols_for_group("HK", "Nonexistent Group") == []
    # Unknown market yields no groups (falls through the normalization path)
    assert service.groups_for_market("XX") == []


def test_entry_count_for_market_counts_loaded_rows_before_symbol_merge(tmp_path):
    _write_csv(tmp_path / "IBD_industry_group.csv", "AAPL,Computer-Hardware/Peripherals\n")
    _write_csv(tmp_path / "hk-deep.csv", "Symbol,EM Industry (EN),Theme (EN)\n")
    _write_csv(
        tmp_path / "india-deep.csv",
        "Symbol,Exchange,Industry (Sector),Subgroup (Theme),Sub-industry\n",
    )
    _write_csv(
        tmp_path / "kabutan_themes_en.csv",
        "Symbol,TSE 33-Sector,TSE 17-Sector,Theme (EN)\n",
    )
    _write_csv(tmp_path / "taiwan-deep.csv", "Symbol,Market,Industry (EN)\n")
    _write_csv(
        tmp_path / "korea-deep.csv",
        (
            "Symbol,Market,Sector,Industry Group,Industry,Sub-Industry\n"
            "005930,KOSPI,Information Technology,Technology Hardware,Semiconductors,Memory Semiconductors\n"
            "005930.KS,KOSPI,Information Technology,Hardware,Semiconductors,Memory Semiconductors\n"
        ),
    )
    _write_csv(tmp_path / "china-deep.csv", "Symbol,Exchange,Sector,Industry Group,Industry,Sub-Industry\n")

    service = MarketTaxonomyService(data_dir=tmp_path)

    assert service.symbols_for_group("KR", "Technology Hardware") == ["005930.KS"]
    assert service.entry_count_for_market("KR") == 2


def test_market_taxonomy_service_raises_load_error_for_missing_required_csv(tmp_path):
    _write_csv(tmp_path / "IBD_industry_group.csv", "AAPL,Computer-Hardware/Peripherals\n")
    _write_csv(
        tmp_path / "india-deep.csv",
        "Symbol,Exchange,Industry (Sector),Subgroup (Theme),Sub-industry\n",
    )
    _write_csv(
        tmp_path / "kabutan_themes_en.csv",
        "Symbol,TSE 33-Sector,TSE 17-Sector,Theme (EN)\n",
    )
    _write_csv(tmp_path / "taiwan-deep.csv", "Symbol,Market,Industry (EN)\n")
    _write_csv(tmp_path / "korea-deep.csv", "Symbol,Market,Sector,Industry Group,Industry,Sub-Industry\n")
    _write_csv(tmp_path / "china-deep.csv", "Symbol,Exchange,Sector,Industry Group,Industry,Sub-Industry\n")

    service = MarketTaxonomyService(data_dir=tmp_path)

    with pytest.raises(TaxonomyLoadError, match="hk-deep.csv"):
        service.groups_for_market("HK")


def test_market_taxonomy_service_raises_load_error_for_malformed_csv(tmp_path):
    _write_csv(tmp_path / "IBD_industry_group.csv", "AAPL,Computer-Hardware/Peripherals\n")
    _write_csv(tmp_path / "hk-deep.csv", "Ticker,Industry\n700,Internet Services\n")
    _write_csv(
        tmp_path / "india-deep.csv",
        "Symbol,Exchange,Industry (Sector),Subgroup (Theme),Sub-industry\n",
    )
    _write_csv(
        tmp_path / "kabutan_themes_en.csv",
        "Symbol,TSE 33-Sector,TSE 17-Sector,Theme (EN)\n",
    )
    _write_csv(tmp_path / "taiwan-deep.csv", "Symbol,Market,Industry (EN)\n")
    _write_csv(tmp_path / "korea-deep.csv", "Symbol,Market,Sector,Industry Group,Industry,Sub-Industry\n")
    _write_csv(tmp_path / "china-deep.csv", "Symbol,Exchange,Sector,Industry Group,Industry,Sub-Industry\n")

    service = MarketTaxonomyService(data_dir=tmp_path)

    with pytest.raises(TaxonomyLoadError, match="missing required columns"):
        service.groups_for_market("HK")


def _write_minimum_required_csvs(tmp_path) -> None:
    """Write the seven CSVs required by the pre-CA loaders so a test can
    isolate CA-specific behaviour without tripping required-CSV errors."""
    _write_csv(tmp_path / "IBD_industry_group.csv", "AAPL,Computer-Hardware/Peripherals\n")
    _write_csv(tmp_path / "hk-deep.csv", "Symbol,EM Industry (EN),Theme (EN)\n")
    _write_csv(
        tmp_path / "india-deep.csv",
        "Symbol,Exchange,Industry (Sector),Subgroup (Theme),Sub-industry\n",
    )
    _write_csv(
        tmp_path / "kabutan_themes_en.csv",
        "Symbol,TSE 33-Sector,TSE 17-Sector,Theme (EN)\n",
    )
    _write_csv(tmp_path / "taiwan-deep.csv", "Symbol,Market,Industry (EN)\n")
    _write_csv(
        tmp_path / "korea-deep.csv",
        "Symbol,Market,Sector,Industry Group,Industry,Sub-Industry\n",
    )
    _write_csv(
        tmp_path / "china-deep.csv",
        "Symbol,Exchange,Sector,Industry Group,Industry,Sub-Industry\n",
    )


def test_load_ca_skips_silently_when_csv_is_missing(tmp_path, caplog):
    _write_minimum_required_csvs(tmp_path)

    service = MarketTaxonomyService(data_dir=tmp_path)

    with caplog.at_level("INFO", logger="app.services.market_taxonomy_service"):
        # Force taxonomy load via any market lookup; CA loader runs as part of
        # refresh() and must not raise when canada-deep.csv is absent.
        assert service.entry_count_for_market("CA") == 0
        assert service.groups_for_market("CA") == []

    info_messages = [
        record.message
        for record in caplog.records
        if record.name == "app.services.market_taxonomy_service"
        and record.levelname == "INFO"
    ]
    assert any("CA taxonomy CSV not found" in message for message in info_messages)


def test_load_ca_loads_canada_deep_when_present(tmp_path):
    _write_minimum_required_csvs(tmp_path)
    _write_csv(
        tmp_path / "canada-deep.csv",
        (
            "Symbol,Exchange,Sector,Industry Group,Industry,Sub-Industry\n"
            "RY.TO,TSX,Financials,Banks,Diversified Banks,Diversified Banks\n"
            "SHOP.TO,TSX,Information Technology,Software,Application Software,E-commerce Platforms\n"
        ),
    )

    service = MarketTaxonomyService(data_dir=tmp_path)

    ry = service.get("RY.TO", market="CA")
    assert ry is not None
    assert ry.industry_group == "Banks"
    assert ry.sector == "Financials"
    assert service.entry_count_for_market("CA") == 2
    assert service.groups_for_market("CA") == ["Banks", "Software"]


def test_market_taxonomy_service_default_data_dir_prefers_container_app_data(tmp_path):
    runtime_root = tmp_path / "runtime"
    service_path = runtime_root / "app" / "app" / "services" / "market_taxonomy_service.py"
    app_data = runtime_root / "app" / "data"
    app_data.mkdir(parents=True)
    _write_csv(app_data / "IBD_industry_group.csv", "AAPL,Computer-Hardware/Peripherals\n")
    _write_csv(app_data / "hk-deep.csv", "Symbol,EM Industry (EN),Theme (EN)\n700,Internet Services,AI Infrastructure\n")
    _write_csv(app_data / "india-deep.csv", "Symbol,Exchange,Industry (Sector),Subgroup (Theme),Sub-industry\nRELIANCE.NS,XNSE,ENERGY (STOCKS),Oil & Gas,Integrated Oil & Gas\n")
    _write_csv(app_data / "kabutan_themes_en.csv", "Symbol,TSE 33-Sector,TSE 17-Sector,Theme (EN)\n7203,Transportation Equipment,Automobiles,Hybrid Vehicles\n")
    _write_csv(app_data / "taiwan-deep.csv", "Symbol,Market,Industry (EN)\n2330,TWSE,Semiconductors\n")
    _write_csv(app_data / "korea-deep.csv", "Symbol,Market,Sector,Industry Group,Industry,Sub-Industry\n005930,KOSPI,Information Technology,Technology Hardware,Semiconductors,Memory Semiconductors\n")
    _write_csv(app_data / "china-deep.csv", "Symbol,Exchange,Sector,Industry Group,Industry,Sub-Industry\n600519,SSE,Consumer Staples,Food & Beverage,Beverage Manufacturing,Liquor\n")

    resolved = MarketTaxonomyService._default_data_dir(service_path=service_path)

    assert resolved == app_data
