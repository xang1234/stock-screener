from __future__ import annotations

from app.services.market_taxonomy_service import MarketTaxonomyService


def _write_csv(path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_market_taxonomy_service_normalizes_hk_jp_and_tw_symbols(tmp_path):
    _write_csv(tmp_path / "IBD_industry_group.csv", "AAPL,Computer-Hardware/Peripherals\n")
    _write_csv(
        tmp_path / "hk-deep.csv",
        "Symbol,EM Industry (EN),Theme (EN)\n700,Internet Services,AI Infrastructure\n",
    )
    _write_csv(
        tmp_path / "kabutan_themes_en.csv",
        "Symbol,TSE 33-Sector,TSE 17-Sector,Theme (EN)\n7203,Transportation Equipment,Automobiles,Hybrid Vehicles\n",
    )
    _write_csv(
        tmp_path / "taiwan-deep.csv",
        "Symbol,Market,Industry (EN)\n2330,TWSE,Semiconductors\n8069,TPEX,Computer Peripherals\n",
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
