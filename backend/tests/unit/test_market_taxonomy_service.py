from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from app.services import market_taxonomy_service as taxonomy_module
from app.services.market_taxonomy_service import MarketTaxonomyService, TaxonomyLoadError


def _write_csv(path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_market_taxonomy_service_normalizes_hk_jp_and_tw_symbols(tmp_path):
    _write_csv(tmp_path / "IBD_industry_group.csv", "AAPL,Computer-Hardware/Peripherals\n")
    _write_csv(
        tmp_path / "hk-deep.csv",
        (
            "Symbol,HSICS Sector,EM Industry (EN),Theme (EN)\n"
            "700,Information Technology,Internet Services,AI Infrastructure\n"
        ),
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
    assert hk.sector == "Information Technology"
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
            "Symbol,HSICS Sector,EM Industry (EN),Theme (EN)\n"
            "700,Information Technology,Internet Services,AI Infrastructure\n"
            "1,Industrials,Conglomerates,\n"
            "5,Financials,Banks,\n"
            "11,Financials,Banks,\n"
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
    assert service.group_symbols_for_market("HK") == {
        "Banks": ["0005.HK", "0011.HK"],
        "Conglomerates": ["0001.HK"],
        "Internet Services": ["0700.HK"],
    }

    assert service.symbols_for_group("HK", "Nonexistent Group") == []
    # Unknown market yields no groups (falls through the normalization path)
    assert service.groups_for_market("XX") == []


def test_sector_map_for_market_majority_votes_native_taxonomy_sectors(tmp_path):
    _write_csv(tmp_path / "IBD_industry_group.csv", "AAPL,Computer-Hardware/Peripherals\n")
    _write_csv(
        tmp_path / "hk-deep.csv",
        (
            "Symbol,HSICS Sector,EM Industry (EN),Theme (EN)\n"
            "700,Information Technology,Internet Services,AI Infrastructure\n"
            "9988,Information Technology,Internet Services,E-Commerce\n"
            "388,Financials,Internet Services,Exchange Operators\n"
            "5,Financials,Banks,\n"
            "11,Financials,Banks,\n"
        ),
    )
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

    assert service.sector_for_group("HK", "Internet Services") == "Information Technology"
    assert service.sector_map_for_market("HK") == {
        "Banks": "Financials",
        "Internet Services": "Information Technology",
    }


def test_entry_count_for_market_counts_loaded_rows_before_symbol_merge(tmp_path):
    _write_csv(tmp_path / "IBD_industry_group.csv", "AAPL,Computer-Hardware/Peripherals\n")
    _write_csv(tmp_path / "hk-deep.csv", "Symbol,HSICS Sector,EM Industry (EN),Theme (EN)\n")
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


def test_refresh_tolerates_a_data_dir_without_any_csv(tmp_path, caplog):
    """A missing CSV skips its market instead of failing the whole taxonomy.

    Before the guards, the first absent file raised FileNotFoundError, which
    refresh() wrapped into TaxonomyLoadError and marked every market unloaded.
    """
    service = MarketTaxonomyService(data_dir=tmp_path)

    with caplog.at_level("INFO", logger="app.services.market_taxonomy_service"):
        assert service.groups_for_market("US") == []
        assert service.entry_count_for_market("HK") == 0

    for market in ("US", "HK", "JP", "TW", "KR", "CN"):
        assert service._loaded_row_counts[market] == 0
    messages = [
        record.message
        for record in caplog.records
        if record.name == "app.services.market_taxonomy_service"
    ]
    for market in ("US", "HK", "JP", "TW", "KR", "CN"):
        assert any(f"{market} taxonomy CSV not found" in message for message in messages)


def test_a_partial_load_keeps_the_markets_whose_csv_is_present(tmp_path):
    """A later missing file must not discard what earlier loaders produced."""
    _write_csv(
        tmp_path / "IBD_industry_group.csv",
        "AAPL,Computer-Hardware/Peripherals\n",
    )

    service = MarketTaxonomyService(data_dir=tmp_path)

    assert service.groups_for_market("US") == ["Computer-Hardware/Peripherals"]
    assert service.entry_count_for_market("US") == 1
    assert service.entry_count_for_market("HK") == 0


def test_a_csv_that_appears_after_the_first_lookup_is_picked_up(tmp_path, monkeypatch):
    """A skipped file must not latch the taxonomy for the life of the process."""
    monkeypatch.setattr(taxonomy_module, "_MISSING_FILE_RETRY_SECONDS", 0.0)
    service = MarketTaxonomyService(data_dir=tmp_path)
    assert service.get("AAPL", market="US") is None

    _write_csv(
        tmp_path / "IBD_industry_group.csv",
        "AAPL,Computer-Hardware/Peripherals\n",
    )

    entry = service.get("AAPL", market="US")
    assert entry is not None
    assert entry.industry_group == "Computer-Hardware/Peripherals"


def test_the_retry_deadline_holds_before_reloading(tmp_path, monkeypatch):
    """The retry must respect its interval, not reload on the next lookup."""
    clock = {"now": 1_000.0}
    monkeypatch.setattr(taxonomy_module, "time", SimpleNamespace(monotonic=lambda: clock["now"]))

    service = MarketTaxonomyService(data_dir=tmp_path)
    assert service.get("AAPL", market="US") is None

    _write_csv(
        tmp_path / "IBD_industry_group.csv",
        "AAPL,Computer-Hardware/Peripherals\n",
    )

    clock["now"] += taxonomy_module._MISSING_FILE_RETRY_SECONDS - 1
    assert service.get("AAPL", market="US") is None

    clock["now"] += 2
    entry = service.get("AAPL", market="US")
    assert entry is not None
    assert entry.industry_group == "Computer-Hardware/Peripherals"


def test_an_absent_file_renews_the_retry_deadline(tmp_path, monkeypatch):
    """A fruitless sweep must postpone the next one, not probe every lookup."""
    clock = {"now": 2_000.0}
    monkeypatch.setattr(taxonomy_module, "time", SimpleNamespace(monotonic=lambda: clock["now"]))

    service = MarketTaxonomyService(data_dir=tmp_path)
    assert service.get("AAPL", market="US") is None

    # Deadline passes with the directory still empty: the sweep renews it.
    clock["now"] += taxonomy_module._MISSING_FILE_RETRY_SECONDS + 1
    assert service.get("AAPL", market="US") is None

    # A file created inside the renewed interval is not picked up yet.
    _write_csv(
        tmp_path / "IBD_industry_group.csv",
        "AAPL,Computer-Hardware/Peripherals\n",
    )
    clock["now"] += 10
    assert service.get("AAPL", market="US") is None

    # Only after the renewed deadline does the lookup load it.
    clock["now"] += taxonomy_module._MISSING_FILE_RETRY_SECONDS
    entry = service.get("AAPL", market="US")
    assert entry is not None
    assert entry.industry_group == "Computer-Hardware/Peripherals"


def test_a_non_us_csv_that_appears_after_the_first_lookup_is_picked_up(tmp_path, monkeypatch):
    """Every guarded loader records its own path, not just the US one."""
    monkeypatch.setattr(taxonomy_module, "_MISSING_FILE_RETRY_SECONDS", 0.0)
    service = MarketTaxonomyService(data_dir=tmp_path)
    assert service.get("AAPL", market="US") is None
    assert service.entry_count_for_market("HK") == 0

    _write_csv(
        tmp_path / "hk-deep.csv",
        (
            "Symbol,HSICS Sector,EM Industry (EN),Theme (EN)\n"
            "700,Information Technology,Internet Services,AI Infrastructure\n"
        ),
    )

    service.get("AAPL", market="US")
    assert service.entry_count_for_market("HK") > 0


def test_a_failed_reload_keeps_the_previous_taxonomy(tmp_path, monkeypatch):
    """A half-written file that arrives later must not wipe a good load."""
    monkeypatch.setattr(taxonomy_module, "_MISSING_FILE_RETRY_SECONDS", 0.0)
    _write_csv(
        tmp_path / "IBD_industry_group.csv",
        "AAPL,Computer-Hardware/Peripherals\n",
    )
    service = MarketTaxonomyService(data_dir=tmp_path)
    assert service.groups_for_market("US") == ["Computer-Hardware/Peripherals"]

    # hk-deep.csv appears without the required columns, e.g. mid-copy.
    _write_csv(tmp_path / "hk-deep.csv", "Symbol,Name\n700,Tencent\n")

    assert service.groups_for_market("US") == ["Computer-Hardware/Peripherals"]
    assert service.entry_count_for_market("US") == 1
    assert service.entry_count_for_market("HK") == 0


def test_a_market_keeps_its_entries_when_its_csv_disappears(tmp_path, monkeypatch):
    """A reload driven by another market must not empty a loaded one."""
    monkeypatch.setattr(taxonomy_module, "_MISSING_FILE_RETRY_SECONDS", 0.0)
    _write_csv(
        tmp_path / "IBD_industry_group.csv",
        "AAPL,Computer-Hardware/Peripherals\n",
    )
    _write_csv(
        tmp_path / "hk-deep.csv",
        (
            "Symbol,HSICS Sector,EM Industry (EN),Theme (EN)\n"
            "700,Information Technology,Internet Services,AI Infrastructure\n"
        ),
    )
    service = MarketTaxonomyService(data_dir=tmp_path)
    assert service.entry_count_for_market("HK") == 1

    # hk-deep.csv vanishes while another skipped file shows up and triggers
    # the reload.
    (tmp_path / "hk-deep.csv").unlink()
    _write_csv(
        tmp_path / "india-deep.csv",
        (
            "Symbol,Exchange,Industry (Sector),Subgroup (Theme),Sub-industry\n"
            "RELIANCE.NS,XNSE,ENERGY (STOCKS),Oil & Gas,Integrated Oil & Gas\n"
        ),
    )

    assert service.entry_count_for_market("IN") == 1
    assert service.entry_count_for_market("HK") == 1
    assert service.get("700", market="HK") is not None


def test_an_india_csv_that_appears_after_the_first_lookup_is_picked_up(tmp_path, monkeypatch):
    """IN is guarded too: it used to skip without recording the path."""
    monkeypatch.setattr(taxonomy_module, "_MISSING_FILE_RETRY_SECONDS", 0.0)
    service = MarketTaxonomyService(data_dir=tmp_path)
    assert service.entry_count_for_market("IN") == 0

    _write_csv(
        tmp_path / "india-deep.csv",
        (
            "Symbol,Exchange,Industry (Sector),Subgroup (Theme),Sub-industry\n"
            "RELIANCE.NS,XNSE,ENERGY (STOCKS),Oil & Gas,Integrated Oil & Gas\n"
        ),
    )

    service.groups_for_market("US")
    assert service.entry_count_for_market("IN") == 1


def test_a_csv_that_was_still_being_copied_is_read_again(tmp_path, monkeypatch):
    """A complete header with truncated rows must not be treated as final."""
    monkeypatch.setattr(taxonomy_module, "_MISSING_FILE_RETRY_SECONDS", 0.0)
    service = MarketTaxonomyService(data_dir=tmp_path)
    assert service.entry_count_for_market("HK") == 0

    header = "Symbol,HSICS Sector,EM Industry (EN),Theme (EN)\n"
    first_row = "700,Information Technology,Internet Services,AI Infrastructure\n"
    path = tmp_path / "hk-deep.csv"
    _write_csv(path, header + first_row)

    # The file appears while it is still being copied: the rows written so far
    # load, and the previous taxonomy is untouched.
    assert service.entry_count_for_market("HK") == 1

    _write_csv(path, header + first_row + "5,Financials,Banks,Rate Sensitive\n")

    # The next pass sees the changed signature and reads the file again.
    assert service.entry_count_for_market("HK") == 2
    assert service.get("5", market="HK") is not None


def test_a_copy_that_finishes_during_the_reload_is_read_again(tmp_path, monkeypatch):
    """The signature must be taken when the file is opened, not after the load."""
    monkeypatch.setattr(taxonomy_module, "_MISSING_FILE_RETRY_SECONDS", 0.0)
    service = MarketTaxonomyService(data_dir=tmp_path)
    assert service.entry_count_for_market("HK") == 0

    header = "Symbol,HSICS Sector,EM Industry (EN),Theme (EN)\n"
    first = "700,Information Technology,Internet Services,AI Infrastructure\n"
    second = "5,Financials,Banks,Rate Sensitive\n"
    path = tmp_path / "hk-deep.csv"
    _write_csv(path, header + first)

    original_jp = taxonomy_module.MarketTaxonomyService._load_jp

    def jp_that_finishes_the_copy(self, load):
        # HK is parsed before JP runs, so the remaining rows land after the
        # parse but inside the same reload.
        original_jp(self, load)
        _write_csv(path, header + first + second)

    monkeypatch.setattr(
        taxonomy_module.MarketTaxonomyService, "_load_jp", jp_that_finishes_the_copy
    )

    assert service.entry_count_for_market("HK") == 1
    # The signature recorded at parse time shows the file grew, so the next
    # pass reads it again instead of serving the truncated parse forever.
    assert service.entry_count_for_market("HK") == 2


def test_a_lookup_does_not_wait_for_a_running_reload(tmp_path, monkeypatch):
    """Readers serve the published snapshot while another thread reloads."""
    monkeypatch.setattr(taxonomy_module, "_MISSING_FILE_RETRY_SECONDS", 0.0)
    _write_csv(
        tmp_path / "IBD_industry_group.csv",
        "AAPL,Computer-Hardware/Peripherals\n",
    )
    service = MarketTaxonomyService(data_dir=tmp_path)
    assert service.entry_count_for_market("US") == 1

    started = threading.Event()
    release = threading.Event()
    original_us = taxonomy_module.MarketTaxonomyService._load_us

    def slow_load(self, load):
        started.set()
        release.wait(5)
        return original_us(self, load)

    monkeypatch.setattr(taxonomy_module.MarketTaxonomyService, "_load_us", slow_load)
    _write_csv(
        tmp_path / "hk-deep.csv",
        "Symbol,HSICS Sector,EM Industry (EN),Theme (EN)\n700,IT,Internet,AI\n",
    )

    reloader = threading.Thread(target=service.entry_count_for_market, args=("HK",))
    reloader.start()
    assert started.wait(5)
    try:
        begin = time.monotonic()
        assert service.entry_count_for_market("US") == 1
        elapsed = time.monotonic() - begin
    finally:
        release.set()
        reloader.join()

    assert elapsed < 0.5


def test_concurrent_lookups_reload_the_taxonomy_once(tmp_path, monkeypatch):
    """Two threads past the deadline must not parse (or merge) twice."""
    monkeypatch.setattr(taxonomy_module, "_MISSING_FILE_RETRY_SECONDS", 0.0)
    _write_csv(
        tmp_path / "IBD_industry_group.csv",
        "AAPL,Computer-Hardware/Peripherals\n",
    )
    service = MarketTaxonomyService(data_dir=tmp_path)
    assert service.entry_count_for_market("US") == 1

    loads = []
    original = taxonomy_module.MarketTaxonomyService._load_us

    def counting_load(self, load):
        loads.append(1)
        time.sleep(0.05)
        return original(self, load)

    monkeypatch.setattr(
        taxonomy_module.MarketTaxonomyService, "_load_us", counting_load
    )
    # hk-deep.csv appears, so the next lookup is due for a reload.
    _write_csv(
        tmp_path / "hk-deep.csv",
        "Symbol,HSICS Sector,EM Industry (EN),Theme (EN)\n700,IT,Internet,AI\n",
    )

    barrier = threading.Barrier(4)
    results = []
    errors = []

    def lookup():
        barrier.wait()
        try:
            results.append(service.get("AAPL", market="US"))
        except Exception as exc:  # noqa: BLE001 - surfaced by the assertion below
            errors.append(exc)

    workers = [threading.Thread(target=lookup) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert errors == []
    assert len(results) == 4
    # Either the reloaded snapshot or the previous one, but never a partial
    # view: both are complete and both contain the US entry.
    assert all(entry is not None for entry in results)
    assert {entry.industry_group for entry in results} == {
        "Computer-Hardware/Peripherals"
    }
    assert len(loads) == 1
    assert service.entry_count_for_market("US") == 1


def test_an_unchanged_data_dir_is_not_reloaded(tmp_path, monkeypatch, caplog):
    """The retry stays cheap: a stat sweep, not a reload every 5 minutes."""
    monkeypatch.setattr(taxonomy_module, "_MISSING_FILE_RETRY_SECONDS", 0.0)
    service = MarketTaxonomyService(data_dir=tmp_path)
    service.get("AAPL", market="US")

    caplog.clear()
    with caplog.at_level("INFO", logger="app.services.market_taxonomy_service"):
        service.get("MSFT", market="US")
    assert caplog.records == []


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
    """Write the seven non-CA CSVs so a test isolates CA-specific behaviour
    from the missing-file log lines."""
    _write_csv(tmp_path / "IBD_industry_group.csv", "AAPL,Computer-Hardware/Peripherals\n")
    _write_csv(tmp_path / "hk-deep.csv", "Symbol,HSICS Sector,EM Industry (EN),Theme (EN)\n")
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

    levels = {
        record.levelname: record.message
        for record in caplog.records
        if record.name == "app.services.market_taxonomy_service"
        and "CA taxonomy CSV not found" in record.message
    }
    # canada-deep.csv is not shipped, so that absence is expected: INFO, not a
    # warning that fires in every worker on every load.
    assert set(levels) == {"INFO"}


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

    # Suffix-less and TSXV-suffix lookups should resolve via _candidate_symbols.
    ry_unsuffixed = service.get("RY", market="CA")
    assert ry_unsuffixed is not None
    assert ry_unsuffixed.symbol == "RY.TO"

    ry_tsxv_lookup = service.get("RY.V", market="CA")
    # CSV listed RY on TSX only — TSXV suffix should still find it via the
    # cross-exchange candidate fallback.
    assert ry_tsxv_lookup is not None
    assert ry_tsxv_lookup.symbol == "RY.TO"


def test_market_taxonomy_service_default_data_dir_prefers_container_app_data(tmp_path):
    runtime_root = tmp_path / "runtime"
    service_path = runtime_root / "app" / "app" / "services" / "market_taxonomy_service.py"
    app_data = runtime_root / "app" / "data"
    app_data.mkdir(parents=True)
    _write_csv(app_data / "IBD_industry_group.csv", "AAPL,Computer-Hardware/Peripherals\n")
    _write_csv(
        app_data / "hk-deep.csv",
        (
            "Symbol,HSICS Sector,EM Industry (EN),Theme (EN)\n"
            "700,Information Technology,Internet Services,AI Infrastructure\n"
        ),
    )
    _write_csv(app_data / "india-deep.csv", "Symbol,Exchange,Industry (Sector),Subgroup (Theme),Sub-industry\nRELIANCE.NS,XNSE,ENERGY (STOCKS),Oil & Gas,Integrated Oil & Gas\n")
    _write_csv(app_data / "kabutan_themes_en.csv", "Symbol,TSE 33-Sector,TSE 17-Sector,Theme (EN)\n7203,Transportation Equipment,Automobiles,Hybrid Vehicles\n")
    _write_csv(app_data / "taiwan-deep.csv", "Symbol,Market,Industry (EN)\n2330,TWSE,Semiconductors\n")
    _write_csv(app_data / "korea-deep.csv", "Symbol,Market,Sector,Industry Group,Industry,Sub-Industry\n005930,KOSPI,Information Technology,Technology Hardware,Semiconductors,Memory Semiconductors\n")
    _write_csv(app_data / "china-deep.csv", "Symbol,Exchange,Sector,Industry Group,Industry,Sub-Industry\n600519,SSE,Consumer Staples,Food & Beverage,Beverage Manufacturing,Liquor\n")

    resolved = MarketTaxonomyService._default_data_dir(service_path=service_path)

    assert resolved == app_data
