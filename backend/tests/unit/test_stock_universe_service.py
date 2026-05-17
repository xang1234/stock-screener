from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.stock_universe import (
    StockUniverse,
    StockUniverseReconciliationRun,
    StockUniverseStatusEvent,
    UNIVERSE_STATUS_ACTIVE,
    UNIVERSE_STATUS_INACTIVE_MISSING_SOURCE,
    UNIVERSE_STATUS_INACTIVE_MANUAL,
    UNIVERSE_STATUS_INACTIVE_NO_DATA,
)
from app.models.stock import StockIndustry
from app.models.ticker_validation import TickerValidationLog
from app.services.stock_universe_service import StockUniverseService

stock_universe_service = StockUniverseService()


class _FakeBulkFetcher:
    def __init__(self, results):
        self._results = results
        self.calls = []

    def fetch_prices_in_batches(self, symbols, period="1mo", market=None):
        self.calls.append(
            {
                "symbols": list(symbols),
                "period": period,
                "market": market,
            }
        )
        return {symbol: self._results.get(symbol, {"has_error": True, "price_data": None}) for symbol in symbols}


def _make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return TestingSessionLocal


def _assert_bulk_universe_rows_prepopulate_required_defaults(objects):
    for row in objects:
        assert row.consecutive_fetch_failures == 0


def test_ingest_cn_snapshot_rows_populates_stock_industry_taxonomy():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    service = StockUniverseService()

    stats = service.ingest_cn_snapshot_rows(
        db,
        rows=[
            {
                "symbol": "600519",
                "name": "Kweichow Moutai",
                "exchange": "SSE",
                "sector": "Consumer Staples",
                "industry_group": "Food & Beverage",
                "industry": "Beverage Manufacturing",
                "sub_industry": "Liquor",
            }
        ],
        source_name="cn_akshare_eastmoney",
        snapshot_id="cn-test-20260430",
        strict=True,
    )

    industry = db.query(StockIndustry).filter_by(symbol="600519.SS").one()
    universe = db.query(StockUniverse).filter_by(symbol="600519.SS").one()
    assert stats["stock_industry_upserts"] == 1
    assert universe.market == "CN"
    assert universe.currency == "CNY"
    assert industry.sector == "Consumer Staples"
    assert industry.industry_group == "Food & Beverage"
    assert industry.industry == "Beverage Manufacturing"
    assert industry.sub_industry == "Liquor"
    db.close()


def test_ingest_sg_snapshot_rows_canonicalizes_sgx_codes():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    service = StockUniverseService()

    stats = service.ingest_sg_snapshot_rows(
        db,
        rows=[
            {
                "symbol": "D05",
                "name": "DBS GROUP HOLDINGS LTD",
                "exchange": "SGX",
                "sector": "Finance",
                "industry": "Banking",
                "market_cap": "100.5",
                "isin": "SG1L01001701",
            },
            {
                "symbol": "A17U.SI",
                "name": "CAPITALAND ASCENDAS REIT",
                "exchange": "XSES",
                "sector": "Real Estate",
                "industry": "REIT",
            },
        ],
        source_name="sgx_official",
        snapshot_id="sgx-securities-2026-05-17",
        strict=True,
    )

    rows = db.query(StockUniverse).order_by(StockUniverse.symbol.asc()).all()
    assert stats["total"] == 2
    assert stats["rejected"] == 0
    assert [row.symbol for row in rows] == ["A17U.SI", "D05.SI"]
    assert rows[0].market == "SG"
    assert rows[0].exchange == "XSES"
    assert rows[0].currency == "SGD"
    assert rows[0].timezone == "Asia/Singapore"
    assert rows[0].local_code == "A17U"
    assert rows[1].market_cap == 100.5
    db.close()


def test_get_active_symbols_uses_is_active_over_stale_active_status():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    db.add_all(
        [
            StockUniverse(
                symbol="AAPL",
                exchange="NASDAQ",
                market_cap=1000,
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                status_reason="Present in Finviz universe sync",
            ),
            StockUniverse(
                symbol="OLD",
                exchange="NYSE",
                market_cap=10,
                is_active=False,
                status=UNIVERSE_STATUS_ACTIVE,
                status_reason=None,
            ),
        ]
    )
    db.commit()

    symbols = stock_universe_service.get_active_symbols(db)

    assert symbols == ["AAPL"]
    db.close()


def test_normalize_status_treats_active_status_plus_inactive_flag_as_inactive():
    record = StockUniverse(
        symbol="OLD",
        is_active=False,
        status=UNIVERSE_STATUS_ACTIVE,
        status_reason=None,
    )

    normalized = stock_universe_service._normalize_status(record)

    assert normalized != UNIVERSE_STATUS_ACTIVE


def test_get_active_symbols_market_filter_falls_back_to_exchange_when_market_blank():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    db.add_all(
        [
            StockUniverse(
                symbol="AAPL",
                exchange="NASDAQ",
                market="US",
                market_cap=1000,
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                status_reason="Present in Finviz universe sync",
            ),
            StockUniverse(
                symbol="IBM",
                exchange="NYSE",
                market="",
                market_cap=500,
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                status_reason="Present in Finviz universe sync",
            ),
            StockUniverse(
                symbol="2330.TW",
                exchange="TWSE",
                market="",
                market_cap=1200,
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                status_reason="Present in source universe sync",
            ),
        ]
    )
    db.commit()

    symbols = stock_universe_service.get_active_symbols(db, market="US")

    assert symbols == ["AAPL", "IBM"]
    db.close()


def test_populate_from_csv_sets_market_identity_fields_from_resolver():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()

    csv_content = "\n".join(
        [
            "symbol,name,exchange,sector,industry,market_cap",
            "0700.HK,Tencent,SEHK,Technology,Internet,500B",
        ]
    )
    stats = stock_universe_service.populate_from_csv(db, csv_content)

    row = db.query(StockUniverse).filter(StockUniverse.symbol == "0700.HK").one()
    assert stats["added"] == 1
    assert row.market == "HK"
    assert row.currency == "HKD"
    assert row.timezone == "Asia/Hong_Kong"
    assert row.local_code == "0700"
    db.close()


def test_populate_from_csv_persists_canonical_symbol_from_security_master():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()

    csv_content = "\n".join(
        [
            "symbol,name,exchange,sector,industry,market_cap",
            "700,Tencent,SEHK,Technology,Internet,500B",
        ]
    )
    stats = stock_universe_service.populate_from_csv(db, csv_content)

    row = db.query(StockUniverse).filter(StockUniverse.symbol == "700.HK").one()
    assert stats["added"] == 1
    assert row.exchange == "SEHK"
    assert row.market == "HK"
    assert row.local_code == "700"
    db.close()


def test_populate_from_csv_uses_tpex_two_suffix_for_unsuffixed_symbols():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()

    csv_content = "\n".join(
        [
            "symbol,name,exchange,sector,industry,market_cap",
            "3008,Largan,TPEX,Technology,Electronics,120B",
        ]
    )
    stock_universe_service.populate_from_csv(db, csv_content)

    row = db.query(StockUniverse).filter(StockUniverse.symbol == "3008.TWO").one()
    assert row.exchange == "TPEX"
    assert row.market == "TW"
    assert row.timezone == "Asia/Taipei"
    db.close()


def test_populate_from_csv_batches_new_universe_rows_and_status_events(monkeypatch):
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    bulk_calls: list[tuple[str, int]] = []
    original_bulk_save = db.bulk_save_objects
    original_add = db.add

    def _record_bulk(objects, **kwargs):
        objects = list(objects)
        if objects:
            bulk_calls.append((type(objects[0]).__name__, len(objects)))
            if isinstance(objects[0], StockUniverse):
                _assert_bulk_universe_rows_prepopulate_required_defaults(objects)
        return original_bulk_save(objects, **kwargs)

    def _guard_add(obj):
        if isinstance(obj, (StockUniverse, StockUniverseStatusEvent)):
            raise AssertionError("expected StockUniverse inserts to be batched")
        return original_add(obj)

    monkeypatch.setattr(db, "bulk_save_objects", _record_bulk)
    monkeypatch.setattr(db, "add", _guard_add)

    csv_content = "\n".join(
        [
            "symbol,name,exchange,sector,industry,market_cap",
            "700,Tencent,SEHK,Technology,Internet,500B",
            "3008,Largan,TPEX,Technology,Electronics,120B",
        ]
    )

    stats = stock_universe_service.populate_from_csv(db, csv_content)

    assert stats["added"] == 2
    assert ("StockUniverse", 2) in bulk_calls
    assert ("StockUniverseStatusEvent", 2) in bulk_calls
    db.close()


def test_ingest_hk_from_csv_normalizes_variants_with_zero_padding_and_lineage():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()

    csv_content = "\n".join(
        [
            "symbol,name,exchange,sector,industry,market_cap",
            "700,Tencent,SEHK,Technology,Internet,500B",
            "0700.HK,Tencent Holdings,HKEX,Technology,Internet,500B",
            "00700,Tencent Holdings,XHKG,Technology,Internet,500B",
        ]
    )

    stats = stock_universe_service.ingest_hk_from_csv(
        db,
        csv_content,
        source_name="hkex_official",
        snapshot_id="hk-20260412",
    )

    row = db.query(StockUniverse).filter(StockUniverse.symbol == "0700.HK").one()
    events = (
        db.query(StockUniverseStatusEvent)
        .filter(StockUniverseStatusEvent.symbol == "0700.HK")
        .all()
    )

    assert stats["added"] == 1
    assert stats["updated"] == 0
    assert stats["total"] == 1
    assert stats["rejected"] == 0
    assert row.local_code == "0700"
    assert row.exchange == "XHKG"
    assert row.market == "HK"
    assert row.source == "hk_ingest"
    assert len(events) == 1
    payload = json.loads(events[0].payload_json)
    assert payload["snapshot_id"] == "hk-20260412"
    assert payload["source_name"] == "hkex_official"
    assert len(payload["lineage_hash"]) == 64


def test_ingest_in_snapshot_rows_prefers_nse_and_keeps_bse_only_symbols():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    service = StockUniverseService()
    fake_fetcher = _FakeBulkFetcher(
        {
            "506854.BO": {"has_error": False, "price_data": [1, 2, 3]},
        }
    )
    service._bulk_fetcher = fake_fetcher

    stats = service.ingest_in_snapshot_rows(
        db,
        rows=[
            {
                "symbol": "RELIANCE.NS",
                "name": "Reliance Industries Limited",
                "exchange": "XNSE",
                "sector": "",
                "industry": "",
                "market_cap": None,
                "isin": "INE002A01018",
            },
            {
                "symbol": "506854.BO",
                "name": "TANFAC Industries Ltd.",
                "exchange": "XBOM",
                "sector": "",
                "industry": "",
                "market_cap": 4816.33,
                "isin": "INE639B01023",
            },
        ],
        source_name="in_reference_bundle",
        snapshot_id="in-reference-bundle-2026-04-21",
        snapshot_as_of="2026-04-21",
        source_metadata={"overlap_isin_count": 0},
    )

    rows = db.query(StockUniverse).order_by(StockUniverse.symbol).all()

    assert stats["added"] == 2
    assert stats["total"] == 2
    assert [row.symbol for row in rows] == ["506854.BO", "RELIANCE.NS"]
    assert rows[0].market == "IN"
    assert rows[0].exchange == "XBOM"
    assert rows[0].currency == "INR"
    assert rows[1].market == "IN"
    assert rows[1].exchange == "XNSE"
    assert rows[1].currency == "INR"
    assert fake_fetcher.calls == [{"symbols": ["506854.BO"], "period": "1mo", "market": "IN"}]
    db.close()


def test_ingest_in_snapshot_rows_filters_bse_only_symbols_without_price_coverage():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    service = StockUniverseService()
    service._bulk_fetcher = _FakeBulkFetcher(
        {
            "500002.BO": {"has_error": True, "price_data": None},
        }
    )

    stats = service.ingest_in_snapshot_rows(
        db,
        rows=[
            {
                "symbol": "RELIANCE.NS",
                "name": "Reliance Industries Limited",
                "exchange": "XNSE",
                "sector": "",
                "industry": "",
                "market_cap": None,
                "isin": "INE002A01018",
            },
            {
                "symbol": "500002.BO",
                "name": "ABB India Limited",
                "exchange": "XBOM",
                "sector": "",
                "industry": "",
                "market_cap": 151635.28,
                "isin": "INE117A01022",
            },
        ],
        source_name="in_reference_bundle",
        snapshot_id="in-reference-bundle-2026-04-21",
        snapshot_as_of="2026-04-21",
        source_metadata={"overlap_isin_count": 0},
    )

    rows = db.query(StockUniverse).order_by(StockUniverse.symbol).all()

    assert stats["added"] == 1
    assert stats["total"] == 1
    assert stats["coverage_rejected"] == 1
    assert [row.symbol for row in rows] == ["RELIANCE.NS"]
    db.close()


def test_ingest_in_snapshot_rows_deactivates_existing_active_bse_symbol_rejected_by_coverage_gate():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    service = StockUniverseService()
    service._bulk_fetcher = _FakeBulkFetcher(
        {
            "500002.BO": {"has_error": True, "price_data": None},
        }
    )
    db.add(
        StockUniverse(
            symbol="500002.BO",
            name="ABB India Limited",
            market="IN",
            exchange="XBOM",
            currency="INR",
            timezone="Asia/Kolkata",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
            source="in_ingest",
        )
    )
    db.commit()

    stats = service.ingest_in_snapshot_rows(
        db,
        rows=[
            {
                "symbol": "500002.BO",
                "name": "ABB India Limited",
                "exchange": "XBOM",
                "sector": "",
                "industry": "",
                "market_cap": 151635.28,
                "isin": "INE117A01022",
            },
        ],
        source_name="in_reference_bundle",
        snapshot_id="in-reference-bundle-2026-04-21",
        snapshot_as_of="2026-04-21",
        source_metadata={"overlap_isin_count": 0},
    )

    row = db.query(StockUniverse).filter(StockUniverse.symbol == "500002.BO").one()
    event = (
        db.query(StockUniverseStatusEvent)
        .filter(StockUniverseStatusEvent.symbol == "500002.BO")
        .order_by(StockUniverseStatusEvent.id.desc())
        .first()
    )

    assert stats["added"] == 0
    assert stats["total"] == 0
    assert stats["coverage_rejected"] == 1
    assert row.is_active is False
    assert row.status == UNIVERSE_STATUS_INACTIVE_NO_DATA
    assert "Rejected by IN BSE coverage gate" in row.status_reason
    assert event is not None
    assert event.new_status == UNIVERSE_STATUS_INACTIVE_NO_DATA
    assert event.trigger_source == "in_ingest_coverage_gate"
    db.close()


def test_ingest_in_snapshot_rows_filters_bse_only_symbols_with_repeated_yfinance_failures():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    service = StockUniverseService()
    fake_fetcher = _FakeBulkFetcher(
        {
            "506854.BO": {"has_error": False, "price_data": [1, 2, 3]},
            "500002.BO": {"has_error": False, "price_data": [1, 2, 3]},
        }
    )
    service._bulk_fetcher = fake_fetcher
    db.add(
        TickerValidationLog(
            symbol="500002.BO",
            error_type="no_data",
            error_message="No data returned from API",
            data_source="yfinance",
            triggered_by="fundamentals_refresh",
            is_resolved=False,
            consecutive_failures=3,
            detected_at=datetime.utcnow(),
        )
    )
    db.commit()

    stats = service.ingest_in_snapshot_rows(
        db,
        rows=[
            {
                "symbol": "506854.BO",
                "name": "TANFAC Industries Ltd.",
                "exchange": "XBOM",
                "sector": "",
                "industry": "",
                "market_cap": 4816.33,
                "isin": "INE639B01023",
            },
            {
                "symbol": "500002.BO",
                "name": "ABB India Limited",
                "exchange": "XBOM",
                "sector": "",
                "industry": "",
                "market_cap": 151635.28,
                "isin": "INE117A01022",
            },
        ],
        source_name="in_reference_bundle",
        snapshot_id="in-reference-bundle-2026-04-21",
        snapshot_as_of="2026-04-21",
        source_metadata={"overlap_isin_count": 0},
    )

    rows = db.query(StockUniverse).order_by(StockUniverse.symbol).all()

    assert stats["added"] == 1
    assert stats["total"] == 1
    assert stats["coverage_rejected"] == 1
    assert [row.symbol for row in rows] == ["506854.BO"]
    assert fake_fetcher.calls == [{"symbols": ["506854.BO"], "period": "1mo", "market": "IN"}]
    db.close()


def test_ingest_in_snapshot_rows_truncates_combined_rejected_preview():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    service = StockUniverseService()
    coverage_rows = [
        {
            "symbol": f"{500100 + index:06d}.BO",
            "name": f"BSE Only {index}",
            "exchange": "XBOM",
            "sector": "",
            "industry": "",
            "market_cap": 1000.0 + index,
            "isin": f"INE{100000000000 + index:012d}",
        }
        for index in range(10)
    ]
    fake_fetcher = _FakeBulkFetcher(
        {
            row["symbol"]: {"has_error": True, "price_data": None}
            for row in coverage_rows
        }
    )
    service._bulk_fetcher = fake_fetcher

    invalid_rows = [
        {
            "symbol": f"INVALID{index}",
            "name": "",
            "exchange": "XNSE",
            "sector": "",
            "industry": "",
            "market_cap": None,
            "isin": f"INE{200000000000 + index:012d}",
        }
        for index in range(20)
    ]

    stats = service.ingest_in_snapshot_rows(
        db,
        rows=[*invalid_rows, *coverage_rows],
        source_name="in_reference_bundle",
        snapshot_id="in-reference-bundle-2026-04-21",
        snapshot_as_of="2026-04-21",
        source_metadata={"overlap_isin_count": 0},
        strict=False,
    )

    assert stats["rejected"] == 30
    assert stats["coverage_rejected"] == 10
    assert len(stats["rejected_rows"]) == 25
    assert stats["rejected_rows_truncated"] is True
    assert fake_fetcher.calls == [
        {
            "symbols": [row["symbol"] for row in coverage_rows],
            "period": "1mo",
            "market": "IN",
        }
    ]
    db.close()


def test_ingest_hk_from_csv_reactivates_existing_inactive_symbol():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    db.add(
        StockUniverse(
            symbol="0700.HK",
            exchange="SEHK",
            market="HK",
            is_active=False,
            status=UNIVERSE_STATUS_INACTIVE_MANUAL,
            status_reason="manual off",
        )
    )
    db.commit()

    csv_content = "\n".join(
        [
            "symbol,name,exchange,sector,industry,market_cap",
            "700,Tencent,SEHK,Technology,Internet,500B",
        ]
    )
    stats = stock_universe_service.ingest_hk_from_csv(
        db,
        csv_content,
        source_name="sehk_official",
        snapshot_id="hk-20260412",
    )
    row = db.query(StockUniverse).filter(StockUniverse.symbol == "0700.HK").one()

    assert stats["added"] == 0
    assert stats["updated"] == 1
    assert row.is_active is True
    assert row.status == UNIVERSE_STATUS_ACTIVE
    assert row.exchange == "XHKG"
    assert row.local_code == "0700"
    db.close()


@pytest.mark.parametrize(
    ("method_name", "source_name", "snapshot_id", "csv_content", "expected_added"),
    [
        (
            "ingest_hk_from_csv",
            "sehk_official",
            "hk-20260412",
            "\n".join(
                [
                    "symbol,name,exchange,sector,industry,market_cap",
                    "700,Tencent,SEHK,Technology,Internet,500B",
                    "1299,AIA,SEHK,Financial,Insurance,100B",
                ]
            ),
            2,
        ),
        (
            "ingest_jp_from_csv",
            "jpx_official",
            "jp-20260412",
            "\n".join(
                [
                    "symbol,name,exchange,sector,industry,market_cap",
                    "7203,Toyota,TSE,Consumer Cyclical,Auto Manufacturers,300B",
                    "6758,Sony,TSE,Technology,Consumer Electronics,150B",
                ]
            ),
            2,
        ),
        (
            "ingest_tw_from_csv",
            "tw_reference_bundle",
            "tw-20260412",
            "\n".join(
                [
                    "symbol,name,exchange,sector,industry,market_cap",
                    "2330,TSMC,TWSE,Technology,Semiconductors,800B",
                    "3008,Largan,TPEX,Technology,Electronics,120B",
                ]
            ),
            2,
        ),
    ],
)
def test_market_ingest_batches_new_universe_rows_and_status_events(
    monkeypatch,
    method_name,
    source_name,
    snapshot_id,
    csv_content,
    expected_added,
):
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    bulk_calls: list[tuple[str, int]] = []
    original_bulk_save = db.bulk_save_objects
    original_add = db.add

    def _record_bulk(objects, **kwargs):
        objects = list(objects)
        if objects:
            bulk_calls.append((type(objects[0]).__name__, len(objects)))
            if isinstance(objects[0], StockUniverse):
                _assert_bulk_universe_rows_prepopulate_required_defaults(objects)
        return original_bulk_save(objects, **kwargs)

    def _guard_add(obj):
        if isinstance(obj, (StockUniverse, StockUniverseStatusEvent)):
            raise AssertionError("expected market ingest inserts to be batched")
        return original_add(obj)

    monkeypatch.setattr(db, "bulk_save_objects", _record_bulk)
    monkeypatch.setattr(db, "add", _guard_add)

    stats = getattr(stock_universe_service, method_name)(
        db,
        csv_content,
        source_name=source_name,
        snapshot_id=snapshot_id,
    )

    assert stats["added"] == expected_added
    assert ("StockUniverse", expected_added) in bulk_calls
    assert ("StockUniverseStatusEvent", expected_added) in bulk_calls
    db.close()


def test_populate_universe_batches_new_rows_with_required_defaults(monkeypatch):
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    bulk_calls: list[tuple[str, int]] = []
    original_bulk_save = db.bulk_save_objects
    original_add = db.add

    def _record_bulk(objects, **kwargs):
        objects = list(objects)
        if objects:
            bulk_calls.append((type(objects[0]).__name__, len(objects)))
            if isinstance(objects[0], StockUniverse):
                _assert_bulk_universe_rows_prepopulate_required_defaults(objects)
        return original_bulk_save(objects, **kwargs)

    def _guard_add(obj):
        if isinstance(obj, (StockUniverse, StockUniverseStatusEvent)):
            raise AssertionError("expected populate_universe inserts to be batched")
        return original_add(obj)

    monkeypatch.setattr(db, "bulk_save_objects", _record_bulk)
    monkeypatch.setattr(db, "add", _guard_add)
    monkeypatch.setattr(
        stock_universe_service,
        "fetch_from_finviz",
        lambda exchange_filter=None: [
            {
                "symbol": "AAPL",
                "name": "Apple",
                "exchange": "NASDAQ",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "market_cap": 3_000_000_000_000.0,
            },
            {
                "symbol": "MSFT",
                "name": "Microsoft",
                "exchange": "NASDAQ",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 2_800_000_000_000.0,
            },
        ],
    )

    stats = stock_universe_service.populate_universe(db, exchange_filter="NASDAQ")

    assert stats["added"] == 2
    assert ("StockUniverse", 2) in bulk_calls
    assert ("StockUniverseStatusEvent", 2) in bulk_calls
    db.close()


def test_ingest_hk_from_csv_rejects_unapproved_source():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    csv_content = "symbol,name,exchange\n700,Tencent,SEHK\n"

    with pytest.raises(ValueError, match="Unapproved HK source"):
        stock_universe_service.ingest_hk_from_csv(
            db,
            csv_content,
            source_name="random_vendor",
            snapshot_id="hk-20260412",
        )
    db.close()


def test_ingest_hk_from_csv_reports_rejected_rows_in_non_strict_mode():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    csv_content = "\n".join(
        [
            "symbol,name,exchange",
            "ABC.HK,Invalid,SEHK",
            "700,Tencent,SEHK",
        ]
    )

    stats = stock_universe_service.ingest_hk_from_csv(
        db,
        csv_content,
        source_name="sehk_official",
        snapshot_id="hk-20260412",
        strict=False,
    )

    assert stats["added"] == 1
    assert stats["total"] == 1
    assert stats["rejected"] == 1
    assert stats["rejected_rows"][0]["source_symbol"] == "ABC.HK"
    db.close()


def test_ingest_jp_from_csv_normalizes_exchange_formats_and_lineage():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()

    csv_content = "\n".join(
        [
            "symbol,name,exchange,sector,industry,market_cap",
            "7203,Toyota Motor,TSE,Consumer Cyclical,Auto Manufacturers,300B",
            "7203.T,Toyota Motor,JPX,Consumer Cyclical,Auto Manufacturers,300B",
            "JPX:7203,Toyota Motor,XTKS,Consumer Cyclical,Auto Manufacturers,300B",
        ]
    )

    stats = stock_universe_service.ingest_jp_from_csv(
        db,
        csv_content,
        source_name="jpx_official",
        snapshot_id="jp-20260412",
    )

    row = db.query(StockUniverse).filter(StockUniverse.symbol == "7203.T").one()
    events = (
        db.query(StockUniverseStatusEvent)
        .filter(StockUniverseStatusEvent.symbol == "7203.T")
        .all()
    )

    assert stats["added"] == 1
    assert stats["updated"] == 0
    assert stats["total"] == 1
    assert stats["rejected"] == 0
    assert row.local_code == "7203"
    assert row.exchange == "XTKS"
    assert row.market == "JP"
    assert row.currency == "JPY"
    assert row.timezone == "Asia/Tokyo"
    assert row.source == "jp_ingest"
    assert len(events) == 1
    payload = json.loads(events[0].payload_json)
    assert payload["snapshot_id"] == "jp-20260412"
    assert payload["source_name"] == "jpx_official"
    assert len(payload["lineage_hash"]) == 64
    assert len(payload["row_hash"]) == 64
    db.close()


def test_ingest_jp_from_csv_reactivates_existing_inactive_symbol():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    db.add(
        StockUniverse(
            symbol="7203.T",
            exchange="TSE",
            market="JP",
            is_active=False,
            status=UNIVERSE_STATUS_INACTIVE_MANUAL,
            status_reason="manual off",
        )
    )
    db.commit()

    csv_content = "\n".join(
        [
            "symbol,name,exchange,sector,industry,market_cap",
            "7203.T,Toyota Motor,TSE,Consumer Cyclical,Auto Manufacturers,300B",
        ]
    )
    stats = stock_universe_service.ingest_jp_from_csv(
        db,
        csv_content,
        source_name="tse_official",
        snapshot_id="jp-20260412",
    )
    row = db.query(StockUniverse).filter(StockUniverse.symbol == "7203.T").one()

    assert stats["added"] == 0
    assert stats["updated"] == 1
    assert row.is_active is True
    assert row.status == UNIVERSE_STATUS_ACTIVE
    assert row.exchange == "XTKS"
    assert row.local_code == "7203"
    db.close()


def test_ingest_jp_from_csv_rejects_unapproved_source():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    csv_content = "symbol,name,exchange\n7203,Toyota,TSE\n"

    with pytest.raises(ValueError, match="Unapproved JP source"):
        stock_universe_service.ingest_jp_from_csv(
            db,
            csv_content,
            source_name="random_vendor",
            snapshot_id="jp-20260412",
        )
    db.close()


def test_ingest_jp_from_csv_reports_rejected_rows_in_non_strict_mode():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    csv_content = "\n".join(
        [
            "symbol,name,exchange",
            "ABCD,Invalid,TSE",
            "7203,Toyota,TSE",
        ]
    )

    stats = stock_universe_service.ingest_jp_from_csv(
        db,
        csv_content,
        source_name="tse_official",
        snapshot_id="jp-20260412",
        strict=False,
    )

    assert stats["added"] == 1
    assert stats["total"] == 1
    assert stats["rejected"] == 1
    assert stats["rejected_rows"][0]["source_symbol"] == "ABCD"
    db.close()


def test_ingest_jp_from_csv_merges_duplicate_rows_to_keep_richer_metadata():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    csv_content = "\n".join(
        [
            "symbol,name,exchange,sector,industry,market_cap",
            "7203,,TSE,,,",
            "7203.T,Toyota Motor,JPX,Consumer Cyclical,Auto Manufacturers,300B",
        ]
    )

    stats = stock_universe_service.ingest_jp_from_csv(
        db,
        csv_content,
        source_name="jpx_official",
        snapshot_id="jp-20260412",
    )
    row = db.query(StockUniverse).filter(StockUniverse.symbol == "7203.T").one()

    assert stats["total"] == 1
    assert row.name == "Toyota Motor"
    assert row.sector == "Consumer Cyclical"
    assert row.industry == "Auto Manufacturers"
    assert row.market_cap == 300_000_000_000.0
    db.close()


def test_ingest_jp_snapshot_rows_truncates_verbose_row_payloads():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    rows = [
        {
            "symbol": f"{1300 + index}",
            "exchange": "TSE",
            "name": f"Company {index}",
        }
        for index in range(30)
    ]

    stats = stock_universe_service.ingest_jp_snapshot_rows(
        db,
        rows=rows,
        source_name="tse_official",
        snapshot_id="jp-20260412",
    )

    assert stats["total"] == 30
    assert len(stats["canonical_rows"]) == 25
    assert stats["canonical_rows_truncated"] is True
    assert stats["rejected_rows_truncated"] is False
    db.close()


def test_ingest_tw_from_csv_normalizes_twse_tpex_variants_and_lineage():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()

    csv_content = "\n".join(
        [
            "symbol,name,exchange,sector,industry,market_cap",
            "2330,Taiwan Semiconductor,TWSE,Technology,Semiconductors,800B",
            "TWSE:2330,TSMC,XTAI,Technology,Semiconductors,800B",
            "3008.TW,Largan Precision,TPEX,Technology,Electronics,120B",
            "TWO:3008,Largan Precision,TWO,Technology,Electronics,120B",
        ]
    )

    stats = stock_universe_service.ingest_tw_from_csv(
        db,
        csv_content,
        source_name="tw_reference_bundle",
        snapshot_id="tw-20260412",
    )

    twse_row = db.query(StockUniverse).filter(StockUniverse.symbol == "2330.TW").one()
    tpex_row = db.query(StockUniverse).filter(StockUniverse.symbol == "3008.TWO").one()
    events = (
        db.query(StockUniverseStatusEvent)
        .filter(StockUniverseStatusEvent.symbol.in_(["2330.TW", "3008.TWO"]))
        .all()
    )

    assert stats["added"] == 2
    assert stats["updated"] == 0
    assert stats["total"] == 2
    assert stats["rejected"] == 0
    assert twse_row.exchange == "TWSE"
    assert twse_row.market == "TW"
    assert twse_row.currency == "TWD"
    assert twse_row.timezone == "Asia/Taipei"
    assert tpex_row.exchange == "TPEX"
    assert tpex_row.market == "TW"
    assert tpex_row.symbol == "3008.TWO"
    assert tpex_row.source == "tw_ingest"
    assert len(events) == 2
    payload = json.loads(events[0].payload_json)
    assert payload["snapshot_id"] == "tw-20260412"
    assert payload["source_name"] == "tw_reference_bundle"
    assert len(payload["lineage_hash"]) == 64
    assert len(payload["row_hash"]) == 64
    db.close()


def test_ingest_tw_from_csv_reactivates_existing_inactive_symbol():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    db.add(
        StockUniverse(
            symbol="3008.TWO",
            exchange="TPEX",
            market="TW",
            is_active=False,
            status=UNIVERSE_STATUS_INACTIVE_MANUAL,
            status_reason="manual off",
        )
    )
    db.commit()

    csv_content = "\n".join(
        [
            "symbol,name,exchange,sector,industry,market_cap",
            "3008,Largan Precision,TPEX,Technology,Electronics,120B",
        ]
    )
    stats = stock_universe_service.ingest_tw_from_csv(
        db,
        csv_content,
        source_name="tpex_official",
        snapshot_id="tw-20260412",
    )
    row = db.query(StockUniverse).filter(StockUniverse.symbol == "3008.TWO").one()

    assert stats["added"] == 0
    assert stats["updated"] == 1
    assert row.is_active is True
    assert row.status == UNIVERSE_STATUS_ACTIVE
    assert row.exchange == "TPEX"
    assert row.local_code == "3008"
    db.close()


def test_ingest_tw_from_csv_infers_tpex_exchange_from_symbol_when_exchange_missing():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    csv_content = "\n".join(
        [
            "symbol,name,exchange,sector,industry,market_cap",
            "3008.TWO,Largan Precision,,Technology,Electronics,120B",
            "TWO:3008,Largan Precision,,Technology,Electronics,120B",
        ]
    )

    stats = stock_universe_service.ingest_tw_from_csv(
        db,
        csv_content,
        source_name="tw_reference_bundle",
        snapshot_id="tw-20260412",
    )
    row = db.query(StockUniverse).filter(StockUniverse.symbol == "3008.TWO").one()

    assert stats["added"] == 1
    assert stats["updated"] == 0
    assert stats["total"] == 1
    assert stats["rejected"] == 0
    assert row.exchange == "TPEX"
    assert row.market == "TW"
    assert row.local_code == "3008"
    db.close()


def test_ingest_tw_from_csv_rejects_unapproved_source():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    csv_content = "symbol,name,exchange\n2330,TSMC,TWSE\n"

    with pytest.raises(ValueError, match="Unapproved TW source"):
        stock_universe_service.ingest_tw_from_csv(
            db,
            csv_content,
            source_name="random_vendor",
            snapshot_id="tw-20260412",
        )
    db.close()


def test_ingest_tw_from_csv_reports_rejected_rows_in_non_strict_mode():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    csv_content = "\n".join(
        [
            "symbol,name,exchange",
            "ABCD,Invalid,TWSE",
            "2330,TSMC,TWSE",
        ]
    )

    stats = stock_universe_service.ingest_tw_from_csv(
        db,
        csv_content,
        source_name="twse_official",
        snapshot_id="tw-20260412",
        strict=False,
    )

    assert stats["added"] == 1
    assert stats["total"] == 1
    assert stats["rejected"] == 1
    assert stats["rejected_rows"][0]["source_symbol"] == "ABCD"
    db.close()


def test_ingest_tw_from_csv_merges_duplicate_rows_to_keep_richer_metadata():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    csv_content = "\n".join(
        [
            "symbol,name,exchange,sector,industry,market_cap",
            "2330,,TWSE,,,",
            "2330.TW,Taiwan Semiconductor,XTAI,Technology,Semiconductors,800B",
        ]
    )

    stats = stock_universe_service.ingest_tw_from_csv(
        db,
        csv_content,
        source_name="twse_official",
        snapshot_id="tw-20260412",
    )
    row = db.query(StockUniverse).filter(StockUniverse.symbol == "2330.TW").one()

    assert stats["total"] == 1
    assert row.name == "Taiwan Semiconductor"
    assert row.sector == "Technology"
    assert row.industry == "Semiconductors"
    assert row.market_cap == 800_000_000_000.0
    db.close()


def test_ingest_tw_snapshot_rows_truncates_verbose_row_payloads():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    rows = [
        {
            "symbol": f"{1100 + index}",
            "exchange": "TWSE",
            "name": f"Company {index}",
        }
        for index in range(30)
    ]

    stats = stock_universe_service.ingest_tw_snapshot_rows(
        db,
        rows=rows,
        source_name="twse_official",
        snapshot_id="tw-20260412",
    )

    assert stats["total"] == 30
    assert len(stats["canonical_rows"]) == 25
    assert stats["canonical_rows_truncated"] is True
    assert stats["rejected_rows_truncated"] is False
    db.close()


def test_ingest_hk_snapshot_rows_persists_reconciliation_diff_against_prior_snapshot():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()

    snapshot_one_rows = [
        {
            "symbol": "700",
            "exchange": "SEHK",
            "name": "Tencent",
            "sector": "Technology",
            "industry": "Internet",
        },
        {
            "symbol": "5",
            "exchange": "SEHK",
            "name": "HSBC",
            "sector": "Financial",
            "industry": "Banks",
        },
    ]
    first_stats = stock_universe_service.ingest_hk_snapshot_rows(
        db,
        rows=snapshot_one_rows,
        source_name="hkex_official",
        snapshot_id="hk-20260412-a",
    )

    first_reconciliation = first_stats["reconciliation"]
    first_run = (
        db.query(StockUniverseReconciliationRun)
        .filter(
            StockUniverseReconciliationRun.market == "HK",
            StockUniverseReconciliationRun.snapshot_id == "hk-20260412-a",
        )
        .one()
    )
    first_artifact = json.loads(first_run.artifact_json)

    assert first_reconciliation["previous_snapshot_id"] is None
    assert first_reconciliation["counts"]["added"] == 2
    assert first_reconciliation["counts"]["removed"] == 0
    assert first_reconciliation["counts"]["changed"] == 0
    assert first_reconciliation["counts"]["unchanged"] == 0
    assert len(first_run.artifact_hash) == 64
    assert first_artifact["added_symbols"] == ["0005.HK", "0700.HK"]

    snapshot_two_rows = [
        {
            "symbol": "0700.HK",
            "exchange": "HKEX",
            "name": "Tencent Holdings",
            "sector": "Technology",
            "industry": "Internet",
        },
        {
            "symbol": "16",
            "exchange": "SEHK",
            "name": "Sun Hung Kai",
            "sector": "Real Estate",
            "industry": "Property",
        },
    ]
    second_stats = stock_universe_service.ingest_hk_snapshot_rows(
        db,
        rows=snapshot_two_rows,
        source_name="hkex_official",
        snapshot_id="hk-20260412-b",
    )
    second_reconciliation = second_stats["reconciliation"]
    second_run = (
        db.query(StockUniverseReconciliationRun)
        .filter(
            StockUniverseReconciliationRun.market == "HK",
            StockUniverseReconciliationRun.snapshot_id == "hk-20260412-b",
        )
        .one()
    )
    second_artifact = json.loads(second_run.artifact_json)

    assert second_reconciliation["previous_snapshot_id"] == "hk-20260412-a"
    assert second_reconciliation["counts"]["total_current"] == 2
    assert second_reconciliation["counts"]["total_previous"] == 2
    assert second_reconciliation["counts"]["added"] == 1
    assert second_reconciliation["counts"]["removed"] == 1
    assert second_reconciliation["counts"]["changed"] == 1
    assert second_reconciliation["counts"]["unchanged"] == 0
    assert second_artifact["added_symbols"] == ["0016.HK"]
    assert second_artifact["removed_symbols"] == ["0005.HK"]
    assert second_artifact["changed_rows"][0]["symbol"] == "0700.HK"
    assert "name" in second_artifact["changed_rows"][0]["changed_fields"]
    db.close()


def test_ingest_tw_snapshot_rows_reconciliation_is_idempotent_for_same_snapshot():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    rows = [
        {"symbol": "2330", "exchange": "TWSE", "name": "TSMC"},
        {"symbol": "3008", "exchange": "TPEX", "name": "Largan"},
    ]
    reversed_rows = list(reversed(rows))

    first_stats = stock_universe_service.ingest_tw_snapshot_rows(
        db,
        rows=rows,
        source_name="tw_reference_bundle",
        snapshot_id="tw-20260412-a",
    )
    second_stats = stock_universe_service.ingest_tw_snapshot_rows(
        db,
        rows=reversed_rows,
        source_name="tw_reference_bundle",
        snapshot_id="tw-20260412-a",
    )

    runs = (
        db.query(StockUniverseReconciliationRun)
        .filter(
            StockUniverseReconciliationRun.market == "TW",
            StockUniverseReconciliationRun.snapshot_id == "tw-20260412-a",
        )
        .all()
    )

    assert len(runs) == 1
    assert first_stats["reconciliation"]["artifact_hash"] == second_stats["reconciliation"]["artifact_hash"]
    assert second_stats["reconciliation"]["counts"]["added"] == 2
    assert second_stats["reconciliation"]["counts"]["removed"] == 0
    assert second_stats["reconciliation"]["counts"]["changed"] == 0
    assert second_stats["reconciliation"]["counts"]["unchanged"] == 0
    db.close()


def test_ingest_hk_reconciliation_preserves_existing_snapshot_baseline_on_rerun():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()

    rows_a = [
        {"symbol": "700", "exchange": "SEHK", "name": "Tencent"},
    ]
    rows_b = [
        {"symbol": "700", "exchange": "SEHK", "name": "Tencent Holdings"},
    ]

    first_a = stock_universe_service.ingest_hk_snapshot_rows(
        db,
        rows=rows_a,
        source_name="hkex_official",
        snapshot_id="hk-20260412-a",
    )
    stock_universe_service.ingest_hk_snapshot_rows(
        db,
        rows=rows_b,
        source_name="hkex_official",
        snapshot_id="hk-20260412-b",
    )
    second_a = stock_universe_service.ingest_hk_snapshot_rows(
        db,
        rows=rows_a,
        source_name="hkex_official",
        snapshot_id="hk-20260412-a",
    )

    run_a = (
        db.query(StockUniverseReconciliationRun)
        .filter(
            StockUniverseReconciliationRun.market == "HK",
            StockUniverseReconciliationRun.snapshot_id == "hk-20260412-a",
        )
        .one()
    )
    run_b = (
        db.query(StockUniverseReconciliationRun)
        .filter(
            StockUniverseReconciliationRun.market == "HK",
            StockUniverseReconciliationRun.snapshot_id == "hk-20260412-b",
        )
        .one()
    )

    assert run_a.previous_snapshot_id is None
    assert first_a["reconciliation"]["previous_snapshot_id"] is None
    assert second_a["reconciliation"]["previous_snapshot_id"] is None
    assert second_a["reconciliation"]["artifact_hash"] == first_a["reconciliation"]["artifact_hash"]
    assert run_b.previous_snapshot_id == "hk-20260412-a"
    db.close()


def test_ingest_hk_reconciliation_preserves_non_null_baseline_on_rerun():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()

    rows_a = [
        {"symbol": "700", "exchange": "SEHK", "name": "Tencent"},
    ]
    rows_b = [
        {"symbol": "700", "exchange": "SEHK", "name": "Tencent Holdings"},
        {"symbol": "5", "exchange": "SEHK", "name": "HSBC"},
    ]
    rows_c = [
        {"symbol": "700", "exchange": "SEHK", "name": "Tencent Holdings Ltd"},
        {"symbol": "16", "exchange": "SEHK", "name": "Sun Hung Kai"},
    ]

    stock_universe_service.ingest_hk_snapshot_rows(
        db,
        rows=rows_a,
        source_name="hkex_official",
        snapshot_id="hk-20260412-a",
    )
    first_b = stock_universe_service.ingest_hk_snapshot_rows(
        db,
        rows=rows_b,
        source_name="hkex_official",
        snapshot_id="hk-20260412-b",
    )
    stock_universe_service.ingest_hk_snapshot_rows(
        db,
        rows=rows_c,
        source_name="hkex_official",
        snapshot_id="hk-20260412-c",
    )
    second_b = stock_universe_service.ingest_hk_snapshot_rows(
        db,
        rows=rows_b,
        source_name="hkex_official",
        snapshot_id="hk-20260412-b",
    )

    run_b = (
        db.query(StockUniverseReconciliationRun)
        .filter(
            StockUniverseReconciliationRun.market == "HK",
            StockUniverseReconciliationRun.snapshot_id == "hk-20260412-b",
        )
        .one()
    )

    assert run_b.previous_snapshot_id == "hk-20260412-a"
    assert second_b["reconciliation"]["previous_snapshot_id"] == "hk-20260412-a"
    assert second_b["reconciliation"]["artifact_hash"] == first_b["reconciliation"]["artifact_hash"]
    db.close()


def test_ingest_hk_snapshot_rows_quarantines_unsafe_deactivation(monkeypatch):
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()

    monkeypatch.setenv("ASIA_UNIVERSE_APPLY_DESTRUCTIVE_ENABLED", "true")
    monkeypatch.setenv("ASIA_RECONCILIATION_QUARANTINE_ENFORCED", "true")
    monkeypatch.setenv("ASIA_RECONCILIATION_MIN_COUNT_HK", "0")
    monkeypatch.setenv("ASIA_RECONCILIATION_MAX_REMOVED_PERCENT", "20")
    monkeypatch.setenv("ASIA_RECONCILIATION_ANOMALY_PERCENT", "90")

    baseline_rows = [
        {"symbol": "700", "exchange": "SEHK", "name": "Tencent"},
        {"symbol": "5", "exchange": "SEHK", "name": "HSBC"},
        {"symbol": "16", "exchange": "SEHK", "name": "Sun Hung Kai"},
    ]
    stock_universe_service.ingest_hk_snapshot_rows(
        db,
        rows=baseline_rows,
        source_name="hkex_official",
        snapshot_id="hk-20260412-a",
    )

    reduced_rows = [
        {"symbol": "700", "exchange": "SEHK", "name": "Tencent"},
    ]
    stats = stock_universe_service.ingest_hk_snapshot_rows(
        db,
        rows=reduced_rows,
        source_name="hkex_official",
        snapshot_id="hk-20260412-b",
    )

    safety = stats["reconciliation"]["safety"]
    hsbc = db.query(StockUniverse).filter(StockUniverse.symbol == "0005.HK").one()
    shk = db.query(StockUniverse).filter(StockUniverse.symbol == "0016.HK").one()

    assert safety["quarantined"] is True
    assert safety["destructive_apply_blocked"] is True
    assert safety["deactivated_count"] == 0
    assert any(breach["gate"] == "max_removed_percent" for breach in safety["gate_breaches"])
    assert safety["alerts"]
    assert hsbc.status == UNIVERSE_STATUS_ACTIVE
    assert hsbc.is_active is True
    assert shk.status == UNIVERSE_STATUS_ACTIVE
    assert shk.is_active is True
    db.close()


def test_ingest_hk_snapshot_rows_applies_safe_deactivation_when_enabled(monkeypatch):
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()

    monkeypatch.setenv("ASIA_UNIVERSE_APPLY_DESTRUCTIVE_ENABLED", "true")
    monkeypatch.setenv("ASIA_RECONCILIATION_QUARANTINE_ENFORCED", "true")
    monkeypatch.setenv("ASIA_RECONCILIATION_MIN_COUNT_HK", "0")
    monkeypatch.setenv("ASIA_RECONCILIATION_MAX_REMOVED_PERCENT", "90")
    monkeypatch.setenv("ASIA_RECONCILIATION_ANOMALY_PERCENT", "90")

    baseline_rows = [
        {"symbol": "700", "exchange": "SEHK", "name": "Tencent"},
        {"symbol": "5", "exchange": "SEHK", "name": "HSBC"},
        {"symbol": "16", "exchange": "SEHK", "name": "Sun Hung Kai"},
    ]
    stock_universe_service.ingest_hk_snapshot_rows(
        db,
        rows=baseline_rows,
        source_name="hkex_official",
        snapshot_id="hk-20260413-a",
    )

    updated_rows = [
        {"symbol": "700", "exchange": "SEHK", "name": "Tencent"},
        {"symbol": "16", "exchange": "SEHK", "name": "Sun Hung Kai"},
    ]
    stats = stock_universe_service.ingest_hk_snapshot_rows(
        db,
        rows=updated_rows,
        source_name="hkex_official",
        snapshot_id="hk-20260413-b",
    )

    safety = stats["reconciliation"]["safety"]
    hsbc = db.query(StockUniverse).filter(StockUniverse.symbol == "0005.HK").one()
    events = (
        db.query(StockUniverseStatusEvent)
        .filter(StockUniverseStatusEvent.symbol == "0005.HK")
        .all()
    )

    assert safety["quarantined"] is False
    assert safety["allow_destructive_apply"] is True
    assert safety["deactivated_count"] == 1
    assert safety["deactivated_symbols"] == ["0005.HK"]
    assert hsbc.status == UNIVERSE_STATUS_INACTIVE_MISSING_SOURCE
    assert hsbc.is_active is False
    assert any(event.new_status == UNIVERSE_STATUS_INACTIVE_MISSING_SOURCE for event in events)
    db.close()


def test_korea_reconciliation_min_count_defaults_to_launch_baseline(monkeypatch):
    monkeypatch.delenv("ASIA_RECONCILIATION_MIN_COUNT_KR", raising=False)

    assert stock_universe_service._min_count_threshold_for_market("KR") == 2526


def test_korea_reconciliation_min_count_can_be_overridden(monkeypatch):
    monkeypatch.setenv("ASIA_RECONCILIATION_MIN_COUNT_KR", "2600")

    assert stock_universe_service._min_count_threshold_for_market("KR") == 2600


def test_get_market_audit_reports_by_market_counts_freshness_and_diff_summary(monkeypatch):
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    monkeypatch.setenv("ASIA_UNIVERSE_AUDIT_STALE_HOURS", "1")

    db.add_all(
        [
            StockUniverse(
                symbol="0700.HK",
                market="HK",
                exchange="SEHK",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                source="hkex_official",
            ),
            StockUniverse(
                symbol="0005.HK",
                market="HK",
                exchange="SEHK",
                is_active=False,
                status=UNIVERSE_STATUS_INACTIVE_MISSING_SOURCE,
                source="hkex_official",
            ),
            StockUniverse(
                symbol="7203.T",
                market="JP",
                exchange="TSE",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                source="jpx_listing",
            ),
            StockUniverse(
                symbol="RELIANCE.NS",
                market="IN",
                exchange="XNSE",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                source="in_reference_bundle",
            ),
            StockUniverse(
                symbol="AAPL",
                market="US",
                exchange="NASDAQ",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                source="finviz",
            ),
        ]
    )
    db.add_all(
        [
            StockUniverseReconciliationRun(
                market="HK",
                source_name="hkex_official",
                snapshot_id="hk-20260412-a",
                previous_snapshot_id=None,
                total_current=2,
                total_previous=3,
                added_count=0,
                removed_count=1,
                changed_count=0,
                unchanged_count=2,
                artifact_hash="a" * 64,
                artifact_json=json.dumps(
                    {
                        "snapshot_rows": [],
                        "safety": {
                            "quarantined": False,
                            "destructive_apply_blocked": False,
                            "gate_breaches": [],
                            "alerts": [],
                        },
                    },
                    sort_keys=True,
                ),
                created_at=datetime.utcnow() - timedelta(hours=2),
            ),
            StockUniverseReconciliationRun(
                market="TW",
                source_name="tw_reference_bundle",
                snapshot_id="tw-20260412-a",
                previous_snapshot_id=None,
                total_current=0,
                total_previous=0,
                added_count=0,
                removed_count=0,
                changed_count=0,
                unchanged_count=0,
                artifact_hash="b" * 64,
                artifact_json=json.dumps(
                    {
                        "snapshot_rows": [],
                        "safety": {
                            "quarantined": True,
                            "destructive_apply_blocked": True,
                            "gate_breaches": [{"gate": "max_removed_percent"}],
                            "alerts": ["TW snapshot gate breach"],
                        },
                    },
                    sort_keys=True,
                ),
                created_at=datetime.utcnow(),
            ),
        ]
    )
    db.commit()

    audit = stock_universe_service.get_market_audit(db)
    hk = audit["by_market"]["HK"]
    india = audit["by_market"]["IN"]
    jp = audit["by_market"]["JP"]
    tw = audit["by_market"]["TW"]
    us = audit["by_market"]["US"]

    assert hk["counts"]["total"] == 2
    assert hk["counts"]["active"] == 1
    assert hk["counts"]["inactive"] == 1
    assert hk["latest_snapshot"]["snapshot_id"] == "hk-20260412-a"
    assert hk["latest_snapshot"]["counts"]["removed"] == 1
    assert hk["latest_snapshot"]["is_stale"] is True
    assert india["snapshot_supported"] is True
    assert india["latest_snapshot"] is None
    assert jp["snapshot_supported"] is True
    assert jp["latest_snapshot"] is None
    assert tw["latest_snapshot"]["safety"]["quarantined"] is True
    assert us["snapshot_supported"] is False
    assert us["latest_snapshot"] is None
    assert audit["checks"]["stale_after_hours"] == 1
    assert set(audit["checks"]["stale_markets"]) == {"HK", "IN", "JP"}
    assert audit["checks"]["missing_snapshot_markets"] == ["IN", "JP"]
    assert audit["checks"]["quarantined_markets"] == ["TW"]
    db.close()


def test_get_stats_includes_market_audit_summary(monkeypatch):
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    monkeypatch.setenv("ASIA_UNIVERSE_AUDIT_STALE_HOURS", "12")

    db.add(
        StockUniverse(
            symbol="0700.HK",
            market="HK",
            exchange="SEHK",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
            source="hkex_official",
        )
    )
    db.commit()

    stats = stock_universe_service.get_stats(db)

    assert "by_market" in stats
    assert "market_checks" in stats
    assert stats["by_market"]["HK"]["counts"]["total"] == 1
    assert stats["market_checks"]["has_stale_markets"] is True
    db.close()


def test_ingest_from_csv_auto_snapshot_ids_are_collision_resistant():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()

    hk_csv = "symbol,name,exchange,sector,industry,market_cap\n0700,Tencent,SEHK,Technology,Internet,500B\n"
    jp_csv = "symbol,name,exchange,sector,industry,market_cap\n7203,Toyota,TSE,Consumer,Auto,250B\n"
    tw_csv = "symbol,name,exchange,sector,industry,market_cap\n2330,TSMC,TWSE,Technology,Semiconductors,650B\n"

    hk_first = stock_universe_service.ingest_hk_from_csv(db, hk_csv, source_name="hk_manual_csv")
    hk_second = stock_universe_service.ingest_hk_from_csv(db, hk_csv, source_name="hk_manual_csv")
    jp_first = stock_universe_service.ingest_jp_from_csv(db, jp_csv, source_name="jp_manual_csv")
    jp_second = stock_universe_service.ingest_jp_from_csv(db, jp_csv, source_name="jp_manual_csv")
    tw_first = stock_universe_service.ingest_tw_from_csv(db, tw_csv, source_name="tw_manual_csv")
    tw_second = stock_universe_service.ingest_tw_from_csv(db, tw_csv, source_name="tw_manual_csv")

    assert hk_first["snapshot_id"] != hk_second["snapshot_id"]
    assert jp_first["snapshot_id"] != jp_second["snapshot_id"]
    assert tw_first["snapshot_id"] != tw_second["snapshot_id"]
    assert hk_first["snapshot_id"].startswith("hk:")
    assert jp_first["snapshot_id"].startswith("jp:")
    assert tw_first["snapshot_id"].startswith("tw:")
    db.close()


def test_ingest_hk_snapshot_rows_does_not_emit_status_events_for_unchanged_active_rows():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    rows = [{"symbol": "700", "exchange": "SEHK", "name": "Tencent"}]

    stock_universe_service.ingest_hk_snapshot_rows(
        db,
        rows=rows,
        source_name="hkex_official",
        snapshot_id="hk-20260414-a",
    )
    initial_events = db.query(StockUniverseStatusEvent).filter(
        StockUniverseStatusEvent.symbol == "0700.HK"
    ).count()

    stock_universe_service.ingest_hk_snapshot_rows(
        db,
        rows=rows,
        source_name="hkex_official",
        snapshot_id="hk-20260414-a",
    )
    after_rerun_events = db.query(StockUniverseStatusEvent).filter(
        StockUniverseStatusEvent.symbol == "0700.HK"
    ).count()

    assert initial_events == 1
    assert after_rerun_events == 1
    db.close()


def test_ingest_hk_from_csv_merges_duplicate_rows_to_keep_richer_metadata():
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    csv_content = "\n".join(
        [
            "symbol,name,exchange,sector,industry,market_cap",
            "700,,SEHK,Technology,Internet,500B",
            "0700.HK,Tencent Holdings,,Technology,Internet,",
        ]
    )

    stats = stock_universe_service.ingest_hk_from_csv(
        db,
        csv_content,
        source_name="hk_manual_csv",
        snapshot_id="hk-20260414-dup",
    )
    row = db.query(StockUniverse).filter(StockUniverse.symbol == "0700.HK").one()

    assert stats["total"] == 1
    assert row.name == "Tencent Holdings"
    db.close()
