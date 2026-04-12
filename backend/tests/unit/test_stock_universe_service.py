from __future__ import annotations

import json

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
)
from app.services.stock_universe_service import StockUniverseService

stock_universe_service = StockUniverseService()


def _make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return TestingSessionLocal


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
    assert len(payload["row_hash"]) == 64
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
