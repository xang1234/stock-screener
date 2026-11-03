from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.domain.universe.ingestion import (
    CanonicalUniverseIngestionResult,
    CanonicalUniverseRow,
    RejectedUniverseRow,
    UniverseIndustryTaxonomy,
    UniverseIngestionSideEffects,
    UniverseLifecycleMetadata,
    UniverseSourceProvenance,
)
from app.models.stock_universe import (
    StockUniverse,
    StockUniverseStatusEvent,
    UNIVERSE_EVENT_LISTING_TIER_CHANGED,
    UNIVERSE_EVENT_STATUS_CHANGED,
    UNIVERSE_STATUS_ACTIVE,
    UNIVERSE_STATUS_INACTIVE_MANUAL,
)
from app.services.stock_universe_service import StockUniverseService
from app.services.universe_ingestion_pipeline import (
    FlatUniverseCanonicalizerAdapter,
    UniverseIngestionPipeline,
    UniversePersistence,
)


class _FakeCanonicalizer:
    def __init__(self, result: CanonicalUniverseIngestionResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def canonicalize_rows(self, rows, **kwargs):
        self.calls.append({"rows": list(rows), **kwargs})
        return self.result


class _FakeFlatCanonicalizer:
    def canonicalize_rows(self, rows, **kwargs):
        return SimpleNamespace(
            canonical_rows=(
                SimpleNamespace(
                    symbol="0700.HK",
                    name="Tencent",
                    market="HK",
                    exchange="XHKG",
                    currency="HKD",
                    timezone="Asia/Hong_Kong",
                    local_code="0700",
                    sector="Communication Services",
                    industry="Internet",
                    market_cap=100.0,
                    source_name="hkex_official",
                    source_symbol="700",
                    source_row_number=1,
                    snapshot_id="hk-2026-05-29",
                    snapshot_as_of="2026-05-29",
                    source_metadata={"row_counts": {"xhkg": 1}},
                    lineage_hash="lineage-0700",
                    row_hash="row-0700",
                ),
            ),
            rejected_rows=(
                SimpleNamespace(
                    source_row_number=2,
                    source_symbol="BAD",
                    reason="invalid symbol",
                ),
            ),
        )


def _make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


def test_flat_canonicalizer_adapter_returns_shared_ingestion_models() -> None:
    adapter = FlatUniverseCanonicalizerAdapter(_FakeFlatCanonicalizer())

    result = adapter.canonicalize_rows(
        [{"symbol": "700"}, {"symbol": "BAD"}],
        source_name="hkex_official",
        snapshot_id="hk-2026-05-29",
        snapshot_as_of="2026-05-29",
        source_metadata={"row_counts": {"xhkg": 1}},
    )

    assert isinstance(result, CanonicalUniverseIngestionResult)
    assert isinstance(result.canonical_rows[0], CanonicalUniverseRow)
    assert result.canonical_rows[0].mic == "XHKG"
    assert result.canonical_rows[0].provenance.source_name == "hkex_official"
    assert result.canonical_rows[0].provenance.row_hash == "row-0700"
    assert isinstance(result.rejected_rows[0], RejectedUniverseRow)


def test_flat_canonicalizer_adapter_resolves_exchange_alias_to_catalog_mic() -> None:
    row = FlatUniverseCanonicalizerAdapter.canonical_row_from_flat(
        SimpleNamespace(
            symbol="3008.TWO",
            name="Largan Precision",
            market="TW",
            exchange="TPEX",
            currency="TWD",
            timezone="Asia/Taipei",
            local_code="3008",
            sector="Technology",
            industry="Electronics",
            market_cap=100.0,
            source_name="tw_reference_bundle",
            source_symbol="3008.TWO",
            source_row_number=1,
            snapshot_id="tw-2026-05-29",
            snapshot_as_of="2026-05-29",
            source_metadata={},
            lineage_hash="lineage-3008",
            row_hash="row-3008",
        )
    )

    assert row.mic == "XTAI"
    assert row.provenance.source_metadata["source_exchange"] == "TPEX"


def test_flat_canonicalizer_adapter_keeps_taxonomy_out_of_provenance() -> None:
    adapter = FlatUniverseCanonicalizerAdapter(
        SimpleNamespace(
            canonicalize_rows=lambda rows, **kwargs: SimpleNamespace(
                canonical_rows=(
                    SimpleNamespace(
                        symbol="600519.SS",
                        name="Kweichow Moutai",
                        market="CN",
                        exchange="SSE",
                        board="SSE_MAIN",
                        currency="CNY",
                        timezone="Asia/Shanghai",
                        local_code="600519",
                        sector="Consumer Staples",
                        industry_group="Food & Beverage",
                        industry="Beverages",
                        sub_industry="Liquor",
                        market_cap=100.0,
                        source_name="cn_reference_bundle",
                        source_symbol="600519",
                        source_row_number=1,
                        snapshot_id="cn-2026-05-29",
                        snapshot_as_of="2026-05-29",
                        source_metadata={},
                        lineage_hash="lineage-600519",
                        row_hash="row-600519",
                    ),
                ),
                rejected_rows=(),
            )
        )
    )

    result = adapter.canonicalize_rows(
        [{"symbol": "600519"}],
        source_name="cn_reference_bundle",
        snapshot_id="cn-2026-05-29",
    )

    assert "industry_group" not in result.canonical_rows[0].provenance.source_metadata
    assert result.side_effects.industry_taxonomy_rows == (
        UniverseIndustryTaxonomy(
            symbol="600519.SS",
            sector="Consumer Staples",
            industry_group="Food & Beverage",
            industry="Beverages",
            sub_industry="Liquor",
        ),
    )


def _row(
    symbol: str,
    *,
    local_code: str,
    listing_tier: str | None = None,
    industry: str = "Banking",
    source_row_number: int = 1,
    lifecycle: UniverseLifecycleMetadata | None = None,
) -> CanonicalUniverseRow:
    return CanonicalUniverseRow(
        symbol=symbol,
        name=f"{symbol} name",
        market="SG",
        mic="XSES",
        local_code=local_code,
        currency="SGD",
        timezone="Asia/Singapore",
        listing_tier=listing_tier,
        sector="Banks",
        industry=industry,
        market_cap=100.0,
        provenance=UniverseSourceProvenance(
            source_name="sgx_official",
            snapshot_id="sgx-2026-05-29",
            snapshot_as_of="2026-05-29",
            source_symbol=local_code,
            source_row_number=source_row_number,
            source_metadata={"row_counts": {"xses": 2}},
            lineage_hash=f"lineage-{local_code}",
            row_hash=f"row-{local_code}",
        ),
        lifecycle=lifecycle or UniverseLifecycleMetadata.active(),
    )


def test_pipeline_persists_rows_reconciliation_and_listing_tier_audit() -> None:
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    db.add(
        StockUniverse(
            symbol="D05.SI",
            name="DBS old",
            market="SG",
            exchange="XSES",
            currency="SGD",
            timezone="Asia/Singapore",
            local_code="D05",
            listing_tier="mainboard",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
            source="sg_ingest",
        )
    )
    db.commit()

    canonicalizer = _FakeCanonicalizer(
        CanonicalUniverseIngestionResult(
            canonical_rows=(
                _row("D05.SI", local_code="D05", listing_tier="catalist"),
                _row("O39.SI", local_code="O39", source_row_number=2),
            )
        )
    )
    service = StockUniverseService()
    pipeline = UniverseIngestionPipeline(
        canonicalizers={"SG": canonicalizer},
        persistence=UniversePersistence.for_stock_universe_service(service),
    )

    stats = pipeline.ingest_snapshot_rows(
        db,
        market="SG",
        rows=[{"symbol": "D05"}, {"symbol": "O39"}],
        source_name="sgx_official",
        snapshot_id="sgx-2026-05-29",
        snapshot_as_of="2026-05-29",
        source_metadata={"row_counts": {"xses": 2}},
        strict=True,
    )

    assert canonicalizer.calls[0]["source_name"] == "sgx_official"
    assert stats["added"] == 1
    assert stats["updated"] == 1
    assert stats["total"] == 2
    assert stats["rejected"] == 0
    assert stats["canonical_rows"][0]["exchange"] == "XSES"
    assert stats["canonical_rows"][0]["source_symbol"] == "D05"
    assert stats["reconciliation"]["counts"]["added"] == 2

    existing = db.query(StockUniverse).filter_by(symbol="D05.SI").one()
    added = db.query(StockUniverse).filter_by(symbol="O39.SI").one()
    assert existing.listing_tier == "catalist"
    assert added.exchange == "XSES"
    assert added.listing_tier is None

    events = db.query(StockUniverseStatusEvent).order_by(
        StockUniverseStatusEvent.id.asc()
    ).all()
    assert [event.event_type for event in events] == [
        UNIVERSE_EVENT_LISTING_TIER_CHANGED,
        UNIVERSE_EVENT_STATUS_CHANGED,
    ]
    tier_payload = json.loads(events[0].payload_json)
    assert tier_payload["previous"] == "mainboard"
    assert tier_payload["current"] == "catalist"
    assert tier_payload["snapshot_id"] == "sgx-2026-05-29"
    db.close()


def test_pipeline_strict_mode_raises_for_rejected_rows() -> None:
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    canonicalizer = _FakeCanonicalizer(
        CanonicalUniverseIngestionResult(
            rejected_rows=(
                RejectedUniverseRow(
                    source_row_number=1,
                    source_symbol="BAD",
                    reason="Invalid SG symbol",
                ),
            )
        )
    )
    pipeline = UniverseIngestionPipeline(
        canonicalizers={"SG": canonicalizer},
        persistence=UniversePersistence.for_stock_universe_service(
            StockUniverseService()
        ),
    )

    with pytest.raises(ValueError, match="SG ingestion rejected 1 row"):
        pipeline.ingest_snapshot_rows(
            db,
            market="SG",
            rows=[{"symbol": "BAD"}],
            source_name="sgx_official",
            snapshot_id="sgx-2026-05-29",
            strict=True,
        )
    db.close()


def test_pipeline_strict_mode_allows_non_blocking_rejected_rows() -> None:
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    canonicalizer = _FakeCanonicalizer(
        CanonicalUniverseIngestionResult(
            rejected_rows=(
                RejectedUniverseRow(
                    source_row_number=1,
                    source_symbol="500002.BO",
                    reason="missing_yfinance_price_coverage",
                    strict=False,
                ),
            ),
            side_effects=UniverseIngestionSideEffects(),
        )
    )
    pipeline = UniverseIngestionPipeline(
        canonicalizers={"IN": canonicalizer},
        persistence=UniversePersistence.for_stock_universe_service(
            StockUniverseService()
        ),
    )

    stats = pipeline.ingest_snapshot_rows(
        db,
        market="IN",
        rows=[{"symbol": "500002.BO"}],
        source_name="in_reference_bundle",
        snapshot_id="in-2026-05-29",
        strict=True,
    )

    assert stats["rejected"] == 1
    db.close()


def test_pipeline_persists_canonical_lifecycle_for_inactive_rows() -> None:
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    deactivated_at = datetime(2026, 5, 29, 4, 30)
    canonicalizer = _FakeCanonicalizer(
        CanonicalUniverseIngestionResult(
            canonical_rows=(
                _row(
                    "D05.SI",
                    local_code="D05",
                    lifecycle=UniverseLifecycleMetadata.inactive(
                        status=UNIVERSE_STATUS_INACTIVE_MANUAL,
                        reason="Source marks row inactive",
                        deactivated_at=deactivated_at,
                    ),
                ),
            )
        )
    )
    pipeline = UniverseIngestionPipeline(
        canonicalizers={"SG": canonicalizer},
        persistence=UniversePersistence.for_stock_universe_service(
            StockUniverseService()
        ),
    )

    stats = pipeline.ingest_snapshot_rows(
        db,
        market="SG",
        rows=[{"symbol": "D05"}],
        source_name="sgx_official",
        snapshot_id="sgx-2026-05-29",
        strict=True,
    )

    assert stats["added"] == 1
    row = db.query(StockUniverse).filter_by(symbol="D05.SI").one()
    assert row.status == UNIVERSE_STATUS_INACTIVE_MANUAL
    assert row.is_active is False
    assert row.status_reason == "Source marks row inactive"
    assert row.deactivated_at == deactivated_at

    event = db.query(StockUniverseStatusEvent).one()
    assert event.event_type == UNIVERSE_EVENT_STATUS_CHANGED
    assert event.old_status is None
    assert event.new_status == UNIVERSE_STATUS_INACTIVE_MANUAL
    db.close()


def test_pipeline_preserves_existing_listing_tier_when_source_omits_tier() -> None:
    TestingSessionLocal = _make_session()
    db = TestingSessionLocal()
    db.add(
        StockUniverse(
            symbol="D05.SI",
            name="DBS old",
            market="SG",
            exchange="XSES",
            currency="SGD",
            timezone="Asia/Singapore",
            local_code="D05",
            listing_tier="mainboard",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
            source="sg_ingest",
        )
    )
    db.commit()
    canonicalizer = _FakeCanonicalizer(
        CanonicalUniverseIngestionResult(
            canonical_rows=(
                _row("D05.SI", local_code="D05", listing_tier=None),
            )
        )
    )
    pipeline = UniverseIngestionPipeline(
        canonicalizers={"SG": canonicalizer},
        persistence=UniversePersistence.for_stock_universe_service(
            StockUniverseService()
        ),
    )

    pipeline.ingest_snapshot_rows(
        db,
        market="SG",
        rows=[{"symbol": "D05"}],
        source_name="sgx_official",
        snapshot_id="sgx-2026-05-29",
        strict=True,
    )

    row = db.query(StockUniverse).filter_by(symbol="D05.SI").one()
    assert row.listing_tier == "mainboard"
    events = db.query(StockUniverseStatusEvent).all()
    assert events == []
    db.close()


def _ingest_sg(db, *rows: CanonicalUniverseRow) -> None:
    pipeline = UniverseIngestionPipeline(
        canonicalizers={
            "SG": _FakeCanonicalizer(CanonicalUniverseIngestionResult(canonical_rows=rows))
        },
        persistence=UniversePersistence.for_stock_universe_service(
            StockUniverseService()
        ),
    )
    pipeline.ingest_snapshot_rows(
        db,
        market="SG",
        rows=[{"symbol": row.local_code} for row in rows],
        source_name="sgx_official",
        snapshot_id="sgx-2026-05-29",
        strict=True,
    )


def _common_stock_flags(db) -> dict[str, bool]:
    return {
        row.symbol: row.is_common_stock
        for row in db.query(StockUniverse).order_by(StockUniverse.symbol)
    }


def test_pipeline_marks_new_fund_rows_not_common_stock() -> None:
    db = _make_session()()

    _ingest_sg(
        db,
        _row("ES3.SI", local_code="ES3", industry="Exchange Traded Fund"),
        _row("CEF.SI", local_code="CEF", industry="Closed-End Fund - Debt", source_row_number=2),
        _row("D05.SI", local_code="D05", source_row_number=3),
        _row("A17U.SI", local_code="A17U", industry="REIT - Industrial", source_row_number=4),
    )

    assert _common_stock_flags(db) == {
        "A17U.SI": True,
        "CEF.SI": False,
        "D05.SI": True,
        "ES3.SI": False,
    }
    db.close()


def test_pipeline_refresh_derives_common_stock_from_the_merged_industry() -> None:
    db = _make_session()()
    db.add_all(
        [
            StockUniverse(
                symbol=symbol,
                market="SG",
                industry=industry,
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                is_common_stock=is_common_stock,
                source=source,
            )
            for symbol, industry, is_common_stock, source in [
                ("ES3.SI", "Exchange Traded Fund", True, "sg_ingest"),
                ("C2PU.SI", "Closed-End Fund - Equity", False, "sg_ingest"),
                ("D05.SI", None, False, "manual"),
            ]
        ]
    )
    db.commit()

    _ingest_sg(
        db,
        _row("ES3.SI", local_code="ES3", industry="Exchange Traded Fund"),
        _row("C2PU.SI", local_code="C2PU", source_row_number=2),
        _row("D05.SI", local_code="D05", source_row_number=3),
    )

    assert _common_stock_flags(db) == {"C2PU.SI": True, "D05.SI": True, "ES3.SI": False}
    db.close()
