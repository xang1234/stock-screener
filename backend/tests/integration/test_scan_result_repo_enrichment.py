"""Integration tests for scan result enrichment during persistence."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database import Base
from app.infra.db.repositories.scan_result_repo import SqlScanResultRepository
from app.models.industry import IBDGroupRank, IBDIndustryGroup
from app.models.scan_result import Scan, ScanResult
from app.models.stock import StockFundamental, StockIndustry


@pytest.fixture
def session():
    """Create an isolated in-memory DB with full schema."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as sess:
        yield sess


def _base_raw_result() -> dict:
    """Return a minimal orchestrator-like result dict."""
    return {
        "composite_score": 82.5,
        "rating": "Buy",
        "current_price": 123.45,
        "screeners_run": ["minervini", "ipo"],
        "details": {"screeners": {}},
    }


def test_persist_orchestrator_results_enriches_reference_fields(session: Session):
    session.add(
        StockFundamental(
            symbol="AAPL",
            eps_rating=88,
            ipo_date=date(2010, 6, 29),
            sector=None,  # ensure stock_industry fallback is used
            industry=None,
        )
    )
    session.add(
        StockIndustry(
            symbol="AAPL",
            sector="Technology",
            industry="Consumer Electronics",
        )
    )
    session.add(
        IBDIndustryGroup(
            symbol="AAPL",
            industry_group="Computer-Hardware/Peripherals",
        )
    )
    session.add_all(
        [
            IBDGroupRank(
                industry_group="Computer-Hardware/Peripherals",
                date=date(2026, 2, 18),
                rank=11,
                avg_rs_rating=75.0,
            ),
            IBDGroupRank(
                industry_group="Computer-Hardware/Peripherals",
                date=date(2026, 2, 19),
                rank=5,
                avg_rs_rating=80.0,
            ),
        ]
    )
    session.commit()

    repo = SqlScanResultRepository(session)
    repo.persist_orchestrator_results("scan-1", [("AAPL", _base_raw_result())])

    row = (
        session.query(ScanResult)
        .filter(ScanResult.scan_id == "scan-1", ScanResult.symbol == "AAPL")
        .one()
    )

    assert row.eps_rating == 88
    assert row.ipo_date == "2010-06-29"
    assert row.gics_sector == "Technology"
    assert row.gics_industry == "Consumer Electronics"
    assert row.ibd_industry_group == "Computer-Hardware/Peripherals"
    assert row.ibd_group_rank == 5


def test_persist_orchestrator_results_uses_ipo_screener_date_fallback(session: Session):
    # No fundamentals row on purpose, so IPO date must come from ipo screener details.
    session.add(StockIndustry(symbol="MSFT", sector="Technology", industry="Software"))
    session.commit()

    raw = _base_raw_result()
    raw["details"] = {
        "screeners": {
            "ipo": {
                "details": {
                    "ipo_date": "1986-03-13",
                }
            }
        }
    }

    repo = SqlScanResultRepository(session)
    repo.persist_orchestrator_results("scan-2", [("MSFT", raw)])

    row = (
        session.query(ScanResult)
        .filter(ScanResult.scan_id == "scan-2", ScanResult.symbol == "MSFT")
        .one()
    )

    assert row.ipo_date == "1986-03-13"
    assert row.gics_sector == "Technology"
    assert row.gics_industry == "Software"


def test_backfill_ibd_metadata_for_existing_scan_rows(session: Session):
    session.add(Scan(scan_id="scan-us-1", status="completed", universe_market="US"))
    session.add_all(
        [
            ScanResult(
                scan_id="scan-us-1",
                symbol="AAPL",
                composite_score=80.0,
                rating="Buy",
                details={},
                ibd_industry_group=None,
                ibd_group_rank=None,
            ),
            ScanResult(
                scan_id="scan-us-1",
                symbol="MSFT",
                composite_score=79.0,
                rating="Watch",
                details={},
                ibd_industry_group=None,
                ibd_group_rank=None,
            ),
        ]
    )
    session.add_all(
        [
            IBDIndustryGroup(
                symbol="AAPL",
                industry_group="Computer-Hardware/Peripherals",
            ),
            IBDIndustryGroup(
                symbol="MSFT",
                industry_group="Software",
            ),
            IBDGroupRank(
                industry_group="Computer-Hardware/Peripherals",
                date=date(2026, 2, 19),
                rank=5,
                avg_rs_rating=80.0,
            ),
            IBDGroupRank(
                industry_group="Software",
                date=date(2026, 2, 19),
                rank=9,
                avg_rs_rating=74.0,
            ),
        ]
    )
    session.commit()

    repo = SqlScanResultRepository(session)
    stats = repo.backfill_ibd_metadata_for_scan("scan-us-1")

    rows = (
        session.query(ScanResult)
        .filter(ScanResult.scan_id == "scan-us-1")
        .order_by(ScanResult.symbol.asc())
        .all()
    )

    assert stats["updated_rows"] == 2
    assert rows[0].symbol == "AAPL"
    assert rows[0].ibd_industry_group == "Computer-Hardware/Peripherals"
    assert rows[0].ibd_group_rank == 5
    assert rows[1].symbol == "MSFT"
    assert rows[1].ibd_industry_group == "Software"
    assert rows[1].ibd_group_rank == 9
