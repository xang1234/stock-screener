"""Integration tests for scan result enrichment during persistence."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database import Base
from app.infra.db.repositories.scan_result_repo import SqlScanResultRepository
from app.models.industry import IBDGroupRank, IBDIndustryGroup
from app.models.scan_result import ScanResult
from app.models.stock import StockFundamental, StockIndustry
from app.models.stock_universe import StockUniverse
from app.services.market_taxonomy_service import MarketTaxonomyEntry


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


def test_persist_orchestrator_results_enriches_non_us_market_taxonomy(session: Session):
    session.add(
        StockUniverse(
            symbol="0700.HK",
            market="HK",
            exchange="XHKG",
            currency="HKD",
            timezone="Asia/Hong_Kong",
        )
    )
    session.commit()

    class _FakeTaxonomyService:
        def get(self, symbol, *, market=None, exchange=None):  # noqa: ARG002
            if symbol == "0700.HK" and market == "HK":
                return MarketTaxonomyEntry(
                    market="HK",
                    symbol="0700.HK",
                    industry_group="Internet Services",
                    themes=("AI Infrastructure", "Cloud"),
                )
            return None

    class _FakeMarketGroupRankingService:
        def get_current_rank_map(self, db, *, market, calculation_date=None):  # noqa: ARG002
            assert market == "HK"
            return {"Internet Services": 4}

    repo = SqlScanResultRepository(
        session,
        taxonomy_service=_FakeTaxonomyService(),
        market_group_ranking_service=_FakeMarketGroupRankingService(),
    )
    repo.persist_orchestrator_results("scan-hk-1", [("0700.HK", _base_raw_result())])

    row = (
        session.query(ScanResult)
        .filter(ScanResult.scan_id == "scan-hk-1", ScanResult.symbol == "0700.HK")
        .one()
    )

    assert row.ibd_industry_group == "Internet Services"
    assert row.ibd_group_rank == 4
    assert row.details["market_themes"] == ["AI Infrastructure", "Cloud"]
