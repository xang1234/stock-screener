from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database import Base
from app.infra.db.models.feature_store import FeatureRun
from app.services.market_group_ranking_service import MarketGroupRankingService


def test_get_rank_movers_separates_gainers_and_losers(monkeypatch):
    service = MarketGroupRankingService()

    monkeypatch.setattr(
        service,
        "get_current_rankings",
        lambda db, *, market, limit=10_000, calculation_date=None: [  # noqa: ARG005
            {"industry_group": "Positive A", "rank_change_1w": 4},
            {"industry_group": "Positive B", "rank_change_1w": 1},
            {"industry_group": "Negative A", "rank_change_1w": -2},
            {"industry_group": "Negative B", "rank_change_1w": -5},
            {"industry_group": "Flat", "rank_change_1w": 0},
        ],
    )

    movers = service.get_rank_movers(Session(), market="HK", period="1w", limit=2)

    assert [row["industry_group"] for row in movers["gainers"]] == ["Positive A", "Positive B"]
    assert [row["industry_group"] for row in movers["losers"]] == ["Negative B", "Negative A"]


def test_get_market_run_series_honors_min_runs_without_cutoff():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[FeatureRun.__table__])
    service = MarketGroupRankingService()

    with Session(engine) as db:
        runs = [
            FeatureRun(
                id=10,
                as_of_date=date(2026, 4, 4),
                run_type="daily_snapshot",
                status="published",
                published_at=datetime(2026, 4, 4, 21, 30, 0),
                config_json={"universe": {"market": "HK"}},
            ),
            FeatureRun(
                id=9,
                as_of_date=date(2026, 4, 3),
                run_type="daily_snapshot",
                status="published",
                published_at=datetime(2026, 4, 3, 21, 30, 0),
                config_json={"universe": {"market": "HK"}},
            ),
            FeatureRun(
                id=8,
                as_of_date=date(2026, 4, 2),
                run_type="daily_snapshot",
                status="published",
                published_at=datetime(2026, 4, 2, 21, 30, 0),
                config_json={"universe": {"market": "HK"}},
            ),
            FeatureRun(
                id=7,
                as_of_date=date(2026, 4, 1),
                run_type="daily_snapshot",
                status="published",
                published_at=datetime(2026, 4, 1, 21, 30, 0),
                config_json={"universe": {"market": "JP"}},
            ),
        ]
        db.add_all(runs)
        db.commit()

        market_runs = service._get_market_run_series(  # noqa: SLF001
            db,
            market="HK",
            latest_run=runs[0],
            cutoff_date=None,
            min_runs=2,
        )

    assert [run.id for run in market_runs] == [10, 9]
