from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database import Base
from app.infra.db.models.feature_store import FeatureRun
from app.services.market_group_ranking_service import MarketGroupRankingService
from app.services.group_ranking_history import select_market_run_series
from app.services.rrg_history_provider import build_rrg_history_provider


class _FakeStoredGroupRankService:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def get_current_rankings(
        self,
        db,
        limit=197,
        calculation_date=None,
        *,
        market="US",
    ):  # noqa: ANN001
        self.calls.append(("current", db, market, limit, calculation_date))
        return [
            {
                "industry_group": "Internet Services",
                "date": "2026-04-04",
                "rank": 4,
                "avg_rs_rating": 81.0,
                "avg_rs_rating_1m": 44.0,
                "avg_rs_rating_3m": 72.0,
            },
            {
                "industry_group": "Software",
                "date": "2026-04-04",
                "rank": 7,
                "avg_rs_rating": 76.0,
                "avg_rs_rating_1m": 51.0,
                "avg_rs_rating_3m": 68.0,
            },
        ]

    def get_rank_movers(
        self,
        db,
        period="1w",
        limit=20,
        calculation_date=None,
        *,
        market="US",
    ):  # noqa: ANN001
        self.calls.append(("movers", db, market, period, limit, calculation_date))
        return {
            "period": period,
            "gainers": [{"industry_group": "Internet Services"}],
            "losers": [{"industry_group": "Software"}],
        }

    def get_group_history(
        self,
        db,
        industry_group,
        days=180,
        *,
        market="US",
    ):  # noqa: ANN001
        self.calls.append(("history", db, market, industry_group, days))
        return {
            "industry_group": industry_group,
            "current_rank": 4,
            "current_avg_rs": 81.0,
            "current_avg_rs_1m": 44.0,
            "current_avg_rs_3m": 72.0,
            "history": [],
            "stocks": [],
        }


def test_get_rank_movers_delegates_to_stored_versioned_service():
    stored = _FakeStoredGroupRankService()
    service = MarketGroupRankingService(group_rank_service=stored)
    db = Session()
    as_of = date(2026, 4, 4)

    movers = service.get_rank_movers(
        db,
        market="hk",
        period="1m",
        limit=2,
        calculation_date=as_of,
    )

    assert movers["gainers"] == [{"industry_group": "Internet Services"}]
    assert stored.calls == [("movers", db, "HK", "1m", 2, as_of)]


def test_select_market_run_series_honors_min_runs_without_cutoff():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[FeatureRun.__table__])

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

        market_runs = select_market_run_series(
            db,
            market="HK",
            latest_run=runs[0],
            cutoff_date=None,
            min_runs=2,
        )

    assert [run.id for run in market_runs] == [10, 9]


def test_current_rank_snapshot_uses_stored_versioned_rankings():
    stored = _FakeStoredGroupRankService()
    service = MarketGroupRankingService(group_rank_service=stored)
    db = Session()

    snapshot = service.get_current_rank_snapshot(db, market="hk")

    assert snapshot.date == "2026-04-04"
    assert snapshot.ranks_by_group == {"Internet Services": 4, "Software": 7}
    assert stored.calls == [("current", db, "HK", 10_000, None)]


def test_group_history_delegates_to_stored_versioned_service():
    stored = _FakeStoredGroupRankService()
    service = MarketGroupRankingService(group_rank_service=stored)
    db = Session()

    detail = service.get_group_history(
        db,
        market="hk",
        industry_group="Internet Services",
        days=90,
    )

    assert detail["current_avg_rs_1m"] == 44.0
    assert detail["current_avg_rs_3m"] == 72.0
    assert stored.calls == [("history", db, "HK", "Internet Services", 90)]


def test_rrg_history_builder_uses_stored_group_service_for_non_us():
    calls: list[tuple] = []

    class _GroupRankService:
        def get_current_rankings(
            self,
            db,
            limit=197,
            calculation_date=None,
            *,
            market,
            formula_version,
        ):  # noqa: ANN001
            calls.append(
                ("current", db, market, limit, calculation_date, formula_version)
            )
            return []

    class _MarketRsRepository:
        def active_formula(self, db, *, market):  # noqa: ANN001
            calls.append(("formula", db, market))
            return "balanced-horizon-percentile-v2"

    provider = build_rrg_history_provider(
        group_rank_service=_GroupRankService(),
        market_rs_repository=_MarketRsRepository(),
    )

    db = Session()
    as_of = date(2026, 4, 3)
    assert provider.get_all_groups_history(
        db,
        market="HK",
        days=400,
        as_of_date=as_of,
    ) == (
        None,
        {},
        {},
    )
    assert calls == [
        ("formula", db, "HK"),
        ("current", db, "HK", 197, as_of, "balanced-horizon-percentile-v2"),
    ]
